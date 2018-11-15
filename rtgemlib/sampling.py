from tqdm import tqdm
import networkx as nx
import pandas as pd
import numpy as np

from .rtgem import RTGEM
from .parentCount import get_parents_count_dfs, getParentsCountVector


def all_parents_were_sampled(tgem, scc):
    nd = scc[0]
    preds = [pa for pa in tgem.dpd_graph.predecessors(nd) if pa != nd]
    num_sampled_pa = np.array(
        [tgem.dpd_graph.node[pa]['sampled'] * 1 for pa in preds]).sum()

    return num_sampled_pa == len(preds) - (len(scc) - 1)


def sample_sccs_nodes(tgems, scc_nodes, t_min, t_max):
    t = t_min
    nds_parents_timescales = [
        tgems.get_node_parents_timescales(nd) for nd in scc_nodes]

    nds_lambdas = [tgems.dpd_graph.node[nd]['lambdas'] for nd in scc_nodes]

    for nd in scc_nodes:
        tgems.dpd_graph.node[nd]['timeserie'] = np.array([])

    while t < t_max:
        nds_parents_timeseries = [
            tgems.get_node_parents_timeseries(nd) for nd in scc_nodes]

        nds_parent_count_dfs = [get_parents_count_dfs(nd_parents_timeseries, nd_parents_timescales)
                                for nd_parents_timeseries, nd_parents_timescales in zip(nds_parents_timeseries, nds_parents_timescales)]

        nds_pcvs_t_expirations = np.array([getParentsCountVector(nd_parents_count_dfs, nd_parents_timescales, t)
                                           for nd_parents_count_dfs, nd_parents_timescales in zip(nds_parent_count_dfs, nds_parents_timescales)])

        lambda_t_nds = [nds_lambdas[i][nd_pcv]
                        for i, nd_pcv in enumerate(nds_pcvs_t_expirations[:, 0])]

        Y = np.random.uniform(size=2)
        taus = -np.log(1 - Y) / lambda_t_nds

        min_tau_i = taus.argmin()
        t_change = nds_pcvs_t_expirations[min_tau_i, 1]
        if t + taus[min_tau_i] <= min(t_change, t_max):
            t += taus[min_tau_i]

            nd = scc_nodes[min_tau_i]
            tgems.dpd_graph.node[nd]['timeserie'] = np.append(
                tgems.dpd_graph.node[nd]['timeserie'], t)
        else:
            t = t_change

    for nd in scc_nodes:
        tgems.dpd_graph.node[nd]['sampled'] = True


def sample_child_node(tgems, node, t_min, t_max):
    timescales_parents = [tgems.dpd_graph.get_edge_data(
        pa, node)['timescales'] for pa in tgems.dpd_graph.predecessors(node)]
    t = t_min
    tgems.dpd_graph.node[node]['timeserie'] = np.array([])

    while t < t_max:
        timeseries_parents = np.array([tgems.dpd_graph.node[pa][
                                      'timeserie'] for pa in tgems.dpd_graph.predecessors(node)])
        node_parents_count_dfs = get_parents_count_dfs(
            timeseries_parents, timescales_parents)

        parent_nodes_count, t_lambda_change = getParentsCountVector(
            node_parents_count_dfs, timescales_parents, t)
        lambda_t = tgems.dpd_graph.node[node]['lambdas'][parent_nodes_count]

        y = np.random.uniform()
        waiting_time = -np.log(1 - y) / lambda_t
        if t + waiting_time < min(t_lambda_change, t_max):
            t += waiting_time
            tgems.dpd_graph.node[node]['timeserie'] = np.append(
                tgems.dpd_graph.node[node]['timeserie'], t)
        else:
            t = t_lambda_change

    tgems.dpd_graph.node[node]['sampled'] = True


def sample_from_tgem(tgem, t_min=0, t_max=30):
    for nd in tgem.dpd_graph.nodes():
        tgem.dpd_graph.node[nd]['sampled'] = False
        tgem.dpd_graph.node[nd]['timeserie'] = np.array([])

    sccs = list(nx.strongly_connected_components(tgem.dpd_graph))
    i = -1

    pbar = tqdm(total=len(sccs))

    while len(sccs) > 0:
        i = (i + 1) % len(sccs)
        if all_parents_were_sampled(tgem, list(sccs[i])):
            if len(sccs[i]) == 1:
                sample_child_node(tgem, list(sccs[i])[0], t_min, t_max)
            else:
                sample_sccs_nodes(tgem, list(sccs[i]), t_min, t_max)
            sccs.pop(i)
            pbar.update(1)
    pbar.close()

    sampled_data_df = pd.DataFrame()
    for nd in tgem.dpd_graph.nodes():
        node_sampled_tm = pd.DataFrame(
            data={'time':  tgem.dpd_graph.nodes[nd]['timeserie'], 'event': nd})
        sampled_data_df = pd.concat([sampled_data_df, node_sampled_tm])

    return sampled_data_df.sort_values('time', ascending=True)
