from tqdm.autonotebook import tqdm

import networkx as nx
import pandas as pd
import numpy as np
import math

from .rtgem import RTGEM
from .parentCount import get_parents_count_dfs, getParentsCountVector, init_parent_count_df, count_timescale, updates_parent_count_df


def all_parents_were_sampled(tgem, scc):
    nd = scc[0]
    preds = [pa for pa in tgem.dpd_graph.predecessors(nd) if pa != nd]
    num_sampled_pa = np.array(
        [tgem.dpd_graph.node[pa]['sampled'] * 1 for pa in preds]).sum()

    return num_sampled_pa == len(preds) - (len(scc) - 1)


def sample_sccs_nodes(tgems, scc_nodes, t_min, t_max, pbar):
    t = t_min
    # nds_parents_timescales = [
    #     tgems.get_node_parents_timescales(nd) for nd in scc_nodes]

    # nds_lambdas = [tgems.dpd_graph.node[nd]['lambdas'] for nd in scc_nodes]

    for nd in scc_nodes:
        tgems.dpd_graph.node[nd]['timeserie'] = np.array([])

        for nd_pa in tgems.dpd_graph.predecessors(nd):
            pa_tms = tgems.dpd_graph.edges[(nd_pa, nd)]['timescales']
            pa_ts = tgems.dpd_graph.nodes[nd_pa]['timeserie']
            tgems.dpd_graph.edges[(nd_pa, nd)]['parent_count'] = count_timescale(
                init_parent_count_df(pa_ts, pa_tms), pa_tms)

    # nds_parents_timeseries = [
    #     tgems.get_node_parents_timeseries(nd) for nd in scc_nodes]

    # nds_parent_count_dfs = [get_parents_count_dfs(nd_parents_timeseries, nd_parents_timescales)
    # for nd_parents_timeseries, nd_parents_timescales in
    # zip(nds_parents_timeseries, nds_parents_timescales)]

    # for nd_parents_count_dfs, nd_parents_timescales in
    # zip(nds_parent_count_dfs, nds_parents_timescales)])

    nds_pcvs_t_expirations = np.array(
        [get_parent_configuration(tgems, nd, t) for nd in scc_nodes])

    lambda_t_nds = [tgems.get_lambda(scc_nodes[i], nd_pcv)
                    for i, nd_pcv in enumerate(nds_pcvs_t_expirations[:, 0])]
    progress = 0
    tt = 0
    t_old = t

    while t < t_max:

        Y = np.random.uniform(size=len(scc_nodes))
        taus = -np.log(1 - Y) / lambda_t_nds

        min_tau_i = taus.argmin()
        t_change = min(nds_pcvs_t_expirations[min_tau_i, 1], t_max)

        if t + taus[min_tau_i] <= t_change:
            t += taus[min_tau_i]

            nd = scc_nodes[min_tau_i]
            tgems.dpd_graph.node[nd]['timeserie'] = np.append(
                tgems.dpd_graph.node[nd]['timeserie'], t)

            for nd_child in tgems.dpd_graph.successors(scc_nodes[min_tau_i]):
                if nd_child in scc_nodes:
                    nd_child_id = scc_nodes.index(nd_child)
                    if nd_child_id != -1:
                        tgems.dpd_graph.edges[(nd, nd_child)]['parent_count'] = updates_parent_count_df(
                            tgems.dpd_graph.edges[(nd, nd_child)]['parent_count'], tgems.dpd_graph.edges[(nd, nd_child)]['timescales'], t)

                    nds_pcvs_t_expirations[
                        nd_child_id, :] = get_parent_configuration(tgems, nd_child, t)

                    lambda_t_nds[nd_child_id] = tgems.get_lambda(
                        nd_child, nds_pcvs_t_expirations[nd_child_id, 0])
        else:
            t = t_change

            nds_pcvs_t_expirations[min_tau_i, :] = get_parent_configuration(
                tgems, scc_nodes[min_tau_i], t)

            lambda_t_nds[min_tau_i] = tgems.get_lambda(
                scc_nodes[min_tau_i], nds_pcvs_t_expirations[min_tau_i, 0])

        progress += ((t - t_old) / t_max) * 100

        if (progress >= 1):
            pbar.update(math.floor(progress))
            tt += math.floor(progress)
            progress = 0

        t_old = t

    pbar.update(pbar.total - tt)

    for nd in scc_nodes:
        tgems.dpd_graph.node[nd]['sampled'] = True


def get_parent_configuration(tgems, node, t):
    pcv = ()
    t_expr = np.inf

    for nd_pa in tgems.dpd_graph.predecessors(node):
        nb_pa_tms = len(tgems.dpd_graph.edges[(nd_pa, node)]['timescales'])

        parent_count_df = tgems.dpd_graph.edges[(nd_pa, node)]['parent_count']
        gt_t = (t >= parent_count_df.index.values)
        res = parent_count_df[
            gt_t & (t < parent_count_df.values[:, -1])]

        if res.shape[0] == 0:
            pcv += (0,) * nb_pa_tms
            exp_ = parent_count_df[~gt_t]
            if exp_.shape[0] > 0:
                t_expr = min(t_expr, exp_.iloc[0, -1])
        else:
            pcv += tuple(tm_cnt for tm_cnt in res.iloc[0].values[:nb_pa_tms])
            t_expr = min(t_expr, res.iloc[0].values[-1])

    return pcv, t_expr


def sample_child_node(tgems, node, t_min, t_max, pbar):
    # timeseries_parents = tgems.get_node_parents_timeseries(node)
    # timescales_parents = tgems.get_node_parents_timescales(node)

    # node_parents_count_dfs = get_parents_count_dfs(
    #     timeseries_parents, timescales_parents)

    t = t_min

    # parent_nodes_count, t_lambda_change = getParentsCountVector(
    #     node_parents_count_dfs, timescales_parents, t)
    # t_lambda_change = min(t_lambda_change, t_max)

    tgems.dpd_graph.node[node]['timeserie'] = np.array([])
    for nd_pa in tgems.dpd_graph.predecessors(node):
        pa_tms = tgems.dpd_graph.edges[(nd_pa, node)]['timescales']
        pa_ts = tgems.dpd_graph.nodes[nd_pa]['timeserie']
        tgems.dpd_graph.edges[(nd_pa, node)]['parent_count'] = count_timescale(
            init_parent_count_df(pa_ts, pa_tms), pa_tms)

    t_old = t_min
    progress = 0
    tt = 0
    parent_nodes_count, t_lambda_change = get_parent_configuration(
        tgems, node, t)

    while t < t_max:
        lambda_t = tgems.dpd_graph.node[node]['lambdas'][parent_nodes_count]

        y = np.random.uniform()
        waiting_time = -np.log(1 - y) / lambda_t

        if t + waiting_time < t_lambda_change:
            t += waiting_time
            tgems.dpd_graph.node[node]['timeserie'] = np.append(
                tgems.dpd_graph.node[node]['timeserie'], t)

        else:
            t = t_lambda_change
            parent_nodes_count, t_lambda_change = get_parent_configuration(
                tgems, node, t)

            t_lambda_change = min(t_lambda_change, t_max)

        # print('updating delta : {}'.format(math.ceil((t - t_old) * 1000)))
        progress += ((t - t_old) / t_max) * 100
        if (progress >= 1):
            pbar.update(math.floor(progress))
            tt += math.floor(progress)
            progress = 0

        t_old = t

    pbar.update(pbar.total - tt)

    tgems.dpd_graph.node[node]['sampled'] = True


def sample_from_tgem(tgem, t_min=0, t_max=30):
    for nd in tgem.dpd_graph.nodes():
        tgem.dpd_graph.node[nd]['sampled'] = False
        tgem.dpd_graph.node[nd]['timeserie'] = np.array([])

    sccs = list(nx.strongly_connected_components(tgem.dpd_graph))
    i = -1

    pbar = tqdm(total=len(sccs), desc='Sampled Nodes: ')
    while len(sccs) > 0:
        i = (i + 1) % len(sccs)
        if all_parents_were_sampled(tgem, list(sccs[i])):
            pbar_2 = tqdm(total=100,
                          desc='Sampling {}:'.format(', '.join(sccs[i])),
                          leave=False)
            # sample_child_node(tgem, list(sccs[i])[0], t_min, t_max, pbar_2)
            sample_sccs_nodes(tgem, list(sccs[i]), t_min, t_max, pbar_2)
            pbar_2.close()
            sccs.pop(i)
            pbar.update(1)
    pbar.close()

    sampled_data_df = pd.DataFrame()
    for nd in tqdm(tgem.dpd_graph.nodes(), desc='exporting data to DF'):
        node_sampled_tm = pd.DataFrame(
            data={'time':  tgem.dpd_graph.nodes[nd]['timeserie'], 'event': nd})
        sampled_data_df = pd.concat([sampled_data_df, node_sampled_tm])

    return sampled_data_df.sort_values('time', ascending=True)
