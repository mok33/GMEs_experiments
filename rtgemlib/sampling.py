from tqdm.autonotebook import tqdm

import networkx as nx
import pandas as pd
import numpy as np
import math

from .rtgem import RTGEM
from .parentCount import init_parents_count, set_parent_count, updates_parents_count, get_parents_count_vector


def all_parents_were_sampled(tgem, scc):
    nd = scc[0]
    preds = [pa for pa in tgem.dpd_graph.predecessors(nd) if pa != nd]
    num_sampled_pa = np.array(
        [tgem.dpd_graph.node[pa]['sampled'] * 1 for pa in preds]).sum()

    return num_sampled_pa == len(preds) - (len(scc) - 1)


def set_nodes_parents_counts(tgems, nodes, t_max):
    for nd in nodes:
        for nd_pa in tgems.dpd_graph.predecessors(nd):
            pa_tms = tgems.dpd_graph.edges[(nd_pa, nd)]['timescales']
            if 'timeserie' not in tgems.dpd_graph.nodes[nd_pa]:
                tgems.dpd_graph.nodes[nd_pa]['timeserie'] = np.array([])
            pa_ts = tgems.dpd_graph.nodes[nd_pa]['timeserie']
            tgems.dpd_graph.edges[(nd_pa, nd)][
                'parent_count'] = set_parent_count(pa_ts, pa_tms, t_max)

    return tgems


def sample_sccs_nodes(tgems, scc_nodes, t_min, t_max, pbar):
    t = t_min
    tgems = set_nodes_parents_counts(tgems, scc_nodes, t_max)
    nds_pcvs_t_expirations = np.array(
        [get_parents_count_vector(tgems.get_parents_count(scc_nodes[i]), t, t_max) for i in range(len(scc_nodes))])

    lambda_t_nds = [tgems.get_lambda(scc_nodes[i], nd_pcv)
                    for i, nd_pcv in enumerate(nds_pcvs_t_expirations[:, 0])]
    progress = 0
    tt = 0
    t_old = t

    while t < t_max:
        Y = np.random.uniform(0, 1, size=len(scc_nodes))
        taus = -np.log(1 - Y) / lambda_t_nds

        min_tau_i = taus.argmin()
        t_change = min(nds_pcvs_t_expirations[min_tau_i, 1], t_max)

        if t + taus[min_tau_i] <= t_change:
            t += taus[min_tau_i]

            nd = scc_nodes[min_tau_i]
            tgems.dpd_graph.node[nd]['timeserie'] = np.append(
                tgems.dpd_graph.node[nd]['timeserie'], t)

            for nd_child in tgems.dpd_graph.successors(nd):
                if nd_child in scc_nodes:
                    nd_child_id = scc_nodes.index(nd_child)
                    tgems.dpd_graph.edges[(nd, nd_child)]['parent_count'] = updates_parents_count(
                        tgems.dpd_graph.edges[(nd, nd_child)]['parent_count'], tgems.dpd_graph.edges[(nd, nd_child)]['timescales'], t, t_max)

                    nds_pcvs_t_expirations[
                        nd_child_id, :] = get_parents_count_vector(
                        tgems.get_parents_count(nd_child), t, t_max)

                    lambda_t_nds[nd_child_id] = tgems.get_lambda(
                        nd_child, nds_pcvs_t_expirations[nd_child_id, 0])
        else:
            t = t_change

            nds_pcvs_t_expirations[min_tau_i, :] = get_parents_count_vector(
                tgems.get_parents_count(scc_nodes[min_tau_i]), t, t_max)

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


def sample_from_tgem(tgem, t_min=0, t_max=30):
    for nd in tgem.dpd_graph.nodes():
        tgem.dpd_graph.node[nd]['sampled'] = False
        tgem.dpd_graph.node[nd]['timeserie'] = np.array([])

    sccs = list(nx.strongly_connected_components(tgem.dpd_graph))
    i = -1

    pbar = tqdm(total=len(sccs), desc='Sampled Nodes: ', leave=False)
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
    for nd in tqdm(tgem.dpd_graph.nodes(), desc='exporting data to DF', leave=False):
        node_sampled_tm = pd.DataFrame(
            data={'time':  tgem.dpd_graph.nodes[nd]['timeserie'], 'event': nd})
        sampled_data_df = pd.concat([sampled_data_df, node_sampled_tm])

    return sampled_data_df.reset_index(drop=True).sort_values('time', ascending=True)
