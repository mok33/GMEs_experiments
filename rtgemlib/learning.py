import pandas as pd
import numpy as np
import copy
from tqdm import tqdm
import scipy as sc

from .parentCount import get_parents_count_vector
from .sampling import set_nodes_parents_counts


def duration(params_df, model, t_max, node=None):
    params_df.loc[:, 'duration'] = 0
    params_df.set_index(['event', 'pcv'], inplace=True)
    nodes = model.dpd_graph.nodes
    if node is not None:
        nodes = [node]

    for n in nodes:
        t_ = 0

        while t_ < t_max:

            pcv, exp_t = get_parents_count_vector(
                model.get_parents_count(n), t_, t_max)

            params_df['duration'].loc[(n, pcv)] += min(exp_t, t_max) - t_
            t_ = exp_t

    params_df.reset_index(inplace=True)


def get_count_duration_df(data, model, t_max, node=None):
    nodes = model.dpd_graph.nodes
    if node is not None:
        nodes = [node]

    params_df = pd.DataFrame([(node, pcv, lm) for node in nodes
                              for pcv, lm in model.dpd_graph.nodes[node]['lambdas'].items()], columns=['event', 'pcv', 'lambda_t'])

    lambda_counts = data.groupby(['event', 'pcv'])['time'].size()

    ts = np.append(data['time'].values, t_max)
    # params_df.loc[:, 'duration'] = params_df[['event', 'pcv']].apply(
    #     lambda row: duration(ts, model, row['event'], row['pcv']), axis=1)
    duration(params_df, model, t_max, node)
    params_df.loc[:, 'count'] = params_df[['event', 'pcv']].apply(
        lambda row: lambda_counts.get(tuple(row), 0), axis=1)

    return params_df


def compute_logLikelihood(count_duration_df):
    log_likelihood = ((np.log(count_duration_df['lambda_t'].replace(0, 1)) * count_duration_df['count']) +
                      (-count_duration_df['lambda_t'] * count_duration_df['duration'])).sum()

    return log_likelihood


def set_pcv_lambda_t(model, data, t_max, event_col='event', time_col='time'):
    data['pcv'] = data[['event', 'time']].apply(lambda row: get_parents_count_vector(
        model.get_parents_count(row['event']), row['time'], t_max)[0], axis=1)

    data.loc[:, 'lambda_t'] = data[['event', 'pcv']].apply(
        lambda row: model.get_lambda(node=row['event'], pcv=row['pcv']), axis=1)


def set_nodes_timeseries(model, data):
    data.groupby('event')['time'].apply(
        lambda ts: model.set_node_timeserie(ts.name, ts.values))

    return model


def initModelFromData(model, observed_data, t_max):

    model = set_nodes_timeseries(model, observed_data)
    model = set_nodes_parents_counts(model, model.dpd_graph.nodes, t_max)

    set_pcv_lambda_t(model, observed_data, t_max)
    return model


def LogLikelihood(model, observed_data, t_max, time_col='time', event_col='event'):
    lambda_count_duration_df = get_count_duration_df(
        observed_data, model, t_max)
    return compute_logLikelihood(lambda_count_duration_df)


def scoreBic(model, observed_data, t_max, time_col='time'):
    likelihood = LogLikelihood(model, observed_data, t_max)
    bic_score = likelihood - (model.size() * np.log(t_max))

    return bic_score


def mle_lambdas(model, count_and_duration):

    count_and_duration.loc[:, 'lambda_t'] = count_and_duration[
        'count'] / (count_and_duration['duration'])

    events_lambdas = count_and_duration.groupby('event').apply(
        lambda df: df.set_index('pcv')['lambda_t'].to_dict()).to_dict()

    for node in model.dpd_graph.nodes:
        model.dpd_graph.nodes[node]['lambdas'] = events_lambdas[node]

    return count_and_duration


def get_node_LogLikelihood(cnt_drt_df, node):
    return compute_logLikelihood(cnt_drt_df[cnt_drt_df['event'] == node])


def LocaleLogLikelihood(model, data, t_max, baseLogL, nodeLogL, changed_node, optimize_lambdas=False):
    # data.set_index('event', inplace=True)
    # count_duration_df.set_index('event', inplace=True)
    changed_node_data = data[data['event'] == changed_node]
    model = set_nodes_parents_counts(model, changed_node, t_max)

    set_pcv_lambda_t(model, changed_node_data, t_max)
    data.loc[data['event'] == changed_node, :] = changed_node_data

    changed_node_cnt_drt_df = get_count_duration_df(
        changed_node_data, model, t_max, changed_node).reset_index(drop=True)

    if optimize_lambdas:
        changed_node_cnt_drt_df.loc[:, 'lambda_t'] = (changed_node_cnt_drt_df[
            'count'] / changed_node_cnt_drt_df['duration'])
        # changed_node_cnt_drt_df.loc[
        #     :, 'lambda_t'] = changed_node_cnt_drt_df.loc[:, 'lambda_t'] / changed_node_cnt_drt_df.loc[:, 'lambda_t'].sum()
        # # changed_node_cnt_drt_df['lambda_t'] = changed_node_cnt_drt_df[
        # #     'lambda_t']

    return (baseLogL - nodeLogL) + compute_logLikelihood(changed_node_cnt_drt_df), changed_node_cnt_drt_df

    # GRAPH OPTIMIZATION


# def apply_operation(op, *args):
#     print(*args)
#     op(*args)

pd.options.mode.chained_assignment = None  # default='warn'


def backward_neighbors_gen(rtgem, data, t_max, cnt_drt_df, LogL, size_log_td, log_td):
    """
    Generates all graphs that can possibly lead to the current graph, when
    applying one of the RTGEMs operators.
    """
    edges = copy.deepcopy(rtgem.dpd_graph.edges)
    for edge in edges:
        node_LogL = get_node_LogLikelihood(cnt_drt_df, edge[1])
        node_size = np.power(2, rtgem.get_node_nb_parents_timescales(edge[1]))

        old_lambdas = copy.deepcopy(rtgem.dpd_graph.nodes[edge[1]]['lambdas'])

        if rtgem.inverse_add_edge_operator(edge):

            logL_ngr, changed_cnt_drt_df = LocaleLogLikelihood(
                rtgem, data, t_max, LogL, node_LogL, edge[1], optimize_lambdas=True)
            rtgem.add_edge_operator(edge)
            rtgem.dpd_graph.nodes[edge[1]]['lambdas'] = old_lambdas

            size_log_td_ngbr = (size_log_td - (node_size / 2) * log_td)
            yield rtgem.inverse_add_edge_operator, [edge], logL_ngr, size_log_td_ngbr, changed_cnt_drt_df

        if rtgem.inverse_extend_operator(edge):
            logL_ngr, changed_cnt_drt_df = LocaleLogLikelihood(
                rtgem, data, t_max, LogL, node_LogL, edge[1], optimize_lambdas=True)
            rtgem.extend_operator(edge)
            rtgem.dpd_graph.nodes[edge[1]]['lambdas'] = old_lambdas

            size_log_td_ngbr = (size_log_td - (node_size / 2) * log_td)
            yield rtgem.inverse_extend_operator, [edge], logL_ngr, size_log_td_ngbr, changed_cnt_drt_df

        for i_tm, timescale in enumerate(rtgem.dpd_graph.edges[edge]['timescales']):
            if rtgem.inverse_split_operator(edge, i_tm):
                logL_ngr, changed_cnt_drt_df = LocaleLogLikelihood(
                    rtgem, data, t_max, LogL, node_LogL, edge[1], optimize_lambdas=True)
                rtgem.split_operator(edge, rtgem.dpd_graph.edges[
                                     edge]['timescales'][i_tm])
                rtgem.dpd_graph.nodes[edge[1]]['lambdas'] = old_lambdas

                size_log_td_ngbr = (size_log_td - (node_size / 2) * log_td)

                yield rtgem.inverse_split_operator, [edge, i_tm], logL_ngr, size_log_td_ngbr, changed_cnt_drt_df


def forward_neighbors_gen(rtgem, data, t_max, cnt_drt_df, LogL, size_log_td, log_td, possible_edges):
    edges = copy.deepcopy(rtgem.dpd_graph.edges)

    for edge in edges:
        if rtgem.get_edge_timescales_horrizon(edge) < t_max:
            node_LogL = get_node_LogLikelihood(cnt_drt_df, edge[1])
            node_size = np.power(
                2, rtgem.get_node_nb_parents_timescales(edge[1]))

            old_lambdas = copy.deepcopy(
                rtgem.dpd_graph.nodes[edge[1]]['lambdas'])

            rtgem.extend_operator(edge)
            logL_ngr, changed_cnt_drt_df = LocaleLogLikelihood(
                rtgem, data, t_max, LogL, node_LogL, edge[1], optimize_lambdas=True)

            rtgem.inverse_extend_operator(edge)
            rtgem.dpd_graph.nodes[edge[1]]['lambdas'] = old_lambdas

            size_log_td_ngbr = (
                (size_log_td - node_size * log_td) + (node_size * 2) * log_td)
            yield rtgem.extend_operator, [edge], logL_ngr, size_log_td_ngbr, changed_cnt_drt_df

        for i_tm, timescale in enumerate(rtgem.dpd_graph.edges[edge]['timescales']):
            node_LogL = get_node_LogLikelihood(cnt_drt_df, edge[1])
            node_size = np.power(
                2, rtgem.get_node_nb_parents_timescales(edge[1]))

            old_lambdas = copy.deepcopy(
                rtgem.dpd_graph.nodes[edge[1]]['lambdas'])
            rtgem.split_operator(edge, timescale)
            logL_ngr, changed_cnt_drt_df = LocaleLogLikelihood(
                rtgem, data, t_max, LogL, node_LogL, edge[1], optimize_lambdas=True)

            rtgem.inverse_split_operator(edge, i_tm)
            rtgem.dpd_graph.nodes[edge[1]]['lambdas'] = old_lambdas

            size_log_td_ngbr = (
                (size_log_td - node_size * log_td) + (node_size * 2) * log_td)

            yield rtgem.split_operator, [edge, timescale], logL_ngr, size_log_td_ngbr, changed_cnt_drt_df

    for edge in possible_edges:
        node_LogL = get_node_LogLikelihood(cnt_drt_df, edge[1])
        node_size = np.power(2, rtgem.get_node_nb_parents_timescales(edge[1]))

        old_lambdas = copy.deepcopy(rtgem.dpd_graph.nodes[edge[1]]['lambdas'])

        rtgem.add_edge_operator(edge)
        logL_ngr, changed_cnt_drt_df = LocaleLogLikelihood(
            rtgem, data, t_max, LogL, node_LogL, edge[1], optimize_lambdas=True)

        rtgem.inverse_add_edge_operator(edge)
        rtgem.dpd_graph.nodes[edge[1]]['lambdas'] = old_lambdas

        size_log_td_ngbr = (
            (size_log_td - node_size * log_td) + (node_size * 2) * log_td)

        yield rtgem.add_edge_operator, [edge], logL_ngr, size_log_td_ngbr, changed_cnt_drt_df


# def backwardSearch(model, data, t_max):
#     model = set_nodes_timeseries(model, observed_data)
#     model = set_nodes_parents_counts(model, model.dpd_graph.nodes)

#     set_pcv_lambda_t(model, data, t_max)

#     lambdas_count_duration_df = get_count_duration_df(data, model, t_max)

#     LogL = compute_logLikelihood(lambdas_count_duration_df)
#     log_td = np.log(data.iloc[-1]['time'] - data.iloc[0]['time'])

#     size_log_td = model.size() * log_td

#     score = LogL - size_log_td
#     local_maximum = False

#     it = 0
#     while not local_maximum:
#         #     max_ngbr_score = -np.inf
#         local_maximum = True
#         print('iteration number: {}: scoreBIC = {}'.format(it, score))

#         for ngbr_info in tqdm(backward_neighbors_gen(model, data, lambdas_count_duration_df,
# LogL, size_log_td, log_td)):

#             inverse_op, args, LogL_ngbr, size_log_td_ngbr, changed_node_cnt_drt_df = ngbr_info
#             score_ngbr = LogL_ngbr - size_log_td_ngbr

#             if score_ngbr > score:
#                 print('max ngbr {}, args={} '.format(inverse_op, args))
#                 inverse_op(*args)
#                 LogL = LogL_ngbr
#                 size_log_td = size_log_td_ngbr
#                 changed_node = changed_node_cnt_drt_df.iloc[0]['event']
#                 lambdas_count_duration_df = lambdas_count_duration_df[
#                     lambdas_count_duration_df['event'] != changed_node]
#                 lambdas_count_duration_df = pd.concat(
#                     [lambdas_count_duration_df, changed_node_cnt_drt_df])

#                 local_maximum = False
#                 score = LogL - size_log_td

#                 break
#         it += 1
#     # initialisation des lambdas, et opti
#     for nd in model.dpd_graph.nodes:
#         model.initLambdas(nd)

#     mle_lambdas(model, data)

#     return model, score
