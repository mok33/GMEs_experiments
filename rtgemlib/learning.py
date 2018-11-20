import pandas as pd
import numpy as np
import copy
from tqdm import tqdm

from .parentCount import get_nodes_pcv_from_data, get_node_pcv_from_data, lambdas_from_pcv_serie


def lambda_duration(time_serie):
    return (time_serie - time_serie.shift().fillna(0)).sum()


def get_count_duration_df(data, lambda_col='lambda_t', time_col='time', event_col='event'):
    lambda_vals = data[lambda_col].unique()
    lambda_count_duration_df = data.drop_duplicates([event_col, 'pcv'])[[event_col, 'pcv',
                                                                         lambda_col]]
    data_idx = data.set_index(
        ['event', lambda_col]).sort_index(level=1)

    lambda_count_duration_df.loc[:, 'duration'] = lambda_count_duration_df[['event', lambda_col]].apply(
        lambda lm: lambda_duration(data_idx.loc[(lm['event'], lm[lambda_col]), 'time']), axis=1)

    lambda_count_duration_df.loc[:, 'count'] = lambda_count_duration_df[['event', lambda_col]].apply(
        lambda lm: data_idx.loc[(lm['event'], lm[lambda_col])].shape[0], axis=1)

    return lambda_count_duration_df


def compute_logLikelihood(count_duration_df):
    log_likelihood = ((np.log(count_duration_df['lambda_t']) * count_duration_df['count']) +
                      (-count_duration_df['lambda_t'] * count_duration_df['duration'])).sum()

    return log_likelihood


def set_pcv_lambda_t(model, data, event_col='event', time_col='time'):
    data.loc[:, 'pcv'] = get_nodes_pcv_from_data(
        model, data, event_col, time_col).values

    data.loc[:, 'lambda_t'] = lambdas_from_pcv_serie(
        model, data)


def LogLikelihood(model, observed_data, time_col='time', event_col='event'):
    set_pcv_lambda_t(model, observed_data)
    lambda_count_duration_df = get_count_duration_df(observed_data)
    return compute_logLikelihood(lambda_count_duration_df)


def scoreBic(model, observed_data, time_col='time'):
    likelihood = LogLikelihood(model, observed_data)

    temps = observed_data[time_col]
    duree = temps.max() - temps.min()

    bic_score = likelihood - model.size() * np.log(duree)

    return bic_score


def mle_lambdas(model, data, time_col='time', event_col='event'):

    set_pcv_lambda_t(model, data)
    count_and_duration = get_count_duration_df(data)

    count_and_duration.loc[:, 'lambda_t'] = count_and_duration[
        'count'] / count_and_duration['duration']

    for _, row in count_and_duration.iterrows():
        model.set_lambda(row['event'], row['pcv'], row['lambda_t'])

    return count_and_duration


def get_node_LogLikelihood(model, cnt_drt_df, node):
    return compute_logLikelihood(cnt_drt_df[cnt_drt_df['event'] == node])


def LocaleLogLikelihood(model, data, baseLogL, nodeLogL, changed_node, optimize_lambdas=False):
    # data.set_index('event', inplace=True)
    # count_duration_df.set_index('event', inplace=True)
    changed_node_data = data[data['event'] == changed_node]

    changed_node_data.loc[:, 'pcv'] = get_node_pcv_from_data(model,
                                                             data,
                                                             changed_node_data,
                                                             event=changed_node).values

    changed_node_cnt_drt_df = get_count_duration_df(
        changed_node_data).reset_index(drop=True)

    if optimize_lambdas:
        changed_node_cnt_drt_df.loc[:, 'lambda_t'] = changed_node_cnt_drt_df[
            'count'] / changed_node_cnt_drt_df['duration']

    return baseLogL - nodeLogL + compute_logLikelihood(changed_node_cnt_drt_df), changed_node_cnt_drt_df

    # GRAPH OPTIMIZATION


# def apply_operation(op, *args):
#     print(*args)
#     op(*args)

pd.options.mode.chained_assignment = None  # default='warn'


def backward_neighbors_gen(rtgem, data, cnt_drt_df, LogL, size_log_td, log_td):
    """
    Generates all graphs that can possibly lead to the current graph, when
    applying one of the RTGEMs operators.
    """
    edges = copy.deepcopy(rtgem.dpd_graph.edges)
    for edge in edges:
        node_LogL = get_node_LogLikelihood(rtgem, cnt_drt_df, edge[1])
        node_size = np.power(2, rtgem.get_node_nb_parents_timescales(edge[1]))

        old_lambdas = copy.deepcopy(rtgem.dpd_graph.nodes[edge[1]]['lambdas'])

        if rtgem.inverse_add_edge_operator(edge):

            logL_ngr, cnt_drt_df = LocaleLogLikelihood(
                rtgem, data, LogL, node_LogL, edge[1], optimize_lambdas=True)
            rtgem.add_edge_operator(edge)
            rtgem.dpd_graph.nodes[edge[1]]['lambdas'] = old_lambdas

            size_log_td_ngbr = (size_log_td - (node_size / 2) * log_td)
            yield rtgem.inverse_add_edge_operator, [edge], logL_ngr, size_log_td_ngbr, cnt_drt_df

        if rtgem.inverse_extend_operator(edge):
            logL_ngr, cnt_drt_df = LocaleLogLikelihood(
                rtgem, data, LogL, node_LogL, edge[1], optimize_lambdas=True)
            rtgem.extend_operator(edge)
            rtgem.dpd_graph.nodes[edge[1]]['lambdas'] = old_lambdas

            size_log_td_ngbr = (size_log_td - (node_size / 2) * log_td)
            yield rtgem.inverse_extend_operator, [edge], logL_ngr, size_log_td_ngbr, cnt_drt_df

        for i_tm, timescale in enumerate(rtgem.dpd_graph.edges[edge]['timescales']):
            if rtgem.inverse_split_operator(edge, i_tm):
                logL_ngr, cnt_drt_df = LocaleLogLikelihood(
                    rtgem, data, LogL, node_LogL, edge[1], optimize_lambdas=True)
                rtgem.split_operator(edge, rtgem.dpd_graph.edges[
                                     edge]['timescales'][i_tm])
                rtgem.dpd_graph.nodes[edge[1]]['lambdas'] = old_lambdas

                size_log_td_ngbr = (size_log_td - (node_size / 2) * log_td)

                yield rtgem.inverse_split_operator, [edge, i_tm], logL_ngr, size_log_td_ngbr, cnt_drt_df


def backwardSearch(model, data):
    set_pcv_lambda_t(model, data)
    lambdas_count_duration_df = get_count_duration_df(data)

    LogL = compute_logLikelihood(lambdas_count_duration_df)
    log_td = np.log(data.iloc[-1]['time'] - data.iloc[0]['time'])

    size_log_td = model.size() * log_td

    score = LogL - size_log_td
    local_maximum = False

    it = 0
    while not local_maximum:
        #     max_ngbr_score = -np.inf
        local_maximum = True
        print('iteration number: {}: scoreBIC = {}'.format(it, score))

        for ngbr_info in tqdm(backward_neighbors_gen(model, data, lambdas_count_duration_df,
                                                     LogL, size_log_td, log_td)):

            inverse_op, args, LogL_ngbr, size_log_td_ngbr, changed_node_cnt_drt_df = ngbr_info
            score_ngbr = LogL_ngbr - size_log_td_ngbr

            if score_ngbr > score:
                print('max ngbr {}, args={} '.format(inverse_op, args))
                inverse_op(*args)
                LogL = LogL_ngbr
                size_log_td = size_log_td_ngbr
                changed_node = changed_node_cnt_drt_df.iloc[0]['event']
                lambdas_count_duration_df = lambdas_count_duration_df[
                    lambdas_count_duration_df['event'] != changed_node]
                lambdas_count_duration_df = pd.concat(
                    [lambdas_count_duration_df, changed_node_cnt_drt_df])

                local_maximum = False
                score = LogL - size_log_td

                break
        it += 1
    # initialisation des lambdas, et opti
    for nd in model.dpd_graph.nodes:
        model.initLambdas(nd)

    mle_lambdas(model, data)

    return model, score
