import pandas as pd
import numpy as np

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

    # lambda_count_duration_df[lambda_col]
    #     .apply(lambda lm: (((data[lambda_col] == lm) * 1.0) *
    #                        (data[time_col] - data[time_col].shift().fillna(0))).sum())
    lambda_count_duration_df.loc[:, 'count'] = lambda_count_duration_df[
        lambda_col].apply(lambda lm: (data[lambda_col] == lm).sum())

    return lambda_count_duration_df


def compute_logLikelihood(count_duration_df):
    log_likelihood = ((np.log(count_duration_df['lambda_t']) * count_duration_df['count']) +
                      (-count_duration_df['lambda_t'] * count_duration_df['duration'])).sum()

    return log_likelihood


def LogLikelihood(model, observed_data, time_col='time', event_col='event'):
    observed_data.sort_values(by=time_col, inplace=True)
    observed_data.loc[:, 'pcv'] = get_nodes_pcv_from_data(
        model, observed_data, event_col, time_col).values

    observed_data.loc[:, 'lambda_t'] = lambdas_from_pcv_serie(
        model, observed_data)
    lambda_count_duration_df = get_count_duration_df(observed_data)
    return compute_logLikelihood(lambda_count_duration_df)


def scoreBic(model, observed_data, time_col='time'):
    likelihood = LogLikelihood(model, observed_data)

    temps = observed_data[time_col]
    duree = temps.max() - temps.min()

    bic_score = likelihood - model.size() * np.log(duree)

    return bic_score


def mle_lambdas(model, data, time_col='time', event_col='event'):

    data.loc[:, 'pcv'] = get_nodes_pcv_from_data(
        model, data, event_col, time_col).values

    data.loc[:, 'lambda_t'] = lambdas_from_pcv_serie(
        model, data)

    count_and_duration = get_count_duration_df(data)

    count_and_duration.loc[:, 'lambdas'] = count_and_duration[
        'count'] / count_and_duration['duration']
    for _, row in count_and_duration.iterrows():
        model.set_lambda(row['event'], row['pcv'], row['lambdas'])

    return count_and_duration


def get_node_LogLikelihood(model, cnt_drt_df, node):
    return compute_logLikelihood(cnt_drt_df[cnt_drt_df['event'] == node])


def LocaleLogLikelihood(model, data, baseLogL, nodeLogL, changed_node):
    # data.set_index('event', inplace=True)
    # count_duration_df.set_index('event', inplace=True)
    changed_node_data = data[data['event'] == changed_node]

    changed_node_data.loc[:, 'pcv'] = get_node_pcv_from_data(model,
                                                             data,
                                                             changed_node_data,
                                                             event=changed_node).values

    changed_node_cnt_drt_df = get_count_duration_df(
        changed_node_data).reset_index()

    return baseLogL - nodeLogL + compute_logLikelihood(changed_node_cnt_drt_df)
