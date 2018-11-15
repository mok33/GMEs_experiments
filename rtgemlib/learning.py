import pandas as pd
import numpy as np

from .parentCount import set_pcv_from_data, lambdas_from_pcv_serie


def get_count_duration_df(data, lambda_col='lambda_t', time_col='time', event_col='event'):
    lambda_vals = data[lambda_col].unique()
    lambda_count_duration_df = data.drop_duplicates([event_col, lambda_col])[[event_col,
                                                                              lambda_col]]

    lambda_count_duration_df['duration'] = lambda_count_duration_df[lambda_col].apply(lambda lm: (((data[lambda_col] == lm) * 1.0) *
                                                                                                  (data[time_col] - data[time_col].shift().fillna(0))).sum())
    lambda_count_duration_df['count'] = lambda_count_duration_df[
        lambda_col].apply(lambda lm: (data[lambda_col] == lm).sum())

    return lambda_count_duration_df


def compute_logLikelihood(count_duration_df):
    log_likelihood = ((np.log(count_duration_df['lambda_t']) * count_duration_df['count']) +
                      (-count_duration_df['lambda_t'] * count_duration_df['duration'])).sum()

    return log_likelihood


def LogLikelihood(model, observed_data, time_col='time', event_col='event'):
    observed_data = observed_data.groupby(event_col).apply(lambda event_time_serie: set_pcv_from_data(model,
                                                                                                      observed_data,
                                                                                                      event_time_serie
                                                                                                      )
                                                           ).reset_index().sort_values(by=time_col, ascending=True)
    observed_data['lambda_t'] = lambdas_from_pcv_serie(model, observed_data)
    lambda_count_duration_df = get_count_duration_df(observed_data)
    return compute_logLikelihood(lambda_count_duration_df)


def scoreBic(model, observed_data, time_col='time'):
    likelihood = LogLikelihood(model, observed_data)

    temps = observed_data[time_col]
    duree = temps.max() - temps.min()

    bic_score = likelihood - model.size() * np.log(duree)

    return bic_score


def mle_lambdas(model, data, time_col='time', event_col='event'):

    data = data.groupby(event_col).apply(lambda event_time_serie: set_pcv_from_data(model,
                                                                                    data,
                                                                                    event_time_serie)
                                         ).reset_index().sort_values(by=time_col, ascending=True)

    count_and_duration = get_count_duration_df(data, 'pcv')
    count_and_duration['lambdas'] = count_and_duration[
        'count'] / count_and_duration['duration']

    for _, row in count_and_duration.iterrows():
        model.set_lambda(row['event'], row['pcv'], row['lambdas'])

    return count_and_duration
