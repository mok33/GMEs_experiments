import pandas as pd
import numpy as np

from .rtgem import RTGEM


def count_timescale(df, timescales_parent):
    """
    df: Dataframe, indexé par la serie temporelle du parent.
    timescales_parent: tableau contenant les timescales du parent. (tableau d'intervalle semi-ouvert)

    retourne un DF indiquant la configuration du parent à chaque t, et le temps de début et d'expiration de la configuration.
    """
    # pour chaque instant t où le parent s'est réalisé
    for i, t in enumerate(df.index):
        # pour chaque timescale
        for tm_num, (a, b) in enumerate(timescales_parent):
            # la colonne indiquant si le parent a eu lieu dans la timscale
            # timescales_parent[tm_num]
            tm_col = df.columns[tm_num]
            # df est indexé par les timestamps auxquels la parent a eu lieu,
            # les timestamps <= t (t est le iéme index) sont séléctionnée,
            # les timestampes sont "mappée" à un tableau de booléen qui indique pour chacun si il appartient ou non au timescale,
            # la somme du tableau correspond au nombre de fois où le parents s'est réalisé dans la timescale.
            # Cette somme est seuilé par 1.
            df.loc[t, tm_col] = (
                ((df.index[:i + 1] > t - b) & (df.index[:i + 1] <= t - a)).sum() >= 1) * 1
            # si le parent s'est réaliser dans [a, b[,
            # on doit ajouter le temps d'expiration de la configuration
            if df.loc[t, tm_col]:
                # la durée maximale (ne prend pas en compte les futures
                # réalisatio)
                duration = (b - a)
                # le temps d'expiration
                df.loc[t, 't_expr'] = min(
                    t + duration, df.loc[t, 't_expr'])
                df.loc[(df.index.values > t) & (
                    df.index < t + duration), tm_col] = 1
    return df


def init_parent_count_df(timeserie, timescales,):
    return pd.DataFrame(data=[], index=timeserie,
                        columns=['[{},{}['.format(tm[0], tm[1]) for tm in timescales] + ['expr([{},{}[)'.format(tm[0], tm[1]) for tm in timescales] + ['t_expr'])


def updates_parent_count_df(parent_count_df, tms,
                            t_n_pa):

    t_pa = np.append(parent_count_df.values, t_n_pa)

    for tm_num, (a, b) in enumerate(tms):
        tm_col = parent_count_df.columns[tm_num]
        t_expr_tms = t_n_pa + (b - a)
        tm_expr_col = 'expr({})'.format(tm_col)

        if parent_count_df.shape[0] == 0:
            parent_count_df.loc[t_n_pa, tm_col] = (
                ((t_pa > t_n_pa - b) & (t_pa <= t_n_pa - a)).sum() >= 1) * 1
        else:
            parent_count_df.loc[
                t_n_pa, tm_col] = (t_n_pa < parent_count_df.iloc[-1][tm_expr_col]) * 1

        if parent_count_df.loc[t_n_pa, tm_col]:
            parent_count_df.loc[t_n_pa, tm_expr_col] = t_expr_tms

    parent_count_df.loc[t_n_pa, 't_expr'] = parent_count_df.loc[
        t_n_pa, parent_count_df.columns[len(tms):-1]].min()
    return parent_count_df


def get_parents_count_dfs(parents_time_serie, parents_timescales):
    """
    parents_time_serie: tableau de np.array contenant la serie temporelle de chaque parent.
    parents_timescales: tableau contenant les timescales de chaque parents, attention parents_timescales[i] doit contenir les
    timescale du parent parents_time_serie[i]

    retourne un tableau de DataFrame, un DF par parent, indiquant la configuration du parent sur une plage temporelle.
    """
    parents_count_dfs = []
    for i, parent_time_serie in enumerate(parents_time_serie):
        activation_windows_df = pd.DataFrame(data=[], index=parent_time_serie,
                                             columns=[']{},{}]'.format(tm[1], tm[0]) for tm in parents_timescales[i]] + ['t_expr'])
        activation_windows_df.loc[:, 't_expr'] = np.inf

        # calcul la configuration du parent dans les instants où ce dernier a eu lieu
        # ET le temps d'expiration de cette configuration.
        activation_windows_df = count_timescale(
            activation_windows_df, parents_timescales[i])
        activation_windows_df = activation_windows_df.reset_index().rename(columns={
            'index': 't'})
        activation_windows_df.loc[:, 'parent_count'] = activation_windows_df.apply(
            lambda r: tuple([int(single_count) for single_count in r.iloc[2:]]), axis=1)

        # fusionne les intervalles de temps consécutifs sur lesquels la
        # configuration du parent ne change pas.
        values_g = (activation_windows_df['parent_count'] != activation_windows_df[
                    'parent_count'].shift()).cumsum()
        parent_count_df = activation_windows_df.groupby(['parent_count', values_g],
                                                        as_index=False).agg({'t': 'min', 't_expr': 'max'}).sort_values('t')
        # le temps d'expiration ne doit pas dépasser
        # le temps de début de la prochaine configuration
        parent_count_df['t_expr'].clip_upper(
            parent_count_df['t'].shift(-1).fillna(parent_count_df['t'].max()), inplace=True)

        parents_count_dfs.append(parent_count_df)

    return parents_count_dfs


def getParentCountVector(parent_c_df, t, nb_p_tms):
    """
    Trouve la configuration du parent à l'instant t à partir du DF parent_c_df
    """
    expr_t = np.inf
    pcv = (0,) * nb_p_tms
    if not parent_c_df.empty:
        res = parent_c_df[parent_c_df.apply(
            lambda r: t >= r['t'] and t < r['t_expr'], axis=1)]

        if res.shape[0] == 0:
            df = parent_c_df[parent_c_df['t'] > t]
            if df.shape[0] > 0:
                expr_t = df.iloc[-1]['t']
        else:
            pcv = res['parent_count'].values[0]
            expr_t = res['t_expr'].values[0]
    return pcv, expr_t


def getParentsCountVector(parents_c_dfs, timescales_parents, t):
    pc_expr = np.array([getParentCountVector(parent_count_df, t, len(timescales_parent))
                        for parent_count_df, timescales_parent in zip(parents_c_dfs, timescales_parents)])
    if parents_c_dfs == []:
        return (), np.inf
    return sum(pc_expr[:, 0], ()), min(pc_expr[:, 1])


def get_node_pcv_from_data(model, all_time_series_df, time_serie, event=None, time_col='time', event_col='event'):
    if event is None:
        event = time_serie.name

    parents_timeseries = [all_time_series_df.set_index(event_col).loc[parent][time_col].values
                          for parent in model.get_node_parents(event)]
    parents_timescales = model.get_node_parents_timescales(event)

    pc_df = get_parents_count_dfs(parents_timeseries, parents_timescales)

    time_serie.loc[:, 'pcv'] = time_serie[time_col].apply(lambda t: getParentsCountVector(pc_df,
                                                                                          parents_timescales,
                                                                                          t)[0])

    return time_serie.set_index(time_col)['pcv']


def get_nodes_pcv_from_data(model, data, event_col='event', time_col='time'):
    return data.groupby(event_col).apply(lambda event_time_serie: get_node_pcv_from_data(model,
                                                                                         data,
                                                                                         event_time_serie)
                                         ).reset_index().sort_values(by=time_col, ascending=True)['pcv']


def lambdas_from_pcv_serie(model, all_time_series_df, event_col='event'):
    lambda_t = all_time_series_df.apply(lambda event: model.get_lambda(event[event_col],
                                                                       event['pcv']), axis=1)

    return lambda_t
