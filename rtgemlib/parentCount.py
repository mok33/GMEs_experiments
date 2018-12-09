import pandas as pd
import numpy as np

from .rtgem import RTGEM


def get_parents_count_vector(parents_count, t, t_max):
    pcv = ()
    t_expr = t_max
    for i, parent_count in enumerate(parents_count):
        cv, t_e = get_parent_count_vector(
            parent_count, t, t_max)

        pcv += cv
        t_expr = min(t_expr, t_e)

    return pcv, t_expr


def get_parent_count_vector(parent_count, t, t_max):
    cv = ()
    t_expr = t_max

    for par_act_expr in parent_count:
        if par_act_expr.shape[0] > 0:

            res = par_act_expr[(t > par_act_expr[:, 0]) &
                               (t < par_act_expr[:, 1])]
            if res.shape[0] > 0:
                cv += (1,)
                t_expr = min(t_expr, res[0, -1])
            else:
                cv += (0,)
                act_t = par_act_expr[t < par_act_expr[:, 0]]
                if(act_t.shape[0] > 0):
                    t_expr = min(t_expr, act_t[0, 0])
        else:
            cv += (0,)

    return cv, t_expr


def init_parents_count(timeserie, timescales):
    return [np.empty((0, 2)) for tm in timescales]


def set_parent_count(pa_timeserie, pa_timescales, t_max):
    parent_count = init_parents_count(pa_timeserie, pa_timescales)
    for t in pa_timeserie:
        parent_count = updates_parents_count(
            parent_count, pa_timescales, t, t_max)

    return parent_count


def updates_parents_count(parent_count, tms,
                          t_n_pa, t_max):

    for tm_num, (a, b) in enumerate(tms):

        t_act_tms = min(t_n_pa + a, t_max)
        t_expr_tms = min(t_n_pa + b, t_max)

        if len(parent_count[tm_num]) > 0 and t_act_tms <= parent_count[tm_num][-1, -1]:
            parent_count[tm_num][-1, -1] = t_expr_tms
        else:
            parent_count[tm_num] = np.append(
                parent_count[tm_num], [[t_act_tms, t_expr_tms]], axis=0)

    return parent_count
