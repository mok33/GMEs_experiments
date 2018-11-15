from rtgemlib import RTGEM
from rtgemlib import sample_from_tgem, LogLikelihood, scoreBic, mle_lambdas

if __name__ == '__main__':
    # first key is the child, second key is ur moma
    model_AB = {
        'B':
        {
            'timescales': {'A': [[1, 2]], 'C': [[5, 6]]},
            'lambdas': {
                (0, 0): 0.6,
                (0, 1): 1.6,
                (1, 0): 3,
                (1, 1): 1
            }
        },
        'A': {
            'timescales': {'B': [[2, 3]]},
            'lambdas': {
                (0,): 0.5,
                (1,): 1
            }
        },
        'C': {
            'timescales': {'C': [[2, 3]]},
            'lambdas': {
                (0,): 0.5,
                (1,): 1
            }
        },
        'D': {
            'timescales': {},
            'lambdas': {(): 2}
        }
    }

    gms = RTGEM(model_AB)
    sampled_data = sample_from_tgem(gms)
    print('L = {};\nBIC = {}'.format(
        LogLikelihood(gms, sampled_data), scoreBic(gms, sampled_data)))

    mle_lambdas(gms, sampled_data)
    print('L = {};\nBIC = {}'.format(
        LogLikelihood(gms, sampled_data), scoreBic(gms, sampled_data)))
