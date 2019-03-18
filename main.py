import pandas as pd
import sys
import csv

from rtgemlib.rtgem import RTGEM, empty_nodes
from rtgemlib.sampling import sample_from_tgem
from rtgemlib.configuration import Configuration
from rtgemlib.learning import LogLikelihood


if __name__ == '__main__':

    generateData=True
    
    try:
        generateData=sys.argv[1]
        print('Data will be generated')
    except:
        print('No data will be generated')

    # defines 5 examples graphs
    G1 = {'A': {
            'timescales': {},
            'lambdas': { (): 6 }
        }, 
        'B': {
            'timescales': {'A' : [[0, 1]]},
            'lambdas': { (0,): 1.5,
                         (1,): 2 }
                         },
        'C': {
            'timescales': {'A' : [[0, 2]]}, 
            'lambdas': { (0,): 7,
                         (1,): 1.5 }
                         },
        'D': {
            'timescales': {}, 
            'lambdas': { (): 2.9 }
      }
     }
    G2 = {'A': {
                'timescales': {},\
                'lambdas': { (): 2.8 }
            },\
            'B': {
                'timescales': {'A' : [[0, 0.5], [0.5, 1]]},\
                'lambdas': { (0,0): 1.7,\
                            (1,0): 2.3,\
                            (0,1): 3.5,\
                            (1,1): 2.1  }
            },
            'C': {
                'timescales': {}, \
                'lambdas': { (): 4 }
            },
            'D': {
                'timescales': {'A' : [[0, 2]], 'B' : [[0, 3]]}, \
                'lambdas': { (0,0): 2.5,\
                            (1,0): 7,\
                            (0,1): 1.5,\
                            (1,1): 4  }
            
            }
        }
    G3 = {'A': {
            'timescales': {},\
            'lambdas': { (): 2.8 }
        },\
        'B': {
            'timescales': {'A' : [[0, 1], [1, 2]]},\
            'lambdas': { (0,0): 4.3,\
                         (1,0): 1.4,\
                         (0,1): 7,\
                         (1,1): 1.5  }
        },
        'C': {
            'timescales': {'B' : [[0, 3]]}, \
            'lambdas': { (0,): 6.8,
                         (1,): 3.1}
        },
        'D': {
            'timescales': {}, \
            'lambdas': { (): 2.8 }
        
        }
      }
    G4 = {'A': {
            'timescales': {},\
            'lambdas': { (): 6 }
        },\
        'B': {
            'timescales': {},\
            'lambdas': { (): 1.5 }
        },
        'C': {
            'timescales': {'A' : [[0, 1], [1, 2]], 'B' : [[0, 3]]}, \
            'lambdas': {(0,0,0): 3.5, \
                        (0,0,1): 1.8, \
                        (0,1,0): 14, \
                        (0,1,1): 10, \

                        (1,0,0) : 8, \
                        (1,1,0): 8,\
                        (1,0,1): 2, \
                        (1,1,1): 1.7 }
        },
        'D': {
            'timescales': {'B' : [[0, 5],[5, 10]]}, \
            'lambdas': { (0,0): 5,\
                         (1,0): 4,\
                         (0,1): 5,\
                         (1,1): 2.7 }
        
      }
     }
    G5 = {'A': {
            'timescales': {},\
            'lambdas': { (): 6 }
        },\
        'B': {
            'timescales': {},\
            'lambdas': { (): 1.5 }
        },
        'C': {
            'timescales': {'B' : [[0, 3], [3, 6]]}, \
            'lambdas': { (0,0): 5,\
                         (1,0): 4,\
                         (0,1): 5,\
                         (1,1): 2.7 }
        },
        'D': {
            'timescales': {'A' : [[0, 2]], 'B': [[0,4], [4,8]]}, \
            'lambdas': {(0,0,0): 2, \
                        (0,0,1): 3.8, \
                        (0,1,0): 1.4, \
                        (0,1,1): 11, \

                        (1,0,0) : 2.8, \
                        (1,1,0): 5.6,\
                        (1,0,1): 6, \
                        (1,1,1): 1.5 }
        
      }
     }

    # build the list containing graphs
    models = []
    models.append(G1)
    models.append(G2)
    models.append(G3)
    # models.append(G4)
    # models.append(G5)

    # build rtgem from the defined models
    rtgem_models = [RTGEM(model) for model in models]

    k = len(models)

    # sample data and export it
    t_max = 10000

    # if data generation has been requested, generation of data
    if generateData:
        datas = [sample_from_tgem(rtgem_models[i], t_min=0, t_max=t_max)
                        for i in range(k)]

        for i in range(k):
                datas[i].to_csv('data/data' + str(i+1) + '.csv', index=False)

    # if no generation, then get data from csv files
    else:
        datas = []
        for i in range(k):
            datas.append(pd.read_csv('data/data' + str(i+1) + '.csv').reset_index(drop=True).sort_values('time', ascending=True))

    # initialization with k empty RTGEMS
    empty_models = [RTGEM(empty_nodes(list(rtgem_models[i].dpd_graph.nodes)), 
    default_end_timescale=1) for i in range(k)] 

    initial_configuration = Configuration(empty_models)
    reference_configuration = Configuration(rtgem_models)

    # priors P(Gi)
    pG = [0.5 for i in range(k)]

    crossed_likelihoods = reference_configuration.compute_crossed_Loglikelihoods(datas, t_max)

    # test upperbound
    node1 = 'A'
    node2 = 'B'

    for level in range(len(reference_configuration.rtgems)):  # tests for each level

        # computes likelihood p(Di|Gi)
        individuals_likelihoods = [LogLikelihood(model=reference_configuration.rtgems[j], observed_data=datas[j], t_max=t_max) 
        for j in range(reference_configuration.k)]
        # LogLikelihood > get_count_duration_df > duration 'duration' n'existe pas ?

        # computes bests likelihoods in non defined configurations 
        bests_likelihoods = reference_configuration.compute_bests(level, datas, t_max, node1, node2)

        U = reference_configuration.upperbound(level=i, individuals_likelihoods=individuals_likelihoods, bests_likelihoods=bests_likelihoods)
        print(U)


