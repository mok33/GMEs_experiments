import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

from .rtgem import RTGEM, symmetric_difference
from .learning import LogLikelihood, set_pcv_lambda_t, get_count_duration_df, mle_lambdas, neighbors_gen

class Configuration:

    def __init__(self, rtgems, individual_priors=None, level=None, Z=2, delta=0.5):
        self.rtgems = rtgems
        self.k = len(rtgems)
        
        if level == None:
          self.level = self.k # default: complete configuration, l = k
        else : 
          self.level = level
          
        if (individual_priors == None or (individual_priors == None and len(individual_priors) != self.k)):
          self.individual_priors = [0.5 for i in range(self.k)]
        else: 
          self.individual_priors = individual_priors
        
        self.Z = Z
        self.delta = delta
        self.prior = self.set_prior(delta)
        self.score = None

    def penalization_product(self):
      product_penalization = 1
      
      for edge in symmetric_difference(self.rtgems):
        node1, node2 = edge[0], edge[1]
        
        ts_matrix = build_ts_matrix(self.rtgems, node1, node2)
        editsij = edits(ts_matrix, node1, node2)
        product_penalization *= (1 - self.delta) ** editsij

      return product_penalization

  
    def set_delta(self, new_delta):
      self.delta = new_delta

    def priors_product(self, until_index):
      """
      Don't call in main
      """
      exponent = 1/(1 + (self.k - 1) * self.delta)
      
      product_priors = 1
      for i in range(until_index):
        product_priors *= self.individual_priors[i] ** exponent
      
      return product_priors

    def set_prior(self, delta):
      self.set_delta(delta)
    
      # priors product
      product_priors = self.priors_product(until_index=self.k)
      
      # penalization product
      product_penalization = self.penalization_product()
        
      prior = self.Z * product_priors * product_penalization
      
      return prior

    
    def max_prior(self, Z, nb_deltas=500):
      
      delta_values = [i/nb_deltas for i in range(nb_deltas+1)]
      real_priors = [self.set_prior(delta=i/nb_deltas) 
                     for i in range(nb_deltas+1)]

      best_delta = delta_values[np.argmax(real_priors)]
      max_prior = real_priors[np.argmax(real_priors)]
      
      return best_delta, max_prior, delta_values, real_priors
    
    def set_score(self, observed_datas, t_max):
      
      self.set_prior(self.delta)

      # computes likelihood p(Di|Gi)
      individuals_likelihoods = [LogLikelihood(model=self.rtgems[i], 
                                 observed_data=observed_datas[i], 
                                 t_max=t_max) for i in range(self.k)]
      
      product_individuals_likelihoods = 1
      for i in range(self.k):
        product_individuals_likelihoods *= individuals_likelihoods[i]
      
      score = self.prior * product_individuals_likelihoods
      self.score = score
        
      return score

    def upperbound(self, level, individuals_likelihoods, bests_likelihoods):
      """
      Calculates upperbound for given level, loglikelihoods being a list of p(Di|Gi) 
      """
      # penalization product
      product_penalization = self.penalization_product()

      # posterior product for Graphs 0 to l (level)
      exponent = 1/(1 + (self.k - 1) * self.delta)
      
      product_priors = 1
      for i in range(level):
        product_priors *= self.individual_priors[i] ** exponent * individuals_likelihoods[i]

      # "bests" product (best structure for each non defined graph)
      product_best = 1
      
      for i in range(level, self.k):
        product_best *= self.individual_priors[i] ** exponent * bests_likelihoods[i]
        
      upperbound = product_priors * product_penalization * product_best

      return upperbound

    def compute_bests(self, level, datas, t_max, node1, node2):

      log_td = np.log(t_max)

      # score = LogL - size_log_td
      possible_edges = list(itertools.product([node1, node2], repeat = 2))

      bests = []

      for i in range(self.k):
        count_duration_df = get_count_duration_df(model=self.rtgems[i], data=datas[i], t_max=t_max)
        lambdas_count_duration_df = mle_lambdas(model=self.rtgems[i], count_and_duration=count_duration_df)

        LogL = LogLikelihood(model=self.rtgems[i], observed_data=datas[i], t_max=t_max)
        size_log_td = self.rtgems[i].size() * log_td

        scoresi = []

        for ngbr_info in neighbors_gen(self.rtgems[i],datas[0], t_max, lambdas_count_duration_df, LogL, size_log_td, log_td, possible_edges):

            _, _, LogL_ngbr, size_log_td_ngbr, _ = ngbr_info
            score_ngbr = LogL_ngbr - size_log_td_ngbr

            scoresi.append(score_ngbr)

        bests.append(np.max(scoresi))

      return bests


      
    def compute_crossed_Loglikelihoods(self, datas, t_max, exportFig=True):
      """
      Computes a matrix with logLikelihood for each (Graphi, Dataj)
      """

      for i in range(self.k):
        set_pcv_lambda_t(model=self.rtgems[i], data=datas[i], t_max=t_max)

      Loglikelihoods = []

      for i in range(self.k):
        Loglikelihoodsi = []

        for j in range(self.k):
          Loglikelihoodij = LogLikelihood(model=self.rtgems[i], observed_data=datas[j], t_max=t_max)
          Loglikelihoodsi.append(Loglikelihoodij)

        Loglikelihoods.append(Loglikelihoodsi)

      if exportFig:
        graphs = ['Graph'+str(i+1) for i in range(self.k)]
        datas = ['Data'+str(i+1) for i in range(self.k)]

        crossed_likelihoods = np.array(Loglikelihoodsi)

        fig, ax = plt.subplots()
        # ax.imshow(crossed_likelihoods) # invalid dimensions for image data: TODO

        # We want to show all ticks...
        ax.set_xticks(np.arange(len(datas)))
        ax.set_yticks(np.arange(len(graphs)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(datas)
        ax.set_yticklabels(graphs)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        ax.set_title("Likelihoods")
        fig.tight_layout()
        # plt.show()

        fig.savefig("images/CrossedLikelihoods.png")

      return Loglikelihoods

      

def build_ts_matrix(rtgems, node1, node2):
  """
    rtgems is a list of rtgems build with RTGEM() constructor
    node1 and node2 are the names of node that are in the rtgems,
    for example node1 = 'A' and node2 = 'B'
  """

  matrix = pd.DataFrame(columns=['ts->', 'ts<-'])

  # two different edges can be build from two nodes
  edge1 = (node1, node2)
  edge2 = (node2, node1)

  timescales = {}
  for i in range(len(rtgems)): # for all graphs
    current_rtgem = rtgems[i]

    # node1 -> node2
    if edge1 in list(current_rtgem.dpd_graph.edges()):
      timescales['ts->'] = current_rtgem.get_node_parents_timescales(node2)[0]
    else:
      timescales['ts->'] = []

    # node2 -> node1
    if edge2 in list(current_rtgem.dpd_graph.edges()):
      timescales['ts<-'] = current_rtgem.get_node_parents_timescales(node1)[0]
    else:
      timescales['ts<-'] = []

    matrix = matrix.append(timescales, ignore_index=True)
    
  # adds useful computations : number of -> and <- edges 
  matrix['->'] = matrix.apply(lambda row: len(row['ts->'])>0, axis=1)
  matrix['<-'] = matrix.apply(lambda row: len(row['ts<-'])>0, axis=1)
  
  # checks  if there is no relation ou bidirectional relation between the variables
  matrix['BiD'] = matrix['->'] & matrix['<-']
  matrix['NoRel'] = (matrix['->']==False) & (matrix['<-']==False)
  
  # count number of timescales
  matrix['nb_ts->'] = matrix.apply(lambda row: len(row['ts->']), axis=1)
  matrix['nb_ts<-'] = matrix.apply(lambda row: len(row['ts<-']), axis=1)
  matrix['nb_ts_bid'] = matrix.apply(lambda row: row['BiD']*(len(row['ts->'])+len(row['ts<-'])), axis=1)
    
  return matrix

def edits(list_of_arcs, node1, node2):
  """
    list_of_arcs = DataFrame d'arcs (présents ou absents) qui sont communs à n GEMs 
    format : '->'=True s'il y a une liaison de A vers B
              '<-'=True s'il y a une liaison de B vers A
    build_ts_matrix returns a matrix in the right format
  """
  nb_graphs = list_of_arcs.shape[0]
  
  try:
    count_right_relation = pd.value_counts(list_of_arcs['->'].values, sort=False)[True]
  except: 
    count_right_relation = 0
  
  try:
    count_left_relation = pd.value_counts(list_of_arcs['<-'].values, sort=False)[True]
  except: 
    count_left_relation = 0
    
  try:
    count_bidirectional_relation = pd.value_counts(list_of_arcs['BiD'].values, sort=False)[True]
  except: 
    count_bidirectional_relation = 0
  
  count_not_going_right = nb_graphs - count_right_relation # total - ceux qui sont déjà vers la droite
  count_not_going_left = nb_graphs - count_left_relation # total - ceux qui sont déjà vers la gauche

  count_ts_bid = list_of_arcs['nb_ts_bid'].sum()
  count_ts_left = list_of_arcs['nb_ts<-'].sum()
  count_ts_right = list_of_arcs['nb_ts->'].sum()

  # tout supprimer
  edits_delete_everything = count_ts_left + count_ts_right

  # mettre tout à ->
  edits_everything_right = count_ts_left + count_not_going_right

  # mettre tout à <-
  edits_everything_left = count_ts_right + count_not_going_left

  # mettre tout en bidirectional
  edits_everything_bidirectional = count_not_going_right + count_not_going_left

  edits = np.min([edits_delete_everything, edits_everything_left, edits_everything_right, edits_everything_bidirectional])
  
  index_min = np.argmin([edits_delete_everything, edits_everything_left, edits_everything_right, edits_everything_bidirectional])

  #TODO: voir pour les cas d'égalité
    
  return edits

