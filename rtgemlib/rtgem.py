import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np
import copy
import itertools
import random
from tqdm.autonotebook import tqdm


class RTGEM:

    def __init__(self, model, default_end_timescale=10, default_lambda=1):
        self.dpd_graph = nx.DiGraph()
        self.set_graph_from_dict(model)
        self.default_end_timescale = default_end_timescale
        self.default_lambda = default_lambda

    def set_graph_from_dict(self, model):
        for node, attrs in model.items():
            self.dpd_graph.add_node(node, lambdas=attrs['lambdas'])
            for parent_node, parent_timescales in attrs['timescales'].items():
                self.dpd_graph.add_edge(
                    parent_node, node, timescales=parent_timescales)

    def get_node_nb_parents_timescales(self, node):
        return sum([len(self.dpd_graph.get_edge_data(pa, node)['timescales']) for pa in self.dpd_graph.predecessors(node)])

    def get_node_parents_timescales(self, node):
        return [self.dpd_graph.get_edge_data(pa, node)['timescales'] for pa in self.dpd_graph.predecessors(node)]

    def get_node_parents_timeseries(self, node):
        return [self.dpd_graph.nodes[pa]['timeserie'] for pa in self.dpd_graph.predecessors(node)]

    def get_parents_count(self, node):
        return [self.dpd_graph.get_edge_data(pa, node)['parent_count'] for pa in self.dpd_graph.predecessors(node)]

    def get_node_parents(self, node):
        return list(self.dpd_graph.predecessors(node))

    def get_lambda(self, node, pcv):
        return self.dpd_graph.nodes[node]['lambdas'][pcv]

    def get_lambdas(self, node):
        return self.dpd_graph.nodes[node]['lambdas']

    def set_node_timeserie(self, node, ts):
        self.dpd_graph.nodes[node]['timeserie'] = ts

    def set_lambda(self, node, pcv, lm):
        self.dpd_graph.nodes[node]['lambdas'][pcv] = lm

    def initLambdas(self, node, random=True, a=0, b=10):
        nb_parent = self.get_node_nb_parents_timescales(node)
        if random:
            self.dpd_graph.nodes[node]['lambdas'] = dict(zip(list(itertools.product(
                *[[0, 1]] * nb_parent)), np.random.uniform(a, b, np.power(2, nb_parent))))

        else:
            self.dpd_graph.nodes[node]['lambdas'] = dict(zip(list(itertools.product(
                *[[0, 1]] * nb_parent)), [self.default_lambda] * np.power(2, nb_parent)))

    def copy(self):
        cp = RTGEM({})
        cp.dpd_graph = copy.deepcopy(self.dpd_graph)
        return cp

    def size(self):
        return np.array([2 ** self.get_node_nb_parents_timescales(nd) for nd in self.dpd_graph.nodes]).sum()

    def get_edge_timescales_horrizon(self, edge):
        return self.dpd_graph.edges[edge]['timescales'][-1][-1]

    def add_edge_operator(self, edge):
        timescale = [0, self.default_end_timescale]
        self.dpd_graph.add_edge(*edge, timescales=[timescale])

        self.initLambdas(edge[1])

    def remove_edge_operator(self, edge):
        self.dpd_graph.remove_edge(*edge)

    def extend_operator(self, edge):
        t_h = self.get_edge_timescales_horrizon(edge)
        self.dpd_graph.edges[edge]['timescales'].append(
            [t_h, 2 * t_h])
        self.initLambdas(edge[1])

    def split_operator(self, edge, timescale):
        index = self.dpd_graph.edges[edge]['timescales'].index(timescale)

        del self.dpd_graph.edges[edge]['timescales'][index]

        first_half_timescale = [timescale[0],
                                (timescale[1] + timescale[0]) / 2]
        second_half_timescale = [
            (timescale[1] + timescale[0]) / 2, timescale[1]]

        self.dpd_graph.edges[edge]['timescales'].insert(
            index, first_half_timescale)
        self.dpd_graph.edges[edge]['timescales'].insert(
            index + 1, second_half_timescale)
        self.initLambdas(edge[1])

    def add_edge_operator_was_applied(self, edge):
        timescales = self.dpd_graph.edges[edge]['timescales']
        return len(timescales) == 1 and timescales[0][0] == 0 and timescales[0][1] == self.default_end_timescale

    def inverse_add_edge_operator(self, edge):
        is_possible = self.add_edge_operator_was_applied(edge)
        if is_possible:
            self.dpd_graph.remove_edge(*edge)
            self.initLambdas(edge[1])

        return is_possible

    def extend_operator_was_applied(self, edge):
        timescales = self.dpd_graph.edges[edge]['timescales']
        th, th_2 = timescales[-1]
        return len(timescales) >= 2 and timescales[-2][-1] == th and timescales[-2][-1] * 2 == th_2
    # backward operator

    def inverse_extend_operator(self, edge):
        is_possible = self.extend_operator_was_applied(edge)

        if is_possible:
            self.dpd_graph.edges[edge]['timescales'].pop()
            self.initLambdas(edge[1])

        return is_possible

    def split_operator_was_applied(self, tm1, tm2):
        a, m1 = tm1
        m2, b = tm2

        return m1 == m2 and m1 == (a + b) / 2

    def inverse_split_operator(self, edge, tm_idx):

        is_possible = False
        if tm_idx < len(self.dpd_graph.edges[edge]['timescales']) - 1:
            tm1, tm2 = self.dpd_graph.edges[edge][
                'timescales'][tm_idx:tm_idx + 2]
            is_possible = self.split_operator_was_applied(tm1, tm2)

        if is_possible:
            a = self.dpd_graph.edges[edge]['timescales'][tm_idx][0]
            self.dpd_graph.edges[edge]['timescales'][tm_idx + 1][0] = a
            self.dpd_graph.edges[edge]['timescales'].pop(tm_idx)
            self.initLambdas(edge[1])
        return is_possible

    def random_walk(self, max_depth, t_n=30):
        """
        Performs a random walk in the exploration graph, by randomly applying an
        operator to an edge/timescale.
        max_depth: the size of
        """

        edges_to_add = list(
            (itertools.product(list(self.dpd_graph.nodes), repeat=2)))
        edges_to_extend = []

        random.shuffle(edges_to_add)
        valid_operations = [1, 2, 3]

        if max_depth > 0:
            e = edges_to_add.pop(0)
            self.add_edge_operator(e)
            edges_to_extend.append(e)
            if len(edges_to_add) == 0:
                valid_operations.remove(1)
            max_depth -= 1

        for i in tqdm(range(max_depth), leave=False):
            op_num = random.choice(valid_operations)
            if op_num == 1:
                e = edges_to_add.pop(0)
                self.add_edge_operator(e)
                edges_to_extend.append(e)

                if len(edges_to_add) == 0:
                    valid_operations.remove(1)

            if op_num == 2:
                random_edge_num = random.randint(0, len(edges_to_extend) - 1)
                random_edge = edges_to_extend[random_edge_num]
                if 2 * self.get_edge_timescales_horrizon(random_edge) > t_n:
                    edges_to_extend.pop(random_edge_num)
                else:
                    self.extend_operator(random_edge)

                if len(edges_to_extend) == 0:
                    valid_operations.remove(2)

            if op_num == 3:
                random_edge = random.choice(list(self.dpd_graph.edges))
                random_tms = random.choice(
                    list(self.dpd_graph.edges[random_edge]['timescales']))

                self.split_operator(random_edge, random_tms)
    
    def print_rtgem(self):
        nodes = ""
        for node in self.dpd_graph.nodes:
            nodes += str(node) + " "
            
        print("Nodes:", nodes)
        
        print("Edges:")
        for edge in self.dpd_graph.edges:
            print(edge[0], "->", edge[1])


    


def pair2list(list_timescales):
  
  list = [pair[0] for pair in list_timescales]
  list.append(list_timescales[len(list_timescales)-1][1])
  
  return list

def distance_between_timescales(ts1, ts2):
    vid = []
    vnid = []

    for elem in ts1:
        if elem in ts2: 
            vid.append(elem)
        else: 
            vnid.append(elem)

        for elem in ts2:
            if elem not in ts1: vnid.append(elem)
        
    distance_btn_timescales = len(vnid)/(len(vnid)+len(vid))
        
    return distance_btn_timescales

def symmetric_difference(list_of_rtgems):
    """
    finds edges that are in one graph and not the others
    """
    Esd = []
    
    all_edges = [list(list_of_rtgems[i].dpd_graph.edges()) for i in range(len(list_of_rtgems))]
    
    # find arcs that are different
    for i in range(len(list_of_rtgems)):
      edges_current_graph = all_edges[i] 
      edges_other_graphs = all_edges[0:i] + all_edges[i:len(list_of_rtgems)]
      
      for edge in edges_current_graph:
        if  ((edge not in edges_other_graphs) and (edge not in Esd)):
          Esd.append(edge)
        
    return Esd

def SHD(GEM1, GEM2):
    """
    Distance measure between two RTGEMs
    """
    Esd = symmetric_difference([GEM1, GEM2])
    Einter = []
    
      # find distance of timescales for arcs that are in common    
    for edge in list(GEM1.dpd_graph.edges()):
      if edge in list(GEM2.dpd_graph.edges()):
        
        # Einter contains pairs of lists of timescales of the arcs that are common to both graphs 
        ts_rtgem1 = pair2list(GEM1.get_node_parents_timescales(edge[1])[GEM1.get_node_parents(edge[1]).index(edge[0])])
        ts_rtgem2 = pair2list(GEM2.get_node_parents_timescales(edge[1])[GEM2.get_node_parents(edge[1]).index(edge[0])])
        Einter.append(distance_between_timescales(ts_rtgem1, ts_rtgem2))
 
    distance_sd = len(Esd)
    distance_inter = sum(Einter)

    return distance_sd + distance_inter # total distance 


def empty_nodes(nodes):
    return dict(zip(nodes, [{'timescales': {}, 'lambdas': {(): 1}}] * len(nodes)))