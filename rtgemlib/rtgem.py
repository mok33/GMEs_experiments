import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np

import itertools


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
        return [self.dpd_graph.node[pa]['timeserie'] for pa in self.dpd_graph.predecessors(node)]

    def get_node_parents(self, node):
        return list(self.dpd_graph.predecessors(node))

    def get_lambda(self, node, pcv):
        return self.dpd_graph.node[node]['lambdas'][pcv]

    def get_lambdas(self, node):
        return self.dpd_graph.node[node]['lambdas']

    def set_lambda(self, node, pcv, lm):
        self.dpd_graph.node[node]['lambdas'][pcv] = lm

    def initLambdas(self, node):
        nb_parent = self.get_node_nb_parents_timescales(node)

        self.dpd_graph.node[node]['lambdas'] = dict(zip(list(itertools.product(
            *[[0, 1]] * nb_parent)), [self.default_lambda] * np.power(2, nb_parent)))

    def copy(self):
        cp = RTGEM({})
        cp.dpd_graph = self.dpd_graph.copy()
        return cp

    def size(self):
        return np.array([2 ** len(self.get_node_parents_timescales(nd)) for nd in self.dpd_graph.nodes]).sum()

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

    # PARENT COUNT à finir
    # # def set_node_parent_configuration_change_times(self, node, parent, t):
    # #     node_parent_count_vector = ()
    # #     t_expr = np.inf

    # # def get_node_parent_configuration
    # def get_node_parents_configuration(self, node, t):
    #     node_parent_count_vector = ()
    #     t_expr = np.inf

    #     for parent in self.get_node_parents(node):
    #         parent_timeserie = self.dpd_graph.nodes[
    #             parent]['timeserie']
    #         for (a, b) in self.dpd_graph.get_edge_data(parent, node)['timescales']:
    #             count_pa_timescale = (
    #                 ((parent_timeserie > t - b) & (parent_timeserie <= t - a)).sum() >= 1) * 1
    #             node_parent_count_vector += (count_pa_timescale,)
    #             t_expr = min(t_expr, (b - a))

    #     return node_parent_count_vector, t_expr
