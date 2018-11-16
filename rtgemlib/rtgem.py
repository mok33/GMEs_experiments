import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np
import copy

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
        cp.dpd_graph = copy.deepcopy(self.dpd_graph)
        return cp

    def size(self):
        return np.array([2 ** len(self.get_node_parents_timescales(nd)) for nd in self.dpd_graph.nodes]).sum()

    def get_edge_timescales_horrizon(self, edge):
        return self.dpd_graph.edges[edge]['timescales'][-1][-1]

    def add_edge_operator(self, edge):
        previous_lambdas = self.get_lambdas(edge[1]) # get lambdas of previous configuration
        
        timescale = [0, self.default_end_timescale]
        self.dpd_graph.add_edge(*edge, timescales=[timescale])

        self.initLambdas(edge[1])
        
        return previous_lambdas
        
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

    def backward_neighbors_gen(self):
        edges = self.dpd_graph.edges
        for edge in edges:
            nbg_1 = self.copy()
            if nbg_1.inverse_add_edge_operator(edge):
                yield 'inverse add_edge {}'.format(edge), nbg_1
                # self.add_edge_operator(edge)

            nbg_2 = self.copy()
            if nbg_2.inverse_extend_operator(edge):
                yield 'inverse_extend {}'.format(edge), nbg_2
                # self.extend_operator(edge)

            for i_tm, timescale in enumerate(self.dpd_graph.edges[edge]['timescales']):
                nbg_3 = self.copy()
                if nbg_3.inverse_split_operator(edge, i_tm):
                    yield 'inverse split e={}  t={}'.format(edge, timescale), nbg_3
                    # self.split_operator(edge, timescale)

    # PARENT COUNT pl à finir
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
