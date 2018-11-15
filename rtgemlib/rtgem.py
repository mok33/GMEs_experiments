import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np


class RTGEM:

    def __init__(self, model, default_end_timescale=10):
        self.dpd_graph = nx.DiGraph()
        self.set_graph_from_dict(model)
        self.default_end_timescale = default_end_timescale

    def set_graph_from_dict(self, model):
        for node, attrs in model.items():
            self.dpd_graph.add_node(node, lambdas=attrs['lambdas'])
            for parent_node, parent_timescales in attrs['timescales'].items():
                self.dpd_graph.add_edge(
                    parent_node, node, timescales=parent_timescales)

    def get_node_parents_timescales(self, node):
        return [self.dpd_graph.get_edge_data(pa, node)['timescales'] for pa in self.dpd_graph.predecessors(node)]

    def get_node_parents_timeseries(self, node):
        return [self.dpd_graph.node[pa]['timeserie'] for pa in self.dpd_graph.predecessors(node)]

    def get_node_parents(self, node):
        return list(self.dpd_graph.predecessors(node))

    def get_lambda(self, node, pcv):
        return self.dpd_graph.node[node]['lambdas'][pcv]

    def set_lambda(self, node, pcv, lm):
        self.dpd_graph.node[node]['lambdas'][pcv] = lm

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
        self.dpd_graph.add_edge(*edge, timescales=timescale)

    def extend_operator(self, edge):
        t_h = self.get_edge_timescales_horrizon(edge)
        self.dpd_graph.edges[edge]['timescales'].append(
            [t_h, 2 * t_h])
