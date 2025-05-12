import unittest
import networkx as nx
import matplotlib.pyplot as plt
from fac_graphs.graph_utils import graph_to_nx, draw_graph
from fac_graphs.models import FactorGraph, FactorNode, VariableNode


class TestGraphUtils:
    @unittest.fixture
    def sample_factor_graph(self):
        v1 = VariableNode("v1")
        v2 = VariableNode("v2")
        f1 = FactorNode([v1, v2], lambda x: x, "f1")
        graph = FactorGraph([f1])
        return graph

    def test_graph_to_nx(self, sample_factor_graph):
        nx_graph = graph_to_nx(sample_factor_graph)

        assert isinstance(nx_graph, nx.Graph)
        assert len(nx_graph.nodes) == 3  # 2 variables + 1 factor
        assert len(nx_graph.edges) == 2  # factor connected to both variables

    def test_graph_to_nx_node_types(self, sample_factor_graph):
        nx_graph = graph_to_nx(sample_factor_graph)

        factor_nodes = [n for n in nx_graph.nodes if isinstance(n, FactorNode)]
        variable_nodes = [n for n in nx_graph.nodes if isinstance(n, VariableNode)]

        assert len(factor_nodes) == 1
        assert len(variable_nodes) == 2

    def test_draw_graph(self, sample_factor_graph, monkeypatch):
        # Mock plt.show to avoid opening a plot window during tests
        monkeypatch.setattr(plt, "show", lambda: None)

        # Test that the function runs without errors
        draw_graph(sample_factor_graph)

    def test_draw_graph_with_figsize(self, sample_factor_graph, monkeypatch):
        monkeypatch.setattr(plt, "show", lambda: None)
        monkeypatch.setattr(plt, "figure", lambda figsize: None)

        # Test that the function accepts a figsize parameter
        draw_graph(sample_factor_graph, figsize=(10, 10))

    def test_draw_graph_node_shapes(self, sample_factor_graph, monkeypatch):
        draw_calls = []
        monkeypatch.setattr(plt, "show", lambda: None)
        monkeypatch.setattr(
            nx, "draw_networkx_nodes", lambda *args, **kwargs: draw_calls.append(kwargs)
        )

        draw_graph(sample_factor_graph)

        assert any(
            call["node_shape"] == "s" for call in draw_calls
        )  # Square for FactorNode
        assert any(
            call["node_shape"] == "o" for call in draw_calls
        )  # Circle for Variable
