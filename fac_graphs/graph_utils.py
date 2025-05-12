"""Utilities for visualization and otherwise."""

from typing import Optional
import matplotlib.pyplot as plt  # type: ignore[import]
import networkx as nx  # type: ignore[import]
from .models.factor_graphs import FactorGraph, FactorNode, AbstractNode


def graph_to_nx(graph: FactorGraph) -> nx.Graph:
    """
    Convert the graph to a networkx graph for visualizing.

    Args:
        - graph: the Graph to convert

    Returns:
        - A networkx graph
    """
    nx_graph = nx.Graph()
    nodes: set[AbstractNode] = set()
    edges = []
    for factor in graph.factors:
        nodes.add(factor)
        nodes.update(factor.connected_variables)
        edges.extend([(factor, v) for v in factor.connected_variables])

    nx_graph.add_nodes_from(nodes)
    nx_graph.add_edges_from(edges)
    return nx_graph


def draw_graph(
    graph: FactorGraph, figsize: Optional[tuple[float, float]] = None
) -> None:
    """
    Draw a thermal graph using networkx.

    Draws FactorNodes as squares. All other nodes are drawn as circles.
    This helps to visualize the graph, if it is a FactorgGraph.

    Args:
        - graph: the Graph to draw
        - figsize: the size of the figure
    """
    plt.figure(figsize=figsize)
    nx_graph = graph_to_nx(graph)
    pos = nx.spring_layout(nx_graph)
    nx.draw_networkx_edges(nx_graph, pos)
    for node in nx_graph.nodes():
        if isinstance(node, FactorNode):
            nx.draw_networkx_nodes(
                nx_graph, pos, nodelist=[node], node_shape="s", node_color="tomato"
            )
        else:
            nx.draw_networkx_nodes(
                nx_graph, pos, nodelist=[node], node_shape="o", node_color="skyblue"
            )

    nx.draw_networkx_labels(nx_graph, pos, labels={n: n.name for n in nx_graph.nodes()})
    plt.show()
