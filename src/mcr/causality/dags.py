import numpy as np
from typing import List
import matplotlib.pyplot as plt
import networkx as nx

from mcr.utils import search_nonsorted


class DirectedAcyclicGraph:
    """
    Directed acyclic graph, used to define Structural Equation Model
    """

    def __init__(self, adjacency_matrix: np.array, var_names: List[str]):
        """
        Args:
            adjacency_matrix: Square adjacency matrix
            var_names: List of variable input_var_names
        """
        assert adjacency_matrix.shape[0] == adjacency_matrix.shape[1]
        assert adjacency_matrix.shape[0] == len(var_names)

        adjacency_matrix = adjacency_matrix.astype(int)
        self.DAG = nx.convert_matrix.from_numpy_matrix(
            adjacency_matrix, create_using=nx.DiGraph
        )
        assert nx.algorithms.dag.is_directed_acyclic_graph(self.DAG)

        self.var_names = np.array(var_names, dtype=str)

    def get_markov_blanket(self, node: str) -> set:
        return self.get_parents(node) | self.get_children(node) | self.get_spouses(node)

    def get_parents(self, node: str) -> set:
        node_ind = search_nonsorted(self.var_names, [node])[0]
        parents = tuple(self.DAG.predecessors(node_ind))
        return set([self.var_names[node] for node in parents])

    def __remove_edge(self, u: str, v: str):
        u_ind = search_nonsorted(self.var_names, [u])[0]
        v_ind = search_nonsorted(self.var_names, [v])[0]
        self.DAG.remove_edge(u_ind, v_ind)

    def do(self, intv_vars):
        for node in intv_vars:
            pars = self.get_parents(node)
            for par in pars:
                self.__remove_edge(par, node)

    def get_children(self, node: str) -> set:
        node_ind = search_nonsorted(self.var_names, [node])[0]
        children = tuple(self.DAG.successors(node_ind))
        return set([self.var_names[node] for node in children])

    def get_spouses(self, node: str) -> set:
        node_ind = search_nonsorted(self.var_names, [node])[0]
        children = tuple(self.DAG.successors(node_ind))
        spouses = tuple(
            [
                par
                for child in children
                for par in tuple(self.DAG.predecessors(child))
                if par != node_ind
            ]
        )
        return set([self.var_names[node] for node in spouses])

    def get_ancestors_node(self, node: str) -> set:
        node_ind = search_nonsorted(self.var_names, [node])[0]
        ancestors_ind = tuple(nx.ancestors(self.DAG, node_ind))
        ancestors = set([self.var_names[ii] for ii in ancestors_ind])
        return ancestors

    def __get_nondescendants_node(self, node: str) -> set:
        node_ind = search_nonsorted(self.var_names, [node])[0]
        descendants_ind = tuple(nx.descendants(self.DAG, node_ind))
        descendants = set([self.var_names[ndi] for ndi in descendants_ind])
        descendants.add(node)
        nondescendants = set([nd for nd in self.var_names if nd not in descendants])
        return nondescendants

    def get_nondescendants(self, nodes: set) -> set:
        ndss = [self.__get_nondescendants_node(node) for node in nodes]
        ndss.append(set(self.var_names))
        return set.intersection(*ndss)

    def plot_dag(self, ax=None):
        """
        Plots DAG with networkx tools
        Args:
            ax: Matplotlib axis
        """
        if ax is None:
            fig, ax = plt.subplots()
        labels_dict = {i: self.var_names[i] for i in range(len(self.DAG))}
        nx.draw_networkx(
            self.DAG,
            pos=nx.kamada_kawai_layout(self.DAG),
            ax=ax,
            labels=labels_dict,
            node_color="white",
            arrowsize=15,
            edgecolors="b",
            node_size=800,
        )

    @staticmethod
    def random_dag(n, p=None, m_n_ratio=None, seed=None, model="np"):
        """
        Creates random Erdős-Rényi graph from G(size, p) sem
        (see https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model)

        Args:
            seed: Random mc_seed
            n: Number of nodes
            p: Probability of creating an edge in 'np' model
            m_n_ratio: m/n ratio, m is the number of edges in 'nm' model
            model: 'np' - G(n, p) / 'nm' - G(n, m)

        Returns: DirectedAcyclicGraph instance

        """
        
        if model == "np":
            G = nx.gnp_random_graph(n, p, seed, directed=True)
        elif model == "nm":
            G = nx.gnm_random_graph(n, int(m_n_ratio * n), seed, directed=True)
        else:
            raise NotImplementedError("Unknown model type")
        G.remove_edges_from([(u, v) for (u, v) in G.edges() if u > v])
        adjacency_matrix = (
            nx.linalg.graphmatrix.adjacency_matrix(G).todense().astype(int)
        )
        var_names = [f"x{i}" for i in range(n)]
        return DirectedAcyclicGraph(adjacency_matrix, var_names)

    def save(self, filepath):
        arr = nx.convert_matrix.to_numpy_array(self.DAG)
        np.save(filepath + "_dag.npy", arr)
        np.save(filepath + "_var_names.npy", self.var_names)

    @staticmethod
    def load(filepath):
        arr = np.load(filepath + "_dag.npy")
        var_names = np.load(filepath + "_var_names.npy")
        dag = DirectedAcyclicGraph(arr, var_names)
        return dag
