"""GNNs."""

from typing import Optional, Callable, Any, Union, Sequence
import jax
from jax import numpy as jnp
from jaxtyping import Float, Array, PRNGKeyArray
import equinox as eqx  # type: ignore[import]


class GCNLayer(eqx.Module):
    r"""
    Graph convolutional layer.

    Based on https://arxiv.org/abs/1609.02907.

    Computes: $ D^{-\frac{1}{2}} A D^{-\frac{1}{2}} H W + b $
    where $D_{i,i} = \sum_j A_{i, j}$, A is the adjacency matrix, H is the previous layer
    outputs, W is the weights and b is the bias.

    Currently only supports dense matrix multiplication of adjacency matrix.

    Inspired by and adapted from: https://github.com/tkipf/pygcn/tree/master.

    Attributes:
        - weight: the weight matrix of the layer
        - bias: the bias vector
        - use_bias: whether or not to use a bias vector
    """

    weight: jnp.ndarray
    bias: Optional[jnp.ndarray]
    use_bias: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: Optional[PRNGKeyArray] = None,
        use_bias: bool = True,
    ) -> None:
        """
        Initialize weights and biases.

        Args:
            - in_features: the number of (per node) input features
            - out_features: the number of (per node) output features
            - key: the random key to use for initialization
            - use_bias: whether or not to use a bias term

        Returns:
            - None
        """
        if key is None:
            raise ValueError("key must be specified")
        key1, subkey = jax.random.split(key, 2)
        stdv = 1 / jnp.sqrt(out_features)
        self.weight = stdv * jax.random.uniform(
            key1, minval=-1, maxval=1, shape=(in_features, out_features)
        )
        if use_bias:
            self.bias = stdv * jax.random.uniform(
                subkey, minval=-1, maxval=1, shape=(out_features,)
            )
        else:
            self.bias = None
        self.use_bias = use_bias

    def __call__(
        self,
        h_input: Float[Array, "nodes in_features"],
        adj: Float[Array, "nodes nodes"],
    ) -> Float[Array, "nodes out_features"]:
        """
        Compute forward pass of GCN layer.

        Args:
            - h_input: the node inputs
            - adj: the adjacency matrix

        Returns:
            - the output set of node features
        """
        # This is D^-1 A H W
        # num_neighbors = jnp.sum(adj, axis=-1, keepdims=True)
        # support = jnp.matmul(h_input, self.weight)
        # output = jnp.matmul(adj, support) / num_neighbors
        num_neighbors = jnp.sum(adj, axis=-1)
        d_out = jnp.diag(num_neighbors) ** -0.5
        d_out = jnp.where(d_out == jnp.inf, 0, d_out)
        support = h_input @ self.weight
        output = d_out @ adj @ d_out @ support
        if self.use_bias:
            return output + self.bias
        return output


class GCNNetwork(eqx.Module):
    """
    Define a network of GCN layers.

    Attributes:
        - gcn_in: the first GCN layer
        - gcn_hidden: the list of any hidden layers, length of depth - 2
        - gcn_out: the output GCN layer
        - dropout: dropout layer
        - activation_fn: the activation function used between layers
        - classification: whether or not to perform a log_softmax operation at the end
    """

    gcn_in: GCNLayer
    gcn_hidden: list
    gcn_out: GCNLayer
    dropout: eqx.nn.Dropout
    activation_fn: Callable
    classification: bool

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        depth: int,
        act_fn: Callable = jax.nn.elu,
        classification: bool = False,
        dropout_rate: float = 0.1,
        key: Optional[PRNGKeyArray] = None,
    ) -> None:
        """
        Instantiate GCN layers and set member variables.

        Args:
            - nfeat: the number of input features
            - nhid: the number of hidden features
            - nclass: the number of output features per node
            - depth: the total number of layers (including both the input and output layer, so
                depth = 2 is the minimum)
            - act_fn: the activation function to use between layers
            - classification: whether or not to perform a log_softmax before returning the node
                values
            - dropout_rate: the rate to use for the dropout layer
            - key: the random key to use for initialization

        Returns:
            - None
        """
        if key is None:
            raise ValueError("key must be specified")
        key1, subkey1, subkey2 = jax.random.split(key, 3)
        self.gcn_in = GCNLayer(nfeat, nhid, key=subkey1)
        self.gcn_out = GCNLayer(nhid, nclass, key=subkey2)
        keys = jax.random.split(key1, depth)
        self.gcn_hidden = [
            GCNLayer(nhid, nhid, key=keys[i]) for i in range(max(0, depth - 2))
        ]
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.activation_fn = act_fn
        self.classification = classification

    def __call__(
        self,
        x: Float[Array, "nodes nfeat"],
        adj: Float[Array, "nodes nodes"],
        enable_dropout: Optional[bool] = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "nodes nclass"]:
        """
        Compute a forward pass of the GCN network.

        Args:
            - x: the input node feature matrix
            - adj: the adjacency matrix
            - enable_dropout: whether or not to use dropout
            - key: the random key to use if using dropout

        Returns:
            - the computes values per node
        """
        x = self.activation_fn(self.gcn_in(x, adj))
        x = self.dropout(x, inference=not enable_dropout, key=key)
        for layer in self.gcn_hidden:
            x = self.activation_fn(layer(x, adj))
            if enable_dropout and key is not None:
                key, subkey = jax.random.split(key, 2)
                x = self.dropout(x, inference=not enable_dropout, key=subkey)
        x = self.gcn_out(x, adj)
        if self.classification:
            return jax.nn.log_softmax(x)
        return x


class GATLayer(eqx.Module):
    r"""
    Graph Attention Layer.

    Based on https://arxiv.org/abs/1710.10903.

    Inspired by and adapted from: https://github.com/Diego999/pyGAT/blob/master/layers.py
    and
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial7/GNN_overview.html

    Computes: $ h_{i} = \sum_{j \in N} a_{i, j} W h_{j} $ where $h_{i}$ is the
    features of node $i$, $W$ is a weight matrix and
    $a_{i, j} = softmax(a^T concat(W h_i, W h_j))$ are the attention weights.

    See equations 3-6 in the paper for more information.

    Attributes:
        - n_heads: the number of heads of attention
        - alpha: the constant used in leaky relu
        - concat: whether or not the final output should be concatenated
        - dropout: the dropout layer
        - W: the weight matrix W
        - a: the weight matrix a
        - b: the bias
    """

    n_heads: int
    alpha: float
    concat: bool
    dropout: eqx.nn.Dropout
    weight: jnp.ndarray
    attention: jnp.ndarray
    bias: Optional[jnp.ndarray]
    use_bias: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int,
        dropout_rate: float = 0.1,
        alpha: float = 0.2,
        use_bias: bool = False,
        concat: bool = True,
        key: Optional[PRNGKeyArray] = None,
    ) -> None:
        """
        Initialize weights and biases.

        Args:
            - in_features: the number of input features per node
            - out_features: the number of output features per node
            - n_heads: the number of attention heads
            - dropout_rate: the probability of dropout
            - alpha: the leaky relu parameter
            - use_bias: whether or not to use a bias term
            - concat: whether the heads should be concatenated together at the end
                or just averaged over. If this is true, the number of out features mod
                the number of heads must be 0
            - key: the random key to use for initialization

        Returns:
            - None

        Raises:
            - ValueError: if out_features % n_heads != 0
        """
        if key is None:
            raise ValueError("key must be specified")
        if concat:
            if out_features % n_heads != 0:
                raise ValueError("out_features % n_heads must equal 0!")
            out_per_head = out_features // n_heads
        else:
            out_per_head = out_features

        self.alpha = alpha
        self.concat = concat
        self.n_heads = n_heads

        self.dropout = eqx.nn.Dropout(p=dropout_rate)
        w_key, a_key = jax.random.split(key, 2)
        self.weight = jax.nn.initializers.glorot_uniform()(
            w_key, (in_features, out_per_head * n_heads), jnp.float32
        )
        if use_bias:
            self.bias = jnp.zeros(shape=(out_features,))
        else:
            self.bias = None
        self.use_bias = use_bias
        self.attention = jax.nn.initializers.glorot_uniform()(
            a_key, (n_heads, 2 * out_per_head), jnp.float32
        )

    def __call__(
        self,
        h_mat: Float[Array, "nodes in_features"],
        adj: Float[Array, "nodes nodes"],
        enable_dropout: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "nodes out_features"]:
        """
        Compute a forward pass of the GAT layer.

        This is eq (4) or (5) (depending on whether concat is true).

        Args:
            - h_mat: the node feature matrix
            - adj: the adjacency matrix
            - enable_dropout: whether or not to use dropout
            - key: the random key to use with dropout

        Returns:
            - the new node feature matrix
        """
        node_features = jnp.matmul(h_mat, self.weight).reshape(
            (h_mat.shape[0], self.n_heads, -1)
        )  # -> (nodes, nheads, out_per_head)
        attention_logits = self._prepare_attention(
            node_features
        )  # -> (nodes, nodes, nheads)
        zero_vec = -9e15 * jnp.ones_like(attention_logits)
        attention = jnp.where(adj[..., None] > 0, attention_logits, zero_vec)
        attention = jax.nn.softmax(attention, axis=1)  # -> (nodes, nodes, nheads)
        attention = self.dropout(attention, inference=not enable_dropout, key=key)
        h_prime = jnp.einsum(
            "njh,jhf->nhf", attention, node_features
        )  # -> (nodes, nheads, out_per_head)
        if self.concat:
            return h_prime.reshape(h_mat.shape[0], -1)  # -> (nodes, out_features)
        return jnp.mean(h_prime, axis=1)  # -> (nodes, out_features)

    def _prepare_attention(
        self, wh_mat: Float[Array, "nodes nheads out_features"]
    ) -> Float[Array, "nodes nodes nheads"]:
        """
        Generate the attention function $e_{i,j}$, or eq 1 in the paper (without the softmax).

        Args:
            - wh_mat: the nodes and heads matrix

        Returns:
            - the attented to nodes matrix
        """
        logit_parent = jnp.sum(
            wh_mat * jnp.squeeze(self.attention[:, : self.attention.shape[1] // 2]),
            axis=-1,
        )  # -> (nodes, nheads)
        logit_child = jnp.sum(
            wh_mat * jnp.squeeze(self.attention[:, self.attention.shape[1] // 2 :]),
            axis=-1,
        )  # -> (nodes, nheads)
        e_out = (
            logit_parent[:, None, :] + logit_child[None, :, :]
        )  # -> (nodes, nodes, nheads)
        return jax.nn.leaky_relu(e_out, self.alpha)


class GATNetwork(eqx.Module):
    """
    Graph Attention Network composed of GAT layers.

    Attributes:
        - in_attn: the input GAT layer
        - attentions: the hidden GAT layers
        - out_attn: the output GAT layer
        - dropout: the dropout layer
        - classification: whether or not to perform a log_softmax operation at the end
    """

    in_attn: GATLayer
    attentions: list
    out_attn: GATLayer
    dropout: eqx.nn.Dropout
    classification: bool

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        depth: int,
        nheads: Union[int, Sequence[int]],
        dropout_rate: float = 0.1,
        classification: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ):
        """
        Instantiate GAT layers and set member variables.

        Args:
            - nfeat: the number of input features
            - nhid: the number of hidden features
            - nclass: the number of output features per node
            - n_heads: the number of attention heads, can be 1 value for every layer,
                or a list of per layer n_heads
            - depth: the total number of layers (including both the input and output layer, so
                depth = 2 is the minimum)
            - dropout_rate: the rate to use for the dropout layer
            - classification: whether or not to perform a log_softmax operation at the end
            - key: the random key to use for initialization

        Returns:
            - None
        """
        if key is None:
            raise ValueError("key must be specified")
        key1, subkey1, subkey2 = jax.random.split(key, 3)
        keys = jax.random.split(key1, depth)
        if isinstance(nheads, int):
            nheads = [nheads] * max(depth, 2)
        self.in_attn = GATLayer(nfeat, nhid, nheads[0], dropout_rate, key=subkey1)
        self.attentions = [
            GATLayer(nhid, nhid, nheads[i + 1], dropout_rate, key=keys[i])
            for i in range(depth - 2)
        ]
        self.out_attn = GATLayer(
            nhid, nclass, nheads[-1], dropout_rate, concat=False, key=subkey2
        )
        self.dropout = eqx.nn.Dropout(p=dropout_rate)
        self.classification = classification

    def __call__(
        self,
        x: Float[Array, "nodes nfeat"],
        adj: Float[Array, "nodes nodes"],
        enable_dropout: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "nodes nclass"]:
        """
        Compute the forward pass of the GAT network.

        Specifically, the repeatedly/iteratively computers eq (5), followed by
        the final layer computing eq (6) in https://arxiv.org/abs/1710.10903.

        Args:
            - x: the node features inputs
            - adj: the adjaceny matrix
            - enable_dropout: whether or not to enable dropout
            - key: the random key to use for dropout

        Returns:
            - the predicted value for each output feature for each node
        """
        if enable_dropout:
            if key is None:
                raise ValueError("key must be specified")
            key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        else:
            key, subkey1, subkey2, subkey3 = None, None, None, None
        x = self.in_attn(x, adj, enable_dropout, subkey1)
        for layer in self.attentions:
            if key is not None:
                key, subkey = jax.random.split(key, 2)
            x = jax.nn.elu(layer(x, adj, enable_dropout, subkey))
        x = self.dropout(x, inference=not enable_dropout, key=subkey2)
        x = jax.nn.elu(self.out_attn(x, adj, enable_dropout, subkey3))
        if self.classification:
            return jax.nn.log_softmax(x)
        return x


class GNNEnergy(eqx.Module):
    """
    Wrapper for GNNs to make them energy functions.

    Attributes:
        - gnn: the graph neural network to use
        - linear: the linear layer to convert the output to a single energy value
    """

    gnn: eqx.Module
    linear: eqx.nn.Linear

    def __init__(
        self, num_nodes: int, gnn: eqx.Module, key: Optional[PRNGKeyArray] = None
    ) -> None:
        """
        Assign member variables.

        Args:
            - num_nodes: the number of nodes in the GNN
            - gnn: the GNN to use, the output shape must be (nodes, 1) or (nodes,)
            - key: the random key to use for random initialization

        Returns:
            - None
        """
        self.gnn = gnn
        if key is None:
            raise ValueError("key cannot be None")
        self.linear = eqx.nn.Linear(num_nodes, 1, key=key)

    def __call__(
        self,
        x: Float[Array, "nodes features"],
        adj: Float[Array, "nodes nodes"],
        **kwargs: Any,
    ) -> Float[Array, ""]:
        """
        Forward pass of the energy function.

        Computes linear(swish(gnn(x))).

        Args:
            - x: the input node features
            - adj: the adjacency matrix

        Returns:
            - an energy of the nodes/adjacency matrix
        """
        if not callable(self.gnn):
            raise ValueError("self.gnn must be callable")
        x = self.gnn(x, adj, **kwargs)
        x = jax.nn.swish(x)
        x = jnp.squeeze(x)
        x = self.linear(x)
        return jnp.squeeze(x)
