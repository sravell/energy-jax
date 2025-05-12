"""Transformers."""

from typing import Optional, List
import jax
from jax import numpy as jnp
from jaxtyping import Float, Array, Int, PRNGKeyArray
import equinox as eqx  # type: ignore[import]


class PositionwiseFeedForwardBlock(eqx.Module):
    """
    A single transformer feed forward block.

    Adapted from: https://docs.kidger.site/equinox/examples/bert/

    Equinox license is available via
     https://github.com/patrick-kidger/equinox/blob/main/LICENSE

    For more technical information see:
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

    A feedforward block that is applied/vmap-ed across each of the sequence elements.

    Attributes:
        - mlp: the first mlp
        - output: the output mlp
        - layernorm: layer normalization applied after the two MLPs
        - dropout: dropout layer applied between the two MLPs
    """

    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        io_size: int,
        dropout_rate: float,
        key: PRNGKeyArray,
    ) -> None:
        """
        Initialize member variables.

        Args:
            - hidden_size: the size of the intermediate representation/output of the first MLP
            - the input and output size of module
            - dropout_rate: the rate of dropout
            - key: the key to use for initializing the networks
        """
        mlp_key, output_key = jax.random.split(key)
        self.mlp = eqx.nn.Linear(
            in_features=io_size, out_features=hidden_size, key=mlp_key
        )
        self.output = eqx.nn.Linear(
            in_features=hidden_size, out_features=io_size, key=output_key
        )

        self.layernorm = eqx.nn.LayerNorm(shape=io_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "seq_len io_size"],
        enable_dropout: Optional[bool] = True,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq_len io_size"]:
        """
        Forward pass of the module.

        Args:
            - inputs: the inputs to the FFN
            - enable_dropout: whether or not to enable dropout
            - key: the random key to use w/ dropout

        Returns:
            - array of the same shape as the input
        """
        # Feed-forward.
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)  # relu in original

        # Project back to input size.
        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Residual and layer norm.
        output += inputs
        output = self.layernorm(output)

        return output


class EncoderBlock(eqx.Module):
    """
    A single encoder block.

    Adapted from: https://docs.kidger.site/equinox/examples/bert/.

    Single global multi-head attention, followed by a residual block and a FFN.

    Attributes:
        - mha: the global MHA network
        - ffn: the feedforward network as previously defined
        - layernorm: layernorm applied after residual connection
    """

    mha: eqx.nn.MultiheadAttention
    ffn: PositionwiseFeedForwardBlock
    layernorm: eqx.nn.LayerNorm

    def __init__(
        self,
        key: PRNGKeyArray,
        n_heads: int = 8,
        d_q: int = 64,
        d_v: int = 64,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
    ) -> None:  # original paper params
        """
        Initialize member variables.

        Args:
            - key: the random key to use for initialization
            - n_heads: the number of heads for MHA
            - d_q: the size of the query
            - d_v: the size of the value
            - d_ff: the intermediate size of the FFN
            - dropout_rate: the rate of dropout

        Returns:
            - None
        """
        mha_key, mlp_key = jax.random.split(key)
        self.mha = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d_q,
            value_size=d_v,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout_rate,
            key=mha_key,
        )
        self.ffn = PositionwiseFeedForwardBlock(
            d_ff, d_q, dropout_rate=dropout_rate, key=mlp_key
        )
        self.layernorm = eqx.nn.LayerNorm(shape=d_q)

    def __call__(
        self,
        inputs: Int[Array, "seq_len d_q"],
        mask: Optional[Int[Array, "seq_len"]] = None,
        enable_dropout: Optional[bool] = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq_len d_q"]:
        """
        Forward pass of the encoder block.

        Args:
            - inputs: the embedded sequence inputs
            - mask: what values to mask, e.g. for padded inputs, which to
                ignore ([1, 1, 0, 0] ignores the last two)
            - enable_dropout: whether or not to use dropout
            - key: the random key for using dropout

        Returns:
            - an array of the same shape as the input
        """
        if mask is not None:
            mask = self.make_self_attention_mask(mask)

        if key is None:
            attention_key, _, ff_key = (None, None, None)
        else:
            attention_key, _, ff_key = jax.random.split(key, 3)

        result = self.mha(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            inference=not enable_dropout,
            key=attention_key,
        )

        result = result + inputs
        result = jax.vmap(self.layernorm)(result)

        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        result = jax.vmap(self.ffn, in_axes=(0, None, 0))(
            result, enable_dropout, ff_keys
        )
        return result

    def make_self_attention_mask(
        self, mask: Int[Array, "seq_len"]
    ) -> Float[Array, "seq_len seq_len"]:
        """
        Create self-attention mask from sequence-level mask.

        Example: make_self_attention_mask([1, 1, 0, 0])

        [1, 1, 0, 0]
        [1, 1, 0, 0]
        [0, 0, 0, 0]
        [0, 0, 0, 0]

        Args:
            - mask: the values to mask out

        Returns:
            - the converted matrix mask
        """
        mask = jnp.multiply(
            jnp.expand_dims(mask, axis=-1), jnp.expand_dims(mask, axis=-2)
        )
        return mask.astype(jnp.float32)


class DecoderBlock(eqx.Module):
    """
    A single block of a decoder.

    A casual MHA followed by a cross attention followed by a FFN.

    Attributes:
        - mha1: the casual self attention
        - mha2: the cross attention
        - ffn: the position feed forward network
        - ln1: the first layer norm (after the first MHA)
        - ln2: the second layer norm (after the second MHA)
    """

    mha1: eqx.nn.MultiheadAttention
    mha2: eqx.nn.MultiheadAttention
    ffn: PositionwiseFeedForwardBlock
    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm

    def __init__(
        self,
        key: PRNGKeyArray,
        n_heads: int = 8,
        d_q: int = 64,
        d_ff: int = 2048,
        dropout_rate: float = 0.1,
    ) -> None:  # original paper params
        """
        Initialize member variables.

        Args:
            - key: the random key to use for initialization
            - n_heads: the number of heads for MHA
            - d_q: the size of the query
            - d_ff: the intermediate size of the FFN
            - dropout_rate: the rate of dropout

        Returns:
            - None
        """
        mha1_key, mha2_key, mlp_key = jax.random.split(key, 3)
        self.mha1 = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d_q,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout_rate,
            key=mha1_key,
        )
        self.mha2 = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d_q,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=dropout_rate,
            key=mha2_key,
        )
        self.ffn = PositionwiseFeedForwardBlock(
            d_ff, d_q, dropout_rate=dropout_rate, key=mlp_key
        )
        self.ln1 = eqx.nn.LayerNorm(shape=d_q)
        self.ln2 = eqx.nn.LayerNorm(shape=d_q)

    def __call__(
        self,
        inputs: Int[Array, "seq_len d_q"],
        encoding: Float[Array, "seq_len d_q"],
        mask: Optional[Int[Array, "seq_len"]] = None,
        enable_dropout: bool = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq_len d_q"]:
        """
        Forward pass of the decoder module.

        Args:
            - inputs: the embedded sequences
            - encoding: the output of the encoder
            - mask: the input values to mask out (e.g. the padded ones)
            - entable_dropout: whether or not to use dropout
            - key: the random key to use for dropout

        Returns:
            - an array the same shape as the input
        """
        causal_mask = self.make_causal_mask(inputs.shape[0])
        if mask is not None:
            input_mask = self.make_self_attention_mask(mask)
            causal_mask = (
                input_mask.astype("int8") | causal_mask.astype("int8")
            ).astype("float32")

        if key is None:
            self_attention_key, cross_attention_key, _, ff_key = (
                None,
                None,
                None,
                None,
            )
        else:
            self_attention_key, cross_attention_key, _, ff_key = jax.random.split(
                key, 4
            )

        result = self.mha1(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=causal_mask,
            inference=not enable_dropout,
            key=self_attention_key,
        )

        result = result + inputs
        result = jax.vmap(self.ln1)(result)

        mha1_output = result.copy()
        result = self.mha2(
            query=result,
            key_=encoding,
            value=encoding,
            # mask=input_mask,
            inference=not enable_dropout,
            key=cross_attention_key,
        )

        result = result + mha1_output
        result = jax.vmap(self.ln2)(result)

        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        result = jax.vmap(self.ffn, in_axes=(0, None, 0))(
            result, enable_dropout, ff_keys
        )
        return result

    def make_self_attention_mask(
        self, mask: Int[Array, "seq_len"]
    ) -> Float[Array, "seq_len seq_len"]:
        """
        Create self-attention mask from sequence-level mask.

        Example: make_self_attention_mask([1, 1, 0, 0])

        [1, 1, 0, 0]
        [1, 1, 0, 0]
        [0, 0, 0, 0]
        [0, 0, 0, 0]

        Args:
            - mask: the values to mask out

        Returns:
            - the converted matrix mask
        """
        mask = jnp.multiply(
            jnp.expand_dims(mask, axis=-1), jnp.expand_dims(mask, axis=-2)
        )
        return mask.astype(jnp.float32)

    def make_causal_mask(self, seq_len: int) -> Float[Array, "seq_len seq_len"]:
        """
        Create a causal mask for self-attention.

        Example: make_causal_mask(4)

        [1, 0, 0, 0]
        [1, 1, 0, 0]
        [1, 1, 1, 0]
        [1, 1, 1, 1]

        Args:
            - mask: the values to mask out

        Returns:
            - the casual matrix
        """
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.float32))
        return mask.astype(jnp.float32)


class PositionEmbedding(eqx.Module):
    """
    Sinusoidal position embedding, as in original transformer paper.

    Adapted from: https://izmyon.hatenablog.com/entry/2023/03/13/222823
    and https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html

    For more info see:
    https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model and
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/

    Attributes:
        - pe: the position embedding array
        - embed_size: the size of the embedding
    """

    pe_matrix: jnp.ndarray
    embed_size: int

    def __init__(self, max_len: int, embed_size: int) -> None:
        """
        Initialize the member variables, create the PE matrix.

        Args:
            - max_len: the maximum input length
            - embed_size: the embeded size, for transformers this is d_q

        Returns:
            - None
        """
        pe_matrix = jnp.zeros((max_len, embed_size))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[:, None]
        div_term = jnp.exp(
            jnp.arange(0, embed_size, 2) * (-jnp.log(10000.0) / embed_size)
        )
        pe_matrix = pe_matrix.at[:, 0::2].set(jnp.sin(position * div_term))
        pe_matrix = pe_matrix.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe_matrix = pe_matrix
        self.embed_size = embed_size

    def __call__(self, x: Int[Array, "seq_len"]) -> Int[Array, "seq_len embed_size"]:
        """
        Encode the positions of a sequence.

        Args:
            - x: the input to encode positionally

        Returns:
            - a matrix with the position embeddings
        """
        x = jnp.arange(self.embed_size).reshape(1, -1).repeat(x.shape[0], axis=0)
        x_input_encoded = x + self.pe_matrix[: x.shape[0]]
        return x_input_encoded


class Encoder(eqx.Module):
    """
    An encoder module, this contains the encoder blocks and position/embedding elements.

    Attributes:
        - encoder_layers: the list of encoder blocks
        - position: the position embedder
        - embedding: the word embedder
        - pad_token: the token that is padded, to be masked out
    """

    encoder_layers: List[EncoderBlock]
    position: PositionEmbedding
    embedding: eqx.nn.Embedding
    pad_token: int

    def __init__(
        self,
        enc_depth: int,
        key: PRNGKeyArray,
        d_q: int = 64,
        d_ff: int = 2048,
        n_heads: int = 8,
        dropout_rate: float = 0.1,
        max_len: int = 5_000,
        input_vocab_size: int = 10_000,
        padding_value: int = -1,
    ) -> None:
        """
        Initialize member variables.

        Args:
            - enc_depth: the number of encoder blocks
            - key: the random key to use for initialization
            - d_q: the size of the query
            - d_ff: the intermediate size of the FFN
            - n_heads: the number of heads for MHA
            - dropout_rate: the rate of dropout
            - max_len: the maximum sequence input length
            - input_vocab_size: the maximum value of an input token
            - padding_value: the padding value to automatically mask out

        Returns:
            - None
        """
        key, subkey = jax.random.split(key, 2)
        enc_keys = jax.random.split(subkey, enc_depth)
        self.encoder_layers = [
            EncoderBlock(
                enc_key,
                d_q=d_q,
                d_v=d_q,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
            )
            for enc_key in enc_keys
        ]

        key, subkey = jax.random.split(key, 2)
        self.embedding = eqx.nn.Embedding(input_vocab_size, d_q, key=subkey)

        self.position = PositionEmbedding(max_len, d_q)
        self.pad_token = padding_value

    def __call__(
        self,
        inputs: Int[Array, "seq_len"],
        mask: Optional[Int[Array, "seq_len"]] = None,
        enable_dropout: Optional[bool] = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq_len d_q"]:
        """
        Forward pass of the encoder.

        If no mask it provided, it automatically pads out accordingly to the provided pad token.

        Args:
            - inputs: the input sequence
            - mask: the values to mask out
            - enable_dropout: whether or not to use dropout
            - key: the random key to use if dropping out
        """
        if mask is None:
            mask = jnp.asarray(inputs != self.pad_token, dtype=jnp.int32)
        pos_encoding = self.position(inputs)
        word_enc = jax.vmap(self.embedding)(inputs.astype("int32"))
        word_pos_enc = word_enc + pos_encoding
        for layer in self.encoder_layers:
            subkey = None
            if key is not None:
                key, subkey = jax.random.split(key, 2)
            word_pos_enc = layer(
                word_pos_enc, mask=mask, enable_dropout=enable_dropout, key=subkey
            )
        return word_pos_enc


class Decoder(eqx.Module):
    """
    A decoder which contains the decoderblocks, position/word embedding.

    Adapted from: https://github.com/bhavnicksm/vanilla-transformer-jax/

    Attributes:
        - decoder_layers: the individual decoderblocks
        - position: the position embedder
        - embedding: the word embedder
        - pad_token: the token that is padded, to be masked out
    """

    decoder_layers: list
    position: PositionEmbedding
    embedding: eqx.nn.Embedding
    pad_token: int

    def __init__(
        self,
        dec_depth: int,
        key: PRNGKeyArray,
        d_q: int = 64,
        d_ff: int = 2048,
        n_heads: int = 8,
        dropout_rate: float = 0.1,
        max_len: int = 5_000,
        input_vocab_size: int = 10_000,
        padding_value: int = -1,
    ) -> None:
        """
        Initialize member variables.

        Args:
            - dec_depth: the number of decoder blocks
            - key: the random key to use for initialization
            - d_q: the size of the query
            - d_ff: the intermediate size of the FFN
            - n_heads: the number of heads for MHA
            - dropout_rate: the rate of dropout
            - max_len: the maximum sequence input length
            - input_vocab_size: the maximum value of an input token
            - padding_value: the padding value to automatically mask out

        Returns:
            - None
        """
        key, subkey = jax.random.split(key, 2)
        dec_keys = jax.random.split(subkey, dec_depth)

        self.decoder_layers = [
            DecoderBlock(
                dec_key,
                d_q=d_q,
                d_ff=d_ff,
                n_heads=n_heads,
                dropout_rate=dropout_rate,
            )
            for dec_key in dec_keys
        ]

        key, subkey = jax.random.split(key, 2)
        self.embedding = eqx.nn.Embedding(input_vocab_size, d_q, key=subkey)

        self.position = PositionEmbedding(max_len, d_q)
        self.pad_token = padding_value

    def __call__(
        self,
        inputs: Int[Array, "seq_len"],
        encoding: Float[Array, "seq_len d_q"],
        mask: Optional[Float[Array, "seq_len"]] = None,
        enable_dropout: Optional[bool] = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq_len d_q"]:
        """
        Forward pass of the decoder.

        If no mask it provided, it automatically pads out accordingly to the provided pad token.

        Args:
            - inputs: the input sequence
            - encoding: the output of the encoder to do cross attention with
            - mask: the values to mask out
            - enable_dropout: whether or not to use dropout
            - key: the random key to use if dropping out

        Returns:
            - an array of the same shape as the input
        """
        if mask is None:
            mask = jnp.asarray(inputs == self.pad_token, dtype=jnp.int32)
        pos_encoding = self.position(inputs)
        word_enc = jax.vmap(self.embedding)(inputs.astype("int32"))

        word_pos_enc = word_enc + pos_encoding
        for layer in self.decoder_layers:
            subkey = None
            if key is not None:
                key, subkey = jax.random.split(key, 2)
            word_pos_enc = layer(
                word_pos_enc, encoding, mask, enable_dropout=enable_dropout, key=subkey
            )
        return word_pos_enc


class Transformer(eqx.Module):
    """
    The full transformer, similar to the original: https://arxiv.org/abs/1706.03762.

    Adapted from: https://github.com/bhavnicksm/vanilla-transformer-jax/

    vanilla-transformer-jax license is available
    https://github.com/bhavnicksm/vanilla-transformer-jax/blob/main/LICENSE

    Attributes:
        - encoder: the encoder model
        - decoder: the decoder model
        - lin_out: the final linear layer on the output of the decoder
    """

    encoder: Encoder
    decoder: Decoder
    lin_out: eqx.nn.Linear

    def __init__(
        self,
        enc_depth: int,
        dec_depth: int,
        key: PRNGKeyArray,
        d_q: int = 64,
        d_ff: int = 2048,
        n_heads: int = 8,
        dropout_rate: float = 0.1,
        max_len: int = 5_000,
        input_vocab_size: int = 1_000,
        target_vocab_size: int = 1_000,
        padding_value: int = -1,
    ) -> None:
        """
        Initialize member variables.

        Args:
            - enc_depth: the number of encoder blocks
            - dec_depth: the number of decoder blocks
            - key: the random key to use for initialization
            - d_q: the size of the query
            - d_ff: the intermediate size of the FFN
            - n_heads: the number of heads for MHA
            - dropout_rate: the rate of dropout
            - max_len: the maximum sequence input length
            - input_vocab_size: the maximum value of an input token
            - target_vocab_size: the maximum value of output tokens (usually equal
                to the input size)
            - padding_value: the padding value to automatically mask out

        Returns:
            - None
        """
        key, subkey1, subkey2 = jax.random.split(key, 3)
        self.encoder = Encoder(
            enc_depth,
            subkey1,
            d_q,
            d_ff,
            n_heads,
            dropout_rate,
            max_len,
            input_vocab_size,
            padding_value,
        )
        self.decoder = Decoder(
            dec_depth,
            subkey2,
            d_q,
            d_ff,
            n_heads,
            dropout_rate,
            max_len,
            input_vocab_size,
            padding_value,
        )
        key, subkey = jax.random.split(key, 2)
        self.lin_out = eqx.nn.Linear(d_q, target_vocab_size, key=subkey)

    def enc_dec_call(
        self,
        src_input: Int[Array, "seq_len"],
        trg_input: Int[Array, "seq_len"],
        mask: Optional[Int[Array, "seq_len"]] = None,
        enable_dropout: Optional[bool] = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, "seq_len output_vocab"]:
        """
        Compute the logits based on the two sequences and the encoder/decoder call.

        Args:
            - src_input: the first sequence, to put into the encoder
            - trg_input: the second sequence, put into the decoder
            - mask: the input values to mask out (automatically masked if not provided)
            - enable_dropout: whether or not to use dropout
            - key: the key to use for the dropout

        Return:
            - the logits for each output vocab possibility for each element in the sequence
        """
        subkey = None
        if key is not None:
            key, subkey = jax.random.split(key, 2)
        enc_out = self.encoder(src_input, mask, enable_dropout, key)
        dec_out = self.decoder(trg_input, enc_out, mask, enable_dropout, subkey)
        logits = jax.vmap(self.lin_out)(dec_out)
        return logits


class DiscreteEncoderEnergy(eqx.Module):
    """
    A Encoder based energy function.

    Stacks encoder layers followed by linear layers converting
    the [seq_len, output] shape to [] shape.

    TO DO: https://arxiv.org/pdf/1903.08689.pdf suggests norm's are bad?

    Attributes: change
        - encoder_layers: the encoder layers
        - lin_out: the final output linear layer
        - out1: the first linear output layer, vmapped to get [seq_len, output] -> [seq_len]
        - out2: the second linear output layer, converts [seq_len] -> []
        - pe: the positional encoding array, e.g. using sine position embedding, for more
            information see `PositionwiseFeedForwardBlock` docs
        - d_k: the dq, dk, dv value, i.e. the dimension of the key, query, and value (which
            are all assumed to be the same)

    """

    encoder: Encoder
    lin_out: eqx.nn.Linear
    out1: eqx.nn.Linear
    out2: eqx.nn.Linear

    def __init__(
        self,
        input_length: int,
        depth: int,
        key: PRNGKeyArray,
        n_heads: int = 8,
        d_q: int = 64,
        d_ff: int = 1028,
        dropout_rate: float = 0.1,
        output_dim: int = 512,
        vocab_size: int = 10_000,
        padding_value: int = -1,
    ) -> None:
        """
        Initialize the member variables.

        Args:
            - input_length: the size of the input
            - depth: number of encoder blocks
            - key: the random key to use
            - n_heads: the number of heads to use for MHA
            - d_q: the dimensions of the key, query, value to use
            - d_ff: the dimensions of the hidden layer of the FFN
            - dropout_rate: the percent dropout to do (if being used)
            - output_dim: the output dimension of the encoder
            - vocab_size: the maximum possible input value integer
            - padding_value: the padding token

        Returns:
            - None
        """
        key, subkey = jax.random.split(key, 2)
        self.encoder = Encoder(
            depth,
            subkey,
            d_q,
            d_ff,
            n_heads,
            dropout_rate,
            input_length,
            vocab_size,
            padding_value,
        )

        linear_1, linear_2, linear_3 = jax.random.split(key, 3)

        self.lin_out = eqx.nn.Linear(d_q, output_dim, key=linear_1)
        self.out1 = eqx.nn.Linear(output_dim, 1, key=linear_2)
        self.out2 = eqx.nn.Linear(input_length, 1, key=linear_3)

    def __call__(
        self,
        x: Float[Array, "seq_len"],
        mask: Optional[Int[Array, "seq_len"]] = None,
        enable_dropout: Optional[bool] = False,
        key: Optional[PRNGKeyArray] = None,
    ) -> Float[Array, ""]:
        """
        Forward pass of the EBM neural network.

        Inputs of ints, e.g. [1, 4, 100, 32]

        Args:
            - x: the input to the nn
            - mask: masking the inputs (e.g. [1, 1, 0, 0]) masks the latter two
            - enable_dropout: whether to use dropout or not
            - key: the key to use for dropout

        Returns:
            - the energy of the state
        """
        x = self.encoder(x, mask, enable_dropout, key)
        x = jax.vmap(self.lin_out)(x)
        x = jax.nn.gelu(x)
        x = jax.vmap(self.out1)(x)  # TODO investigate flatten vs. vmap here
        x = jax.nn.gelu(x)
        x = self.out2(x)
        return jnp.squeeze(x)
