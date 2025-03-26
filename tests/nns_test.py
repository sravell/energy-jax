"""Unit testing."""

from absl.testing import absltest
from jax import numpy as jnp
import jax
import equinox as eqx
from energax.ebms import nn_ebms
from energax import nns


class EBMTest(absltest.TestCase):
    def test_shapes_mlp(self):
        """
        Test shaping, for NNs there is limited unit testing, but this is like
        https://github.com/patrick-kidger/equinox/blob/main/tests/test_nn.py#L171
        """
        mlp = nn_ebms.DiscreteNNEBM(nns.MLP(10, 10, 10, jax.random.PRNGKey(0)))
        self.assertEqual(mlp.param_count(), 1001)
        x = jnp.array([i for i in range(10)])
        out = eqx.filter_jit(mlp.energy_function)(x)
        self.assertTrue(out.shape == ())

        mlp = nn_ebms.ContinuousNNEBM(nns.MLP(10, 10, 10, jax.random.PRNGKey(0)))
        self.assertEqual(mlp.param_count(), 1001)
        x = jnp.array([i for i in range(10)])
        out = eqx.filter_jit(mlp.energy_function)(x)
        self.assertTrue(out.shape == ())

    def test_shapes_transformer(self):
        enc = nn_ebms.DiscreteNNEBM(
            nns.DiscreteEncoderEnergy(
                input_length=10,
                depth=2,
                key=jax.random.PRNGKey(0),
                n_heads=3,
                d_ff=48,
                d_q=4,
                output_dim=32,
                vocab_size=20,
            ),
        )
        self.assertEqual(enc.param_count(), 1350)
        x = jnp.array([i for i in range(10)])
        out = eqx.filter_jit(enc.energy_function)(x)
        self.assertTrue(out.shape == ())

    def test_kwargs_transformer(self):
        enc = nn_ebms.DiscreteNNEBM(
            nns.DiscreteEncoderEnergy(
                input_length=10,
                depth=2,
                key=jax.random.PRNGKey(0),
                n_heads=3,
                d_ff=48,
                d_q=4,
                output_dim=32,
                vocab_size=20,
            ),
        )
        x = jnp.array([i for i in range(10)])
        out = eqx.filter_jit(enc.energy_function)(x, key=jax.random.PRNGKey(0))
        self.assertTrue(out.shape == ())

    def test_gnn_energy(self):
        enc = nn_ebms.DiscreteNNEBM(
            nns.GNNEnergy(
                10,
                nns.GATNetwork(
                    nfeat=3,
                    nhid=8,
                    nclass=1,
                    depth=1,
                    nheads=4,
                    key=jax.random.PRNGKey(0),
                ),
                key=jax.random.PRNGKey(0),
            ),
        )
        x = jnp.ones(shape=(10, 3))
        adj = jnp.ones(shape=(10, 10))
        out = eqx.filter_jit(enc.energy_function)(
            x, adj=adj, enable_dropout=True, key=jax.random.PRNGKey(0)
        )
        self.assertTrue(out.shape == ())


class NNTests(absltest.TestCase):
    def test_transformer(self):
        target_size = 12 + 1
        transformer = nns.Transformer(
            enc_depth=2,
            dec_depth=2,
            n_heads=2,
            d_q=16,
            d_ff=32,
            max_len=10,
            input_vocab_size=12 + 1,
            target_vocab_size=target_size,
            key=jax.random.PRNGKey(0),
        )
        output = transformer.enc_dec_call(
            jnp.array([i for i in range(10)]),
            jnp.array([i for i in range(10)]),
            enable_dropout=True,
            key=jax.random.PRNGKey(0),
        )
        self.assertTrue(output.shape == (10, target_size))

    def test_resnet(self):
        model = nns.resnet18(norm_layer=eqx.nn.Identity, key=jax.random.PRNGKey(0))
        imgs = jnp.ones(shape=(10, 3, 32, 32))
        results = eqx.filter_vmap(model)(imgs)
        self.assertTrue(results.shape == (10, 1))

    def test_unet(self):
        n_out = 3
        n_in = 3
        model = nns.UNet(n_in, n_out, jax.random.PRNGKey(0))
        imgs = jnp.ones(shape=(10, n_in, 32, 32))
        results = eqx.filter_vmap(model)(imgs)
        self.assertTrue(results.shape == (10, n_out, 32, 32))

    def test_ncsn(self):
        config = {
            "rescaled": False,
            "act": jax.nn.swish,
            "ngf": 32,
            "channels": 3,
            "image_size": 32,
        }
        model = nns.NCSNv2(config, jax.random.PRNGKey(0))
        imgs = jnp.ones(shape=(10, config["channels"], 32, 32))
        results = eqx.filter_vmap(model)(imgs)
        self.assertTrue(results.shape == (10, config["channels"], 32, 32))


if __name__ == "__main__":
    absltest.main()
