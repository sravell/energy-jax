import unittest
import jax
from jax import numpy as jnp
from jax import flatten_util
import equinox as eqx
from equinox.nn import MLP
from energax.optimizers import (
    calculate_natural_gradient,
    fisher_metric,
    ngd_constructor_full_metric,
    ngd_constructor,
    sgld,
)
from sklearn.datasets import make_classification


def loss(model, X, sample=None, regularization=0.0):
    e_plus = eqx.filter_vmap(model, in_axes=(0))(X)
    return e_plus.sum()


# adapted from https://docs.kidger.site/equinox/examples/mnist/
def loss_fn(model, x, y):
    pred_y = eqx.filter_vmap(model)(x)
    pred_y = jax.nn.log_softmax(pred_y)
    return cross_entropy(y, pred_y)


def cross_entropy(y, pred_y):
    pred_y = jnp.take_along_axis(pred_y, y, axis=1)
    return -jnp.mean(pred_y)


@eqx.filter_jit
def compute_accuracy(model, x, y):
    """This function takes as input the current model
    and computes the average accuracy on a batch.
    """
    pred_y = jax.vmap(model)(x)
    pred_y = jnp.argmax(pred_y, axis=1)
    return jnp.mean(y == pred_y)


X2, Y2 = make_classification(
    n_features=2,
    n_redundant=0,
    n_informative=2,
    n_samples=1000,
    n_clusters_per_class=1,
    n_classes=2,
    shuffle=False,
    random_state=42,
)


class OptimizerTests(unittest.TestCase):
    def setUp(self) -> None:
        class Test(eqx.Module):
            weights: jnp.ndarray

            def __init__(self):
                self.weights = jnp.ones((2, 2))

            def __call__(self, x):
                return self.energy_function(x)

            def energy_function(self, x):
                """Some simple dummy energy function, that gives
                non-trivial gradients for testing purposes."""
                return jnp.exp(-self.weights @ x).sum()

        self.test = Test()
        self.X = jnp.array([[1.0, 2], [4.0, 5], [0.1, -0.3]])
        self.sampled = jnp.array([[1.0, 2], [4.0, 5], [0.1, -0.3]])
        self.value_grad_fct = eqx.filter_value_and_grad(loss)
        self.grad = jnp.array([-0.17242098, 0.26622966, -0.17242098, 0.26622966])
        self.ngrad = jnp.array([-23.568851, -3.6848521, -23.568851, -3.6848521])
        self.wtehgrad = jnp.array([0.11637416, -0.01832488, -0.05656732, 0.07547278])
        """These gradients are claculated via the following code:
        
        weights = jnp.ones((2, 2))
        X = jnp.array([[1.0, 2], [4.0, 5], [0.1, -0.3]])

        def energy_function(weights, x):
            return jnp.exp(-weights @ x).sum()

        def loss(weights, x):
            return jax.vmap(energy_function, in_axes=(None, 0))(weights, x).sum()

        grad = jax.grad(loss)(weights, X)

        fisher_information = jnp.outer(grad, grad)

        ngrad = jnp.linalg.pinv(fisher_information) @ grad
        """

    def test_dummy_model(self):
        out = self.test(jnp.array([1.0, 1.0]))
        self.assertEqual(out, 2 * jnp.exp(-2))

    def test_ngd(self):
        value, grads = self.value_grad_fct(self.test, self.X)
        g1, unflat = flatten_util.ravel_pytree(grads)
        ngd_grads = calculate_natural_gradient(self.test, grads, self.X, self.sampled)
        g2, unflat = flatten_util.ravel_pytree(ngd_grads)
        self.assertTrue(jnp.allclose(g1, self.grad))
        self.assertTrue(jnp.allclose(g2, self.ngrad))

    def test_ngd_update(self):
        # optimizer = optax.sgd(1.0)
        # opt_state = optimizer.init(eqx.filter(self.test, eqx.is_array))
        ngd_update = ngd_constructor(loss)
        grads, value = ngd_update(self.test, self.X, self.sampled)
        grads, unflat = flatten_util.ravel_pytree(grads)
        self.assertTrue(jnp.allclose(self.ngrad, grads, atol=1e-4))

    def test_fisher_metric(self):
        metric = fisher_metric()
        value, grads = self.value_grad_fct(self.test, self.X)
        g1, unflat = flatten_util.ravel_pytree(grads)
        M = metric(self.test, self.X, self.sampled).sum(axis=0)
        g2 = jnp.linalg.pinv(M) @ g1
        self.assertTrue(jnp.allclose(g2, self.ngrad, atol=1e-4))

    def test_ngd_full_metric(self):
        # optimizer = optax.sgd(1.0)
        # opt_state = optimizer.init(eqx.filter(self.test, eqx.is_array))
        ngd_update = ngd_constructor_full_metric(loss)
        value, grads = self.value_grad_fct(self.test, self.X)
        grads, value_ngd = ngd_update(self.test, self.X, self.sampled)
        self.assertTrue(jnp.allclose(value, value_ngd))
        grads, unflat = flatten_util.ravel_pytree(grads)
        self.assertTrue(jnp.allclose(self.ngrad, grads, atol=1e-4))

    def test_sgld(self):
        optimizer = sgld(scale_factor=-self.X.shape[0])
        opt_state = optimizer.init(eqx.filter(self.test, eqx.is_array))

        value, grads = self.value_grad_fct(self.test, self.X)
        grads, opt_state = optimizer.update(grads, opt_state, self.test)
        grads, unflat = flatten_util.ravel_pytree(grads)
        self.assertTrue(jnp.allclose(self.wtehgrad, grads, atol=1e-4))

    def test_sgld_with_metric(self):
        n_feats = 2
        key = jax.random.PRNGKey(42)

        nn_1 = MLP(
            key=key,
            in_size=n_feats,
            width_size=16,
            depth=1,
            out_size=2,
            final_activation=jax.nn.relu,
        )

        loss_call = eqx.filter_jit(loss_fn)
        optimizer = sgld(learning_rate=0.01, scale_factor=-len(X2))
        opt_state_nn = optimizer.init(eqx.filter(nn_1, eqx.is_array))

        @eqx.filter_jit
        def make_step(model, opt_state, x, y):
            loss_value, grads = eqx.filter_value_and_grad(loss_call)(model, x, y)

            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        data = jnp.array(X2)
        y = jnp.expand_dims(Y2, axis=1).astype(int)
        for _ in range(10):
            (
                nn_1,
                opt_state_nn,
                _,
            ) = make_step(
                nn_1,
                opt_state_nn,
                data,
                y,
            )

        final_acc = compute_accuracy(nn_1, data, jnp.squeeze(y))
        self.assertTrue(final_acc > 0.9)

    def test_sgld_with_metric_and_preconditioning(self):
        n_feats = 2
        key = jax.random.PRNGKey(42)

        nn_2 = MLP(
            key=key,
            in_size=n_feats,
            width_size=16,
            depth=1,
            out_size=2,
            final_activation=jax.nn.relu,
        )

        loss_call = eqx.filter_jit(loss_fn)
        optimizer = sgld(
            learning_rate=0.01,
            scale_factor=-len(X2) / 2,
            use_preconditioning=True,
            momentum=0.9,
        )
        opt_state_nn = optimizer.init(eqx.filter(nn_2, eqx.is_array))

        @eqx.filter_jit
        def make_step(model, opt_state, x, y):
            loss_value, grads = eqx.filter_value_and_grad(loss_call)(model, x, y)

            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss_value

        data = jnp.array(X2)
        y = jnp.expand_dims(Y2, axis=1).astype(int)
        for _ in range(10):
            (
                nn_2,
                opt_state_nn,
                _,
            ) = make_step(
                nn_2,
                opt_state_nn,
                data,
                y,
            )

        final_acc = compute_accuracy(nn_2, data, jnp.squeeze(y))
        self.assertTrue(final_acc > 0.9)
