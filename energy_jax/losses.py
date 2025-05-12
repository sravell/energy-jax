"""Loss functions for training EBMs."""

from typing import Optional
from jaxtyping import Num, Array, Float, PRNGKeyArray
import jax
from jax import numpy as jnp
import equinox as eqx  # type: ignore[import]
from .ebms.ebm import AbstractModel


def ssm(
    model: AbstractModel,
    data: Num[Array, "..."],
    key: PRNGKeyArray,
    n_particles: int = 1,
) -> Float[Array, ""]:
    r"""
    Sliced Score Matching (SSM) loss function.

    This implements a way of computing the score matching objective by using
    random vector multiplication.

    This computes an objective:
    $$J(\theta, x) = \sum_{v, x} v^T \nabla_x s_\theta(x) v + \frac{1}{2} \left (v^T s_\theta(x) \right )^2$$
    where v is a random vector and s is the score function. This is given by eq (7) in the paper.

    This can be computed relatively efficiently using autodiff to do vector jacobian products
    without having to take the full jacobian.

    Reference: https://arxiv.org/abs/1905.07088

    Code adapted from:
    https://github.com/ermongroup/sliced_score_matching/blob/master/losses/sliced_sm.py

    Args:
        - model: the ebm or score network to compute the loss for
        - data: the batch of data from p(x)
        - key: the random key to use
        - n_particles: the number of random vectors to use (M = 1 was suggested by the paper
            but more could potentially improve performance at a cost to longer compute times)

    Returns:
        - the SSM loss value
    """
    dup_samples = jnp.broadcast_to(data, (n_particles,) + data.shape)
    dup_samples = jnp.reshape(dup_samples, (-1,) + data.shape[1:])
    vectors = jax.random.normal(key=key, shape=dup_samples.shape)
    vectors = vectors / jnp.linalg.norm(vectors, axis=-1, keepdims=True)

    score_fn = model.score
    grad1_fn = eqx.filter_vmap(score_fn)
    grad1 = grad1_fn(dup_samples)
    loss1 = jnp.sum(grad1 * vectors, axis=-1) ** 2 * 0.5

    # gradv = lambda z: jnp.sum(grad1_fn(z) * vectors)
    def gradv(z_input: Float[Array, ""]) -> Float[Array, ""]:
        return jnp.sum(grad1_fn(z_input) * vectors)

    grad2 = eqx.filter_grad(gradv)(dup_samples)
    loss2 = jnp.sum(vectors * grad2, axis=-1)

    loss1 = jnp.reshape(loss1, (n_particles, -1)).mean(axis=0)
    loss2 = jnp.reshape(loss2, (n_particles, -1)).mean(axis=0)
    loss = loss1 + loss2
    return loss.mean()


def ssm_vr(
    model: AbstractModel,
    data: Num[Array, "..."],
    key: PRNGKeyArray,
    n_particles: int = 1,
) -> Float[Array, ""]:
    r"""
    Sliced Score Matching (SSM) loss function with variance reduction.

    Given the the random vectors are guassian this allows us to analytically integrate
    the objective to reduce variance.

    This computes an objective:
    $$ J(\theta, x) = v^T \nabla_x s_\theta (x) v + \frac{1}{2} || s_\theta (x) ||^2 $$
    where v is a random vector and s is the score function. This is given by eq (8) in the paper.

    This can be computed relatively efficiently using autodiff to do vector jacobian
    products without having to take the full jacobian.

    Reference: https://arxiv.org/abs/1905.07088

    Args:
        - model: the ebm or score network to compute the loss for
        - data: the batch of data from p(x)
        - key: the random key to use
        - n_particles: the number of random vectors to use (M = 1 was suggested by the paper
            but more could potentially improve performance at a cost to longer compute times)

    Returns:
        - the SSM loss value
    """
    dup_samples = jnp.broadcast_to(data, (n_particles,) + data.shape)
    dup_samples = jnp.reshape(dup_samples, (-1,) + data.shape[1:])
    vectors = jax.random.normal(key=key, shape=dup_samples.shape)

    score_fn = model.score
    grad1_fn = eqx.filter_vmap(score_fn)
    grad1 = grad1_fn(dup_samples)
    loss1 = jnp.sum(grad1 * grad1, axis=-1) / 2.0

    def gradv(z_input: Float[Array, ""]) -> Float[Array, ""]:
        return jnp.sum(grad1_fn(z_input) * vectors)

    # gradv = lambda z: jnp.sum(grad1_fn(z) * vectors)
    grad2 = eqx.filter_grad(gradv)(dup_samples)
    loss2 = jnp.sum(vectors * grad2, axis=-1)

    loss1 = jnp.reshape(loss1, (n_particles, -1)).mean(axis=0)
    loss2 = jnp.reshape(loss2, (n_particles, -1)).mean(axis=0)
    loss = loss1 + loss2
    return loss.mean()


def dsm(
    model: AbstractModel, data: Num[Array, "..."], key: PRNGKeyArray, sigma: float = 0.2
) -> Float[Array, ""]:
    r"""
    Single noise level denoising score matching.

    Approximates the score matching objective by replacing $p(x)$ with a noisy
    distribution $q(\tilde{x} | x)$ that has a known score. We then estimate
    the score of $q(\tilde{x}) = \int q(\tilde{x} | x) p(x) dx$ instead of
    $p(x)$.

    This is done via the objective
    $$ J(\theta, x) = \frac{1}{2} \mathbbm{E}_{q(\tilde{x})} [ ||s_\theta (\tilde{x}) -
    \nabla_x log q (\tilde{x} | x)||^2_2 ] $$

    In our case, we have $q(\tilde{x} | x)$ being a gaussian perturbation, so we
    have analytic knowledge of the score, yielding:
    $$ J(\theta, x) = \frac{1}{2} \mathbbm{E}_{p(x)} \mathbbm{E}_{x \sim \mathcal{N}(x, \sigma^2 I)}
    [ ||s_\theta (\tilde{x}) + (\tilde{x} - x) / \sigma^2 ||^2_2 ] $$


    Reference: https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf

    Code adapted from: https://github.com/ermongroup/ncsn/blob/master/losses/dsm.py

    Args:
        - model: the energy based model to optimize over
        - data: the data from p(x)
        - key: the random key to use
        - sigma: the noise level of the q distribution

    Returns:
        - the approximate objective based on DSM
    """
    perturbed_samples = data + jax.random.normal(key, shape=data.shape) * sigma
    target = -1 / (sigma**2) * (perturbed_samples - data)
    scores = eqx.filter_vmap(model.score)(perturbed_samples)
    target = jnp.reshape(target, (target.shape[0], -1))
    scores = jnp.reshape(scores, (scores.shape[0], -1))
    loss = 1 / 2.0 * ((scores - target) ** 2).sum(axis=-1).mean(axis=0)
    return loss


def anneal_dsm(
    model: AbstractModel,
    data: Float[Array, "batch xshape"],
    sigmas: Float[Array, "num_sigmas"],
    key: PRNGKeyArray,
    anneal_power: float = 2.0,
) -> Float[Array, ""]:
    """
    Multi noise level denoising score matching.

    Learn multiple noise levels of perturbations simultaneously. Basically, we compute the
    denoising score matching loss for randomly chosen sigmas. I.e. we have multiple levels of
    noise that we learn simultaneously. See equation (5) and (6) for what we are implementing.

    Reference: https://arxiv.org/abs/1907.05600

    Code adapted from: https://github.com/ermongroup/ncsn/blob/master/losses/dsm.py

    Args:
        - model: the energy based model to optimize over
        - data: the data from p(x)
        - sigmas: the noise levels to choose from
        - key: the random key to use
        - anneal_power: the power of the sigmas

    Returns:
        - the approximate objective based on DSM
    """
    sigma_labels = jax.random.randint(key, (data.shape[0],), 0, len(sigmas))
    used_sigmas = sigmas[sigma_labels].reshape(
        data.shape[0], *([1] * len(data.shape[1:]))
    )
    noise = jax.random.normal(key, data.shape) * used_sigmas
    perturbed_samples = data + noise
    target = -1 / (used_sigmas**2) * noise
    scores = eqx.filter_vmap(model.score)(perturbed_samples)
    target = target.reshape(target.shape[0], -1)
    scores = scores.reshape(scores.shape[0], -1)
    loss = ((scores - target) ** 2).sum(axis=-1) * used_sigmas.squeeze() ** anneal_power
    loss = 1 / 2 * loss
    return loss.mean(axis=0)


def contrastive_divergence(
    model: AbstractModel,
    data: Num[Array, "real_samples x_shape"],
    model_samples: Num[Array, "fake_samples x_shape"],
    regularization: Optional[float] = 0.0,
) -> Float[Array, ""]:
    r"""
    Compute (regularized) contrastive divergence loss.

    Computes $ \mathbbm{E}_{x \sim p(x)} [E(x)] - \mathbbm{E}_{x \sim p_\theta(x)} [E(x)] +
    \alpha \mathbbm{E}_{x \sim p(x), y \sim p_\theta(x)} [E(x)^2 + E(y)^2] $ or CD term +
    alpha * Reg term.

    Args:
        - model: the EBM to use as the energy function
        - data: the samples from the real p(x)
        - model_samples: samples generated from the EBM distribution
        - regularization: the amount of regularization to use to prevent energies from exploding

    Returns:
        - the scalar value of the loss
    """
    e_plus = eqx.filter_vmap(model.energy_function)(data)
    e_minus = eqx.filter_vmap(model.energy_function)(model_samples)

    loss_cd = jnp.mean(e_plus) - jnp.mean(e_minus)
    loss_reg = jnp.mean(e_plus**2) + jnp.mean(e_minus**2)
    return jnp.squeeze(loss_cd + regularization * loss_reg)


def softplus_contrastive_divergence(
    model: AbstractModel,
    data: Num[Array, "real_samples x_shape"],
    model_samples: Num[Array, "fake_samples x_shape"],
    regularization: Optional[float] = 0.0,
    temperature: Optional[float] = 1.0,
) -> Float[Array, ""]:
    r"""
    Compute (regularized) softplus contrastive divergence loss.

    Computes softplus(CD term) + reg term. Adapted from:
    https://github.com/openai/ebm_code_release/blob/master/train.py#L841.

    Can help with stability of training over CD.

    Args:
        - model: the EBM to use as the energy function
        - data: the samples from the real p(x)
        - model_samples: samples generated from the EBM distribution
        - regularization: the amount of regularization to use to prevent energies from exploding
        - temperature: the temperature of the energy function

    Returns:
        - the scalar value of the loss
    """
    e_plus = eqx.filter_vmap(model.energy_function)(data)
    e_minus = eqx.filter_vmap(model.energy_function)(model_samples)

    diff_cd = temperature * (e_plus - e_minus)
    loss_reg = jnp.mean(e_plus**2 + e_minus**2)
    return jnp.squeeze(jnp.mean(jax.nn.softplus(diff_cd)) + regularization * loss_reg)


def logsumexp_contrastive_divergence(
    model: AbstractModel,
    data: Num[Array, "real_samples x_shape"],
    model_samples: Num[Array, "fake_samples x_shape"],
    regularization: Optional[float] = 0.0,
) -> Float[Array, ""]:
    r"""
    Compute (regularized) softplus contrastive divergence loss.

    Computes E[positive] + logsumexp(E[negative]) + reg term. Adapted from:
    https://github.com/openai/ebm_code_release/blob/master/train.py#L829.

    Can help with stability of training over CD.

    Args:
        - model: the EBM to use as the energy function
        - data: the samples from the real p(x)
        - model_samples: samples generated from the EBM distribution
        - regularization: the amount of regularization to use to prevent energies from exploding

    Returns:
        - the scalar value of the loss
    """
    e_plus = eqx.filter_vmap(model.energy_function)(data)
    e_minus = eqx.filter_vmap(model.energy_function)(model_samples)

    e_minus_reduced = e_minus - jnp.min(e_minus)
    coeff = jax.lax.stop_gradient(jnp.exp(-e_minus_reduced))
    norm_constant = jax.lax.stop_gradient(jnp.sum(coeff)) + 1e-4
    pos_loss = jnp.mean(e_plus)
    neg_loss = coeff * (-1 * e_minus) / norm_constant

    loss_cd = pos_loss + jnp.sum(neg_loss)
    loss_reg = jnp.mean(e_plus**2 + e_minus**2)
    return jnp.squeeze(loss_cd + regularization * loss_reg)
