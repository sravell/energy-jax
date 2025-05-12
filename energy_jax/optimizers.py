"""Optimizer functions."""

from typing import Optional, Callable, NamedTuple
import jax
from jax import numpy as jnp
from jax import flatten_util
from jaxtyping import Float, Array, Num, PyTree
import optax  # type: ignore[import]
import chex
import equinox as eqx
from .ebms.ebm import AbstractEBM


def log_p_x(
    model: AbstractEBM,
    x: Num[Array, "samples_x x_shape"],
    sampled_data: Num[Array, "samples_Z x_shape"],
) -> Float[Array, "samples_x x_shape"]:
    r"""
    Calculate the log probability of the data.

    This function is used to obtain the gradient
    :math:`\nabla \log p(x) = \nabla () - E(x) + \mathbb{E}_{x \sim p_\theta(x)} [E(x)] )`
    without taking the average over data points.

    This is required to calculate the Fisher information matrix.
    :math:`\mathbb{E}_{x \sim p_\theta(x)} \nabla \log p_\theta(x) \nabla \log p_\theta(x)^T`

    The difference in the signs compared to the CD loss function is due to the fact
    that for the CD loss we calculate :math:`KL(p_{data} || p_\theta)`
    which results in a minus sign because of the division :math:`\log \frac{p_{data}}{p_{\theta}}`.

    For more information see:
    - https://agustinus.kristia.de/techblog/2018/03/11/fisher-information/
    - https://gebob19.github.io/natural-gradient/

    Args:
        - model: the EBM to use as the energy function
        - x: Samples or data to calculate p(x) for
        - sampled_data: samples generated from the EBM distribution

    Returns:
        - -E(x_input) + <E(sampled_data)>
    """
    e_plus = model.energy_function(x)
    e_minus = eqx.filter_vmap(model.energy_function)(sampled_data)

    loss_cd = -e_plus + jnp.mean(e_minus)
    return loss_cd


def fisher_vp(
    function: Callable, model: AbstractEBM, vector: Num[Array, "params"]
) -> Float[Array, "params"]:
    r"""
    Calculate the Fisher vector product.

    Adapted from naturalgradient,
    license avaible at https://github.com/gebob19/naturalgradient/blob/main/LICENSE

    :math:`F v = J J^T v` where :math:`J` is the jacobian of the function
    with respect to its params.

    Strictly speaking, this function calculates (J v)^T J = v^T (J^T J).
    Which is equal to (F*v)^T.
    Since (F*v)^T is a vector and in jax M@x = M@x.T the extra transpose doesn't have an effect.

    Args:
        - function: the function to calculate the Fisher vector product for
        - model: the EBM to use as the energy function
        - vector: the vector to multiply the Fisher matrix with

    Returns:
        - the Fisher vector product F*v
    """
    # jvp calculates J@v, with J the jacobian of the function
    # i.e. df/dx @ v. In this specific case f(x) is log_p_x_batch(model)
    # for fixed x_input and sampled_data
    _, jac_vector_prod = eqx.filter_jvp(function, (model,), (vector,))
    # vjp calculates v.T @ df/dx
    _, f_vjp = eqx.filter_vjp(function, model)
    return f_vjp(jac_vector_prod)[0]


def calculate_natural_gradient(
    model: AbstractEBM,
    grads: AbstractEBM,
    x: Num[Array, "samples x_shape"],
    sampled_data: Num[Array, "samples_Z x_shape"],
    maxiter: Optional[int] = 10,
) -> AbstractEBM:
    r"""
    Calculate the natural gradient.

    :math:`F^{-1} grad` with the Fisher information matrix :math:`F`.

    Args:
        - model: the EBM to use as the energy function
        - grads: the gradients of the loss function wrt the model parameters
        - x: Samples or data to calculate p(x) for
        - sampled_data: samples generated from the EBM distribution to approximate Z
        - maxiter: the maximum number of iterations for the conjugate gradient solver

    Returns:
        - the natural gradient
    """
    log_p_x_batch = eqx.filter_vmap(log_p_x, in_axes=(None, 0, None))

    def fct(ebm_model: AbstractEBM) -> Float[Array, ""]:
        return log_p_x_batch(ebm_model, x, sampled_data)

    def fvp(vec: Num[Array, "params"]) -> Float[Array, "params"]:
        return fisher_vp(fct, model, vec)

    # vec is a placeholder vector, we don't want F*v but F-1*grads
    # For this we need to solve F*vec = grads for vec
    ngrad, _ = jax.scipy.sparse.linalg.cg(
        fvp, grads, maxiter=maxiter
    )  # approx solve with Conjugate Gradient
    return ngrad


def ngd_constructor(
    loss: Callable,
    maxiter: Optional[int] = 10,
) -> Callable:
    r"""
    Natural gradient descent constructor function.

    This is the natural gradient descent algorithm without
    calculating the Fisher information matrix explicitly. But instead
    using the Fisher vector product.

    The Bogoliubov-Kubo-Mori metric as described in
    https://arxiv.org/pdf/2206.04663.pdf (G22) is equivalent to the Fisher information matrix
    in classical systems.
    For quantum systems :math:`I^{BKM} = -tr(\partial_k \rho \partial_l \log \rho)`.
    For classical systems this translates to :math:`I^{BKM} = \int \partial_k p(x) \partial_l \log p(x) d_x`.
    Which is equivalent to :math:`sum_{x \sim p(x)} \partial_k \log p(x) \partial_l \log p(x)`
    using the relation :math:`\frac{\partial_k p(x)}{p(x)} = \partial_k \log p(x)`.

    Args:
        - loss: the loss function to use for the update step
        - optimizer: the optimizer to use for the update step
        - maxiter: the maximum number of iterations for the conjugate gradient solver

    Returns:
        - A function to calculate the natural gradient
    """
    value_grad_fct = eqx.filter_value_and_grad(loss)

    @eqx.filter_jit
    def ngd_update(
        model: AbstractEBM,
        x: Num[Array, "samples x_shape"],
        sampled_data_z: Num[Array, "samples_Z x_shape"],
    ) -> tuple[AbstractEBM, Float[Array, ""]]:
        """
        Do one ngd update step.

        Args:
            - model: the EBM to use as the energy function
            - x_input: data batch to calculate loss
            - sampled_data_Z: samples generated from the EBM distribution to approximate Z

        Returns:
            - the updates and the loss value
        """
        value, grads = value_grad_fct(model, x, sampled_data_z)

        updates = calculate_natural_gradient(
            model, grads, x, sampled_data_z, maxiter=maxiter
        )
        return updates, value

    return ngd_update


def fisher_metric() -> Callable:
    r"""
    Calculate the Fisher information matrix explicitly.

    This is the matrix :math:`JJ^T` where :math:`J` is the jacobian of the function
    with respect to its params.

    Calcualting the Fisher information matrix explicitly is not feasible for large models.
    And should only be used if one wants access to the full matrix.

    Returns:
        - A function to calculate the Fisher information matrix
    """
    grad_log_p_x = eqx.filter_grad(log_p_x)

    def metric(
        model: AbstractEBM,
        x: Num[Array, "samples x_shape"],
        sampled_data: Num[Array, "samples_Z x_shape"],
    ) -> Num[Array, "params_shape params_shape"]:
        r"""
        Calculate the Fisher information matrix.

        Args:
            - model: the EBM to use as the energy function
            - x_input: Samples or data to calculate p(x) for
            - sampled_data: samples generated from the EBM distribution to approximate Z

        Returns:
            - the Fisher information matrix for a single input x
                :math:`\nabla \log p(x) \nabla \log p(x)^T`
        """
        grads = grad_log_p_x(model, x, sampled_data)
        grad_flat, _ = flatten_util.ravel_pytree(grads)
        return jnp.outer(grad_flat, grad_flat)

    return eqx.filter_vmap(metric, in_axes=(None, 0, None))


def ngd_constructor_full_metric(
    loss: Callable,
    inverse: Optional[str] = "penrose",
) -> Callable:
    """
    NGD step with the full Fisher information matrix.

    Args:
        - loss: the loss function to use for the update step
        - inverse: the inverse to use for the Fisher information matrix
        - regularization: the regularization strength to use for the update step

    Returns:
        - A function to calculate the natural gradient
    """
    value_grad_fct = eqx.filter_value_and_grad(loss)

    @eqx.filter_jit
    def ngd_update(
        model: AbstractEBM,
        x: Num[Array, "samples x_shape"],
        sampled_data_z: Num[Array, "samples_Z x_shape"],
    ) -> tuple[AbstractEBM, Float[Array, ""]]:
        """
        Do one ngd update step with the full Fisher metric tensor.

        Args:
            - model: the EBM to use as the energy function
            - x_input: Samples or data to calculate p(x) for
            - sampled_data_Z: samples generated from the EBM distribution to approximate Z

        Returns:
            - the updated model and optimizer state
        """
        value, grads = value_grad_fct(model, x, sampled_data_z)
        g_flat, unflat = flatten_util.ravel_pytree(grads)
        metric_fct = fisher_metric()
        metric = metric_fct(model, x, sampled_data_z).sum(axis=0)

        if inverse == "penrose":
            update = jnp.linalg.pinv(metric) @ g_flat
        elif inverse == "offset":
            epsilon = 1e-5
            # offset = jnp.eye(metric.shape[0]) * epsilon
            offset = epsilon
            update = jnp.linalg.solve(metric + offset, g_flat)
        else:
            raise ValueError("Invalid inverse option")

        updates = unflat(update)
        return updates, value

    return ngd_update


def safe_int32_increment(
    count: chex.Numeric, add_counter: chex.Numeric
) -> chex.Numeric:
    """
    Increments int32 counter by b.

    Modified from the safe_int32_increment of
    https://github.com/google-deepmind/optax/blob/master/optax/_src/numerics.py
    but instead of adding only the value 1, we allow to add any number add_counter.

    Normally `max_int + add_counter` would overflow to `min_int`.
    This functions ensures that when `max_int` is reached the counter
    stays at `max_int`.

    Args:
      - count: a counter to be incremented.
      - add_counter: the value to be added to the counter.

    Returns:
      - A counter incremented by b, or max_int if the maximum precision is reached.
    """
    chex.assert_type(count, jnp.int32)
    max_int32_value = jnp.iinfo(jnp.int32).max
    add_counter = jnp.array(add_counter, dtype=jnp.int32)
    return jnp.where(count < max_int32_value, count + add_counter, max_int32_value)


def add_noise_wellingteh(
    eta: float = 0.1,
    gamma: float = 0.55,
    add_counter: int = 1,
    seed: int = 0,
) -> optax.GradientTransformation:
    r"""
    Add gradient noise according to [Welling and Teh, 2011].

    The noise is sampled from a Gaussian distribution with variance :math:`\eta_t`
    where:

    :math:`\eta_t = \frac{\eta}{(t + b)^\gamma}`
    where :math:`\eta` is the initial noise variance, :math:`t` is the iteration counter,
    :math:`b` is a constant (add_counter) added to the counter
    and :math:`\gamma` is a decay exponent.

    Modified from the add_noise method of
    https://github.com/google-deepmind/optax/blob/master/optax/_src/transform.py
    but we follow the noise schedule scheme of [Welling and Teh, 2011].

    References:
      - [Welling and Teh, 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)

    Args:
      - eta: Initial variance :math:`\eta` of the gaussian noise added to the gradient.
      - gamma: Decay exponent :math:`\gamma` for annealing of the variance.
      - add_counter: Increment of counter :math:`b`.
      - seed: Seed for random number generation.

    Returns:
      - A `GradientTransformation` object.
    """

    def init_fn(params: PyTree) -> optax.AddNoiseState:
        """Initialize state function for the GradientTransformation.

        Args:
            - params: Parameters.

        Returns:
            - An `optax.AddNoiseState` object.
        """
        del params
        return optax.AddNoiseState(
            count=jnp.zeros([], jnp.int32), rng_key=jax.random.PRNGKey(seed)
        )

    def update_fn(
        updates: PyTree, state: optax.AddNoiseState, params: Optional[PyTree] = None
    ) -> tuple[PyTree, optax.AddNoiseState]:
        """Update state function for the GradientTransformation.

        Args:
            - updates: The gradient updates.
            - state: The current `optax.AddNoiseState`.
            - params: Parameters.

        Returns:
            - The gradient updates with added noise.
            - The updated `optax.AddNoiseState`.
        """
        del params
        num_vars = len(jax.tree_util.tree_leaves(updates))
        treedef = jax.tree_util.tree_structure(updates)
        count_inc = safe_int32_increment(state.count, add_counter)
        variance = eta / count_inc**gamma
        standard_deviation = jnp.sqrt(variance)
        all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
        noise = jax.tree_util.tree_map(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
            updates,
            jax.tree_util.tree_unflatten(treedef, all_keys[1:]),
        )
        updates = jax.tree_util.tree_map(
            lambda g, n: g + standard_deviation.astype(g.dtype) * n,
            updates,
            noise,
        )
        return updates, optax.AddNoiseState(count=count_inc, rng_key=all_keys[0])

    return optax.GradientTransformation(init_fn, update_fn)


class ScaleByPreconditioningState(NamedTuple):
    """State for the Preconditioning algorithm.

    Adapted from optax.

    Attributes:
        - counter: The current count of the number of steps.
        - mu: The current mean of the gaussian noise.
        - standard_deviations: The current standard deviations of the gaussian noise.
        - rng_key: The current random number generator key.
    """

    counter: chex.Numeric  # shape=(), dtype=jnp.int32.
    mu: optax.Updates
    standard_deviations: optax.Updates
    rng_key: chex.PRNGKey


def update_variance_preconditioning(
    updates: optax.Updates,
    previous_stds: optax.Updates,
    mu: optax.Updates,
    mu1: optax.Updates,
    momentum: float,
    order: chex.Numeric = 1,
) -> optax.Updates:
    r"""Compute the standard deviations :math:`C_t` when using preconditioning in SGLD.

    :math:`C_t = \rho C_{t-1} + (1-\rho) [\grad loss(\theta_t) - \mu_t]^p
    [\grad loss(\theta_t) - \mu_{t-1}]^p`

    Code is inspired from the optax.update_moment function.

    Args:
        - updates: The gradient updates.
        - previous_stds: The standard deviations to update :math: `C_{t-1}`.
        - mu: The current mean of the gaussian noise :math: `\mu_t`.
        - mu1: The previous mean of the the gaussian noise :math: `\mu_{t-1}`.
        - momentum: The decay parameter :math:`\rho`.
        - order: The order p of the moment.

    References:
      - [Bhardwaj, 2019](https://arxiv.org/abs/1906.04324)

    Returns:
        - The updated variance.
    """
    return jax.tree_util.tree_map(
        lambda g, mubefore, mucurrent, t: momentum * t
        + (1 - momentum) * ((g - mucurrent) ** order) * ((g - mubefore) ** order),
        updates,
        mu1,
        mu,
        previous_stds,
    )


def add_noise_with_preconditioning(
    momentum: float = 0.0,
    noise_factor: float = 1.0,
    seed: int = 0,
) -> optax.GradientTransformation:
    r"""
    Add gradient noise according to [Bhardwaj, 2019].

    Modified from the add_noise method of
    https://github.com/google-deepmind/optax/blob/master/optax/_src/transform.py
    but we follow the noise schedule scheme of [Bhardwaj, 2019].

    The noise mean :math:`\mu_t` and standard deviation `C_t` are obtained by:

    :math:`\mu_t = \rho \mu_{t-1} + (1-\rho) \grad loss(\theta_t)`
    :math:`C_t = \rho C_{t-1} + (1-\rho) [\grad loss(\theta_t) - \mu_t]
    [\grad loss(\theta_t) - \mu_{t-1}]`


    Additionally, one can multiply the noise with a factor noise_factor.

    References:
      - [Bhardwaj, 2019](https://arxiv.org/abs/1906.04324)

    Args:
      - eta: Initial variance :math:`\eta` of the gaussian noise added to the gradient.
      - gamma: Decay exponent :math:`\gamma` for annealing of the variance.
      - add_counter: Increment of counter :math:`b`.
      - noise_factor: Factor to multiply the noise with.
      - seed: Seed for random number generation.

    Returns:
      - A `GradientTransformation` object.
    """

    def init_fn(params: PyTree) -> ScaleByPreconditioningState:
        """Initialize state function for the GradientTransformation.

        Args:
            - params: Parameters.

        Returns:
            - A `ScaleByPreconditioningState` object.
        """
        mu = jax.tree_util.tree_map(jnp.zeros_like, params)
        cond = jax.tree_util.tree_map(jnp.zeros_like, params)
        return ScaleByPreconditioningState(
            counter=jnp.zeros([], jnp.int32),
            mu=mu,
            standard_deviations=cond,
            rng_key=jax.random.PRNGKey(seed),
        )

    def update_fn(
        updates: PyTree,
        state: ScaleByPreconditioningState,
        params: Optional[PyTree] = None,
    ) -> tuple[PyTree, ScaleByPreconditioningState]:
        """Update state function for the GradientTransformation.

        Args:
            - updates: The gradient updates.
            - state: The current `ScaleByPreconditioningState`.
            - params: Parameters.

        Returns:
            - The gradient updates with added noise.
            - The updated `ScaleByPreconditioningState`.
        """
        del params
        num_vars = len(jax.tree_util.tree_leaves(updates))
        treedef = jax.tree_util.tree_structure(updates)

        # update counter
        count_inc = safe_int32_increment(state.counter, 1)

        # Update of the mean mu_t
        mu = optax.update_moment(updates, state.mu, momentum, 1)

        # Update of the standard deviations C_t
        standard_deviations = update_variance_preconditioning(
            updates, state.standard_deviations, mu, state.mu, momentum, order=1
        )

        # Generate normal samples with mean 0 and std 1
        all_keys = jax.random.split(state.rng_key, num=num_vars + 1)
        noise = jax.tree_util.tree_map(
            lambda g, k: jax.random.normal(k, shape=g.shape, dtype=g.dtype),
            updates,
            jax.tree_util.tree_unflatten(treedef, all_keys[1:]),
        )

        # Add to the gradient the noise with mean mu_t and std C_t
        updates = jax.tree_util.tree_map(
            lambda g, mean_noise, std_noise, n: g
            + mean_noise
            + n * noise_factor * std_noise,
            updates,
            mu,
            standard_deviations,
            noise,
        )
        return updates, ScaleByPreconditioningState(
            counter=count_inc,
            mu=mu,
            standard_deviations=standard_deviations,
            rng_key=all_keys[0],
        )

    return optax.GradientTransformation(init_fn, update_fn)


def get_wellingteh_schedule(
    eta: float = 0.1,
    gamma: float = 0.55,
    add_counter: int = 1,
) -> Callable:
    r"""
    Get a schedule of the learning rate :math:`\epsilon_t` similar to [Welling and Teh, 2011].

    The schedule of the learning rate is:
    :math:`\epsilon_t = \frac{\eta}{(t + b)^\gamma}`
    where :math:`\eta` is the initial learning rate, :math:`t` is the iteration counter,
    :math:`b` is a constant (add_counter) added to the counter
    and :math:`\gamma` is a decay exponent.

    References:
      - [Welling and Teh, 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)


    Args:
      - eta: Base variance :math:`\eta` of the gaussian noise added to the gradient.
      - gamma: Decay exponent for annealing of the variance.
      - add_counter: Increment of counter.
      - seed: Seed for random number generation.

    Returns:
      - A schedule function for the learning rate.
    """

    def schedule(count: int) -> chex.Numeric:
        """Schedule function for the learning rate.

        Args:
            - count: The current count of the number of steps.

        Returns:
            - The learning rate.
        """
        count_inc = safe_int32_increment(count, add_counter)
        decayed_value = eta / count_inc**gamma
        return decayed_value

    return schedule


def sgld(
    learning_rate: float = 0.01,
    scale_factor: float = -1.0,
    gamma: float = 0.55,
    add_to_counter: int = 1,
    momentum: float = 0.0,
    use_preconditioning: bool = False,
    noise_factor: float = 1.0,
    seed: int = 0,
) -> optax.GradientTransformation:
    r"""
    Get an optax variant of the Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

    This is adapted from Eq.4 of [Welling and Teh, 2011] without a prior term.
    Gradients are updated with added noise and using a scheduler for the learning rate.
    The added noise variance can follow the learning rate schedule [Welling and Teh, 2011].

    The learning rate :math:`\epsilon_t` at iteration :math:`t` is:
    :math:`\epsilon_t = \frac{\eta}{(t + b)^\gamma}`
    where :math:`\eta` is the initial learning rate,
    :math:`t` is the iteration counter,
    :math:`b` is a constant added to the counter (add_to_counter),
    and :math:`\gamma` is a decay exponent.

    The learning rate is multiplied with scale_factor.

    One can also add pre-conditioning as in [Bhardwaj, 2019].
    In this case, an adaptive preconditioner is used when adding noise
    to the gradients based on a diagonal approximation of second order moment of gradient updates.
    One uses a momentum parameter :math:`\rho` and a noise_factor in this case.
    The noise mean :math:`\mu_t` and standard deviation `C_t` are obtained by:
    :math:`\mu_t = \rho \mu_{t-1} + (1-\rho) \grad loss(\theta_t)`
    :math:`C_t = \rho C_{t-1} + (1-\rho) [\grad loss(\theta_t) - \mu_t]
    [\grad loss(\theta_t) - \mu_{t-1}]`

    References:
      - [Welling and Teh, 2011](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf)
      - [Bhardwaj, 2019](https://arxiv.org/abs/1906.04324)

    Args:
      - learning_rate: Initial learning rate and variance for the Gaussian noise.
      - scale_factor: A fixed global scaling factor for the learning rate.
      Note that in Welling and Teh, should set to the number of datapoints N (or minus N).
      - gamma: A parameter controlling the annealing of noise over iterations,
      the variance decays according to `(b+t)^-\gamma`.
      - add_to_counter: Added to counter of iterations :math:`b`.
      - momentum: Momentum parameter for the SGD update.
      - use_preconditioning: Whether to use preconditioning of the noise variance.
      - noise_factor: Factor to multiply the noise with.
      - seed: Seed for the pseudo-random generation process.

    Returns:
      - The corresponding `GradientTransformation`.
    """
    if use_preconditioning:
        return optax.chain(
            add_noise_with_preconditioning(momentum, noise_factor, seed),
            optax.scale_by_schedule(
                get_wellingteh_schedule(learning_rate, gamma, add_to_counter)
            ),
            optax.scale(scale_factor),
        )

    return optax.chain(
        optax.scale_by_schedule(
            get_wellingteh_schedule(learning_rate, gamma, add_to_counter)
        ),
        optax.scale(0.5 * scale_factor),
        add_noise_wellingteh(learning_rate, gamma, add_to_counter, seed),
    )
