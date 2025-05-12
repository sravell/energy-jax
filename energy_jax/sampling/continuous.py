"""Sampling from continuous EBMs."""

from typing import Tuple, Dict, Optional, Any, Union
import jax
from jax.scipy.stats import multivariate_normal
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree
import diffrax
import equinox as eqx
import blackjax  # type: ignore[import]
from jaxtyping import Float, Array, PRNGKeyArray, PyTree
from ..ebms.ebm import AbstractModel
from .sampler import AbstractContinuousSampler
from .adaptation import AbstractAdaptiveSampler, ErrorAdaptationState
from ..utils import scan_wrapper, random_uniform_pytree


def langevin_step(
    model: AbstractModel,
    stepsize: Union[float, Float[Array, ""]],
    x_k: PyTree[Float[Array, "..."]],
    sigma: Union[float, Float[Array, ""]],
    key: PRNGKeyArray,
    grad_clip: Optional[float] = None,
    state_clip: Optional[Tuple[float, float]] = None,
    is_score_network: bool = False,
    score_sigma: Optional[Float[Array, ""]] = None,
    **kwargs: Any,
) -> Tuple[
    PyTree[Float[Array, "..."]],
    PyTree[Float[Array, "..."]],
    Union[Float[Array, ""], None],
]:
    r"""
    Perform a single step of Langevin Monte Carlo Markov Chain (MCMC) to update the current state.

    Langevin MCMC combines gradient information with random walk exploration to sample from
    a target distribution. The update rule for a single step is as follows:

    $$ x_{k+1} = x_k - \eta \nabla E(x_k) + N(0, \sigma^2) $$

    A common approach is to set $ \sigma = \sqrt{2 * \eta} $

    Args:
        - model: the neural network
        - stepsize: the eta parameter, how much to use the gradient
        - x_k: the current input
        - sigma: the variance of the noise
        - key: the random key to use
        - grad_clip: clipping the gradients in langevin dynamics
        - state_clip: clipping the newly proposed state
        - is_score_network: whether or not to use the score or compute the gradient
            manually
        - score_sigma: the noise level to put into the score network

    Returns:
        - the next langevin sample
        - the gradient of the previous sample
        - the energy of x_k (if available, i.e. not for scores)
    """
    if is_score_network:
        gradient = -1 * model.score(x_k, sigma=score_sigma, **kwargs)
    else:
        energy, gradient = eqx.filter_value_and_grad(model.energy_function)(
            x_k, **kwargs
        )
    grad_flat, unflat = ravel_pytree(gradient)
    if grad_clip is not None:
        gradient = jnp.clip(gradient, -grad_clip, grad_clip)
    xk_flat, _ = ravel_pytree(x_k)
    x_k1 = (
        xk_flat
        - stepsize * grad_flat
        + jax.random.normal(key, shape=grad_flat.shape) * sigma
    )
    if state_clip is not None:
        x_k1 = jnp.clip(x_k1, state_clip[0], state_clip[1])
    if is_score_network:
        energy_out = None
    else:
        energy_out = energy
    x_k1 = unflat(x_k1)
    return x_k1, gradient, energy_out


class LangevinSampler(AbstractContinuousSampler, strict=True):
    r"""
    Sample from an EBM using langevin method.

    Note that the model must output E(x) for p~exp(-E(x)), to use with generic
    log prob functions, you should pass lambda x : -1 * model(x).

    If you are using MALA, it can sometimes be necessary to decrease the learning rate
    to increase the acceptance rate.

    Attributes:
        - stepsize: eta, how much to update based on the gradient
        - sigma: the variance of the noise
        - num_langevin_steps: how many langevin steps to take
        - sample_clip: clipping to use for the generated samples
        - grad_clip: clipping the gradients in langevin dynamics
        - metropolis_adjustment: whether or not to do a metropolis conditioned accepted,
            i.e. generate the accept probability based on
            $ \min \left [ 1, \frac{p(x')q(x|x')}{p(x)q(x'|x)} \right ] $
        - return_full_chain: whether or no to return the full chain rather than just
            the last sample

    Returns:
        - a dictionary with the sample and energy
    """

    stepsize: float
    sigma: float
    num_langevin_steps: int
    sample_clip: Optional[Tuple[float, float]]
    grad_clip: Optional[float]
    metropolis_adjustment: bool
    return_full_chain: bool
    xshape: Union[Tuple[int], PyTree, None]
    num_chains: Union[int, None]
    minval: Union[float, None]
    maxval: Union[float, None]

    def __init__(
        self,
        xshape: Union[Tuple[int], PyTree, None],
        num_chains: Union[int, None],
        minval: Union[float, None],
        maxval: Union[float, None],
        stepsize: float,
        sigma: float,
        num_langevin_steps: int,
        sample_clip: Optional[Union[Tuple[float, float], float]] = None,
        grad_clip: Optional[float] = None,
        metropolis_adjustment: bool = False,
        return_full_chain: bool = False,
    ) -> None:
        """Initialize member variables."""
        self.stepsize = stepsize
        self.sigma = sigma
        if isinstance(sample_clip, float):
            self.sample_clip = (-sample_clip, sample_clip)
        else:
            self.sample_clip = sample_clip
        self.grad_clip = grad_clip
        self.num_langevin_steps = num_langevin_steps
        self.metropolis_adjustment = metropolis_adjustment
        self.xshape = xshape
        self.num_chains = num_chains
        self.minval = minval
        self.maxval = maxval
        self.return_full_chain = return_full_chain

    def step(
        self,
        model: AbstractModel,
        state: Union[PyTree[Float[Array, "..."]], Float[Array, "xshape"]],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Union[PyTree[Float[Array, "..."]], Float[Array, "xshape"]]:
        """
        Generate a single (potentially MALA) langevin step.

        For more information see:
        - https://pints.readthedocs.io/en/stable/mcmc_samplers/mala_mcmc.html
        - https://en.wikipedia.org/wiki/Metropolis-adjusted_Langevin_algorithm

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the random key to use

        Returns:
            - the next position
        """
        current_position = state
        key, subkey = jax.random.split(key)
        next_position, current_gradient, current_energy = langevin_step(
            model,
            self.stepsize,
            current_position,
            self.sigma,
            subkey,
            self.grad_clip,
            self.sample_clip,
            **kwargs,
        )
        if not self.metropolis_adjustment:
            return next_position
        next_energy, next_gradient = eqx.filter_value_and_grad(model.energy_function)(
            next_position, **kwargs
        )
        energy_difference = -next_energy + current_energy  # EBM

        pos_flat, _ = ravel_pytree(current_position)
        grad_flat, _ = ravel_pytree(current_gradient)
        next_flat, _ = ravel_pytree(next_position)
        next_grad_flat, _ = ravel_pytree(next_gradient)
        # The resulting proposals are Gaussians, adapted from
        # https://github.com/kazewong/flowMC/blob/main/src/flowMC/sampler/MALA.py#L73
        qx_x1 = multivariate_normal.logpdf(
            pos_flat,
            next_flat - self.stepsize * next_grad_flat,
            self.sigma,
        )
        # The next distribution is N(x1 | x0 - stepsize * grad(E), sigma * I)
        # we want the log probability of this proposal to compute the q'/q
        qx1_x = multivariate_normal.logpdf(
            next_flat,
            pos_flat - self.stepsize * grad_flat,
            self.sigma,
        )
        if not isinstance(qx1_x, jax.Array) or not isinstance(qx_x1, jax.Array):
            raise TypeError("Error in hastings correction.")
        hastings = qx_x1 - qx1_x
        acceptance = energy_difference + hastings

        key, subkey = jax.random.split(key)
        step_position = jax.lax.cond(
            jnp.log(jax.random.uniform(subkey)) >= acceptance,
            lambda _: current_position,
            lambda _: next_position,
            None,
        )
        return step_position

    def run_chain(
        self,
        model: AbstractModel,
        state: Union[PyTree[Float[Array, "..."]], Float[Array, "xshape"], None],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict:
        """
        Run one chain of Langevin MCMC.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the random key to use

        Returns:
            - the results of one chain
        """
        if state is None:
            key, subkey = jax.random.split(key, 2)
            initial_state = self._random_initial_state(subkey)
        else:
            initial_state = state

        one_step = scan_wrapper(
            lambda state, key: self.step(model, state, key, **kwargs),
            self.return_full_chain,
        )

        keys = jax.random.split(key, self.num_langevin_steps)
        final_state, all_states = jax.lax.scan(one_step, initial_state, keys)

        energy = eqx.filter_jit(model.energy_function)(final_state, **kwargs)
        if self.return_full_chain:
            return {"position": all_states, "energy": None}
        return {"position": final_state, "energy": energy}


class ErrorAdaptiveLangevinSampler(
    AbstractAdaptiveSampler,
    AbstractContinuousSampler,
    strict=True,
):
    """
    Langevin sampling with adaptative step sizes that adjusts the stepsize based on the estimated error.

    Error is computed via $$E_{max} = r_tol*max(max(2_step_state), max(full_state)) + a_tol$$

    Adapted from: https://github.com/google-research/torchsde/blob/master/torchsde/_core/base_solver.py#L117

    Note: currently does not support metropolis adjustment.

    Attributes:
        - rtol: relative tolerance for error estimation
        - atol: absolute tolerance for error estimation
        - epsilon: the minimum error
        - dt_min: the minimum stepsize
    """

    stepsize: float
    sigma: float
    num_langevin_steps: int
    sample_clip: Optional[Tuple[float, float]]
    grad_clip: Optional[float]
    metropolis_adjustment: bool
    rtol: Float[Array, ""]
    atol: Float[Array, ""]
    epsilon: Float[Array, ""]
    dt_min: Float[Array, ""]
    return_full_chain: bool
    xshape: Union[Tuple[int], PyTree, None]
    num_chains: Union[int, None]
    minval: Union[float, None]
    maxval: Union[float, None]

    def __init__(
        self,
        xshape: Union[Tuple[int], PyTree, None],
        num_chains: Union[int, None],
        minval: Union[float, None],
        maxval: Union[float, None],
        stepsize: float,
        sigma: float,
        num_langevin_steps: int,
        rtol: float,
        atol: float,
        epsilon: float,
        dt_min: float,
        sample_clip: Optional[Union[Tuple[float, float], float]] = None,
        grad_clip: Optional[float] = None,
        return_full_chain: bool = False,
    ) -> None:
        """Initialize member variables."""
        self.rtol = jnp.array(rtol)
        self.atol = jnp.array(atol)
        self.epsilon = jnp.array(epsilon)
        self.dt_min = jnp.array(dt_min)
        self.stepsize = stepsize
        self.sigma = sigma
        if isinstance(sample_clip, float):
            self.sample_clip = (-sample_clip, sample_clip)
        else:
            self.sample_clip = sample_clip
        self.grad_clip = grad_clip
        self.num_langevin_steps = num_langevin_steps
        self.metropolis_adjustment = False
        self.xshape = xshape
        self.num_chains = num_chains
        self.minval = minval
        self.maxval = maxval
        self.return_full_chain = return_full_chain

    def _get_keys(self, key: PRNGKeyArray) -> PRNGKeyArray:
        """Get the keys used for the chain."""
        return jax.random.split(key, self.num_langevin_steps)

    def _init_adaptive(self) -> ErrorAdaptationState:
        return ErrorAdaptationState(jnp.array(self.stepsize))

    def step(
        self,
        model: AbstractModel,
        state: Union[PyTree[Float[Array, "..."]], Float[Array, "xshape"]],
        key: PRNGKeyArray,
        adaptive_state: Optional[ErrorAdaptationState] = None,  # type: ignore[override]
        **kwargs: Any,
    ) -> Tuple[PyTree, ErrorAdaptationState]:
        """
        Take a single adaptive step.

        Do a full langevin step, then two half steps. Compute the error between
        these two estimations and scale the stepsize accordingly. This is known
        as Richardson Extrapolation.

        Adapted from: https://github.com/google-research/torchsde/blob/master/torchsde/_core/base_solver.py#L117

        Args:
            - model: the energy function to sample from
            - state: the input state to the step
            - key: the random key to use for the step
            - adaptive_state: the state manager for the adaptive memory

        Returns
            - the updated state and also the updated adaptation state
        """
        if adaptive_state is None:
            raise ValueError("adaptive_state must not be None!")

        init_pos = state
        sigma = self.sigma / jnp.sqrt(self.stepsize / adaptive_state.current_step_size)
        key, subkey, subkey1, subkey2 = jax.random.split(key, 4)
        full_state, _, _ = langevin_step(
            model,
            adaptive_state.current_step_size,
            init_pos,
            sigma,
            subkey,
            self.grad_clip,
            self.sample_clip,
            **kwargs,
        )
        mid_point, _, _ = langevin_step(
            model,
            adaptive_state.current_step_size / 2.0,
            init_pos,
            sigma,
            subkey1,
            self.grad_clip,
            self.sample_clip,
            **kwargs,
        )
        two_step_state, _, _ = langevin_step(
            model,
            adaptive_state.current_step_size / 2.0,
            mid_point,
            sigma,
            subkey2,
            self.grad_clip,
            self.sample_clip,
            **kwargs,
        )
        tolerance = (
            self.rtol
            * jnp.maximum(
                jnp.max(jnp.abs(two_step_state)), jnp.max(jnp.abs(full_state))
            )
            + self.atol
        )
        tolerance = jnp.clip(tolerance, a_min=self.epsilon)
        error_estimate = (two_step_state - full_state) / tolerance
        error_estimate = jnp.clip(
            jnp.sqrt(jnp.mean(error_estimate**2)), a_min=self.epsilon
        )

        next_step_size, prev_error_ratio = self._update_step_size(
            error_estimate,
            adaptive_state.current_step_size,
            prev_error_ratio=adaptive_state.prev_error_ratio,
        )

        next_step_size, final_error_ratio = jax.lax.cond(
            next_step_size < self.dt_min,
            lambda _: (self.dt_min, jnp.array(-1.0)),
            # -1.0 is basically None, but jax doesn't like None here
            lambda _: (next_step_size, prev_error_ratio),
            None,
        )

        next_adaptive_state = ErrorAdaptationState(next_step_size, final_error_ratio)

        return jax.lax.cond(
            (error_estimate <= 1.0) | (next_step_size <= self.dt_min),
            lambda _: (two_step_state, next_adaptive_state),
            lambda _: (init_pos, next_adaptive_state),
            None,
        )

    def _update_step_size(
        self,
        error_est: Float[Array, ""],
        prev_step_size: Float[Array, ""],
        safety: float = 0.9,
        facmin: float = 0.2,
        facmax: float = 1.4,
        prev_error_ratio: Optional[Float[Array, ""]] = None,
    ) -> Tuple[Float[Array, ""], Float[Array, ""]]:
        """
        Update the step size.

        The ifactors and pfactors dampend the adjustment of the step size. If $E/E_{max} > 1$
        then we want to adjust the step size so we increase the i_factor. Then the step size
        is adjusted with the factor $(0.9*||E/E_{max}||^{-1})^{2/3} $.

        If $E/E_{max} < 1$ then it is adjusted by
        $(0.9*||E/E_{max}||^{-1})^{2/9}  * (||E/E_{max}||^{prev} / ||E/E_{max}||)^{0.13}$
        Where $prev$ means from the previous step.

        The point of these factors is, that 2/9 exponent is small enough that a^2/3 is very
        close to 1, except the argument diverges a lot from 1 which is what we want if
        the error < E_max. The potence 2/3 on the other side has a smaller effect on the
        argument. So $(0.9*||E/E_{max}||^{-1})^{2/3} $ is always substantially smaller than 1.

        Adapted from: https://github.com/google-research/torchsde/blob/master/torchsde/_core/adaptive_stepping.py#L21.

        Args:
            - error_est: the estimated error
            - prev_step_size: the previous step size
            - safety: what to scale the error ratio by,
                error_ratio = safety / error_estimate
            - facmin: the minimum scaling factor to multiple the stepsize by
            - facmax: the maximum scaling factor to multiple the stepsize by
            - prev_error_ratio: the previous iteration's error ratio

        Returns:
            - the updated step size, updated error ratio
        """
        pfactor, ifactor = jax.lax.cond(
            error_est > 1.0, lambda _: (0.0, 1 / 1.5), lambda _: (0.13, 1 / 4.5), None
        )
        error_ratio = safety / error_est
        if prev_error_ratio is None:
            prev_error_ratio = error_ratio
        prev_error_ratio = jax.lax.cond(
            prev_error_ratio < 0,
            lambda _: error_ratio,
            lambda _: prev_error_ratio,
            None,
        )
        factor = error_ratio**ifactor * (error_ratio / prev_error_ratio) ** pfactor
        updated_error_ratio, facmin = jax.lax.cond(
            error_est <= 1.0,
            lambda _: (error_ratio, 1.0),
            lambda _: (prev_error_ratio, facmin),
            None,
        )
        factor = jnp.clip(factor, facmin, facmax)
        new_step_size = prev_step_size * factor
        return new_step_size, updated_error_ratio


# TODO: new diffrax 0.6.0
# TODO: andraz PR
class OverdampedLangevinSDESampler(AbstractContinuousSampler, strict=True):
    """
    Sample the Langevin SDE.

    Solves the SDE: dx/dt = - grad(E(x)) dt + sigma dW

    TODO: Render equation
    TODO: Handle kwargs/args better

    Attributes:
        - initial_stepsize: the initial stepsize of the solver
        - sigma: the noise term
        - max_solver_steps: the maximum number of steps the solver takes before
            failing unconditionally
        - solver: the solver to use (can be adaptive)
        - saves: the saved datapoints
        - initial_time: the starting time to integrate
        - final_time: the ending time to integrate
        - brownian_tolerance: the tolerance of the discretization of the brownian motion
    """

    initial_stepsize: Union[float, None]
    sigma: Union[float, PyTree[float]]
    max_solver_steps: int
    solver: diffrax.AbstractSolver
    saves: diffrax.SaveAt
    initial_time: float
    final_time: float
    brownian_shape: Union[jax.ShapeDtypeStruct, PyTree[jax.ShapeDtypeStruct]]
    brownian_tolerance: float
    adaptive_controller: diffrax.AbstractStepSizeController
    xshape: Union[Tuple[int], PyTree, None]
    num_chains: Union[int, None]
    minval: Union[float, None]
    maxval: Union[float, None]

    def __init__(
        self,
        xshape: Union[Tuple[int], PyTree],
        num_chains: Union[int, None],
        minval: Union[float, None],
        maxval: Union[float, None],
        initial_stepsize: Union[float, None],
        sigma: Union[float, PyTree[float]],
        num_samples_per_chain: int,
        max_solver_steps: int,
        solver: diffrax.AbstractSolver,
        initial_time: float,
        burn_in_time: float,
        final_time: float,
        brownian_shape: Union[Tuple[int], PyTree[jax.ShapeDtypeStruct]],
        brownian_tolerance: float = 1e-4,
        adaptive_controller: Optional[
            diffrax.AbstractAdaptiveStepSizeController
        ] = None,
    ) -> None:
        """Initialize member variables."""
        self.xshape = xshape
        self.num_chains = num_chains
        self.minval = minval
        self.maxval = maxval
        self.initial_stepsize = initial_stepsize
        self.sigma = sigma
        self.max_solver_steps = max_solver_steps
        self.solver = solver
        self.initial_time = initial_time
        self.final_time = final_time
        if num_samples_per_chain == -1:
            self.saves = diffrax.SaveAt(t1=True)
        else:
            self.saves = diffrax.SaveAt(
                ts=jnp.linspace(burn_in_time, final_time, num_samples_per_chain)
            )
        if isinstance(brownian_shape, tuple):
            self.brownian_shape = jax.ShapeDtypeStruct(brownian_shape, "float32")
        else:
            self.brownian_shape = brownian_shape
        self.brownian_tolerance = brownian_tolerance
        self.adaptive_controller = (
            adaptive_controller
            if adaptive_controller is not None
            else diffrax.ConstantStepSize()
        )

    def step(
        self, model: AbstractModel, state: PyTree, key: PRNGKeyArray, **kwargs: Any
    ) -> PyTree:
        """Take a single step of the chain."""
        raise NotImplementedError

    def run_chain(
        self,
        model: AbstractModel,
        state: Union[PyTree, None],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict[str, PyTree]:
        """
        Run one chain of Langevin SDE sampling.

        Note: does not return an energy.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the random key to use

        Returns:
            - the results of one chain
        """
        if state is None:
            key, subkey = jax.random.split(key)
            state = self._random_initial_state(subkey)
        drift = lambda t, y, args: model.score(y, **args)
        if isinstance(self.sigma, float):
            diffusion = lambda t, y, args: jax.tree_util.tree_map(
                lambda i: self.sigma * jnp.ones_like(i), y
            )
        else:
            diffusion = lambda t, y, args: self.sigma
        brownian_motion = diffrax.VirtualBrownianTree(
            self.initial_time,
            self.final_time,
            tol=self.brownian_tolerance,
            shape=self.brownian_shape,
            key=key,
        )
        terms: diffrax.MultiTerm = diffrax.MultiTerm(
            diffrax.ODETerm(drift),  # type: ignore
            diffrax.WeaklyDiagonalControlTerm(diffusion, brownian_motion),  # type: ignore
        )
        sol = diffrax.diffeqsolve(
            terms,
            self.solver,
            self.initial_time,
            self.final_time,
            args=kwargs,
            dt0=self.initial_stepsize,
            y0=state,
            saveat=self.saves,
            max_steps=self.max_solver_steps,
            stepsize_controller=self.adaptive_controller,
        )
        sde_samples = sol.ys
        if self.saves.subs.t1:
            sde_samples = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), sde_samples)
        return {"position": sde_samples, "energy": None}


class UnderdampedLangevinSDESampler(AbstractContinuousSampler, strict=True):
    """
    Solve the Langevin SDE with hamiltonian dynamics.

    Solves the coupled SDE:
    dx/dt = dH/dp
    dp/dt = -gamma * p - dH/dx + sigma dW
    where H = p^2/2 + E(x)

    Note: depending on the solver, this may not scale as quickly on wall clock
    time as the other solvers. Depending on the problem and solver, things may
    get slow >100 dimensions.

    Attributes:
        - gamma: the friction
        - momentum_init_range: the range for random initialization of momentum
    """

    initial_stepsize: Union[float, None]
    sigma: Union[float, PyTree[float]]
    max_solver_steps: int
    solver: diffrax.AbstractSolver
    saves: diffrax.SaveAt
    initial_time: float
    final_time: float
    brownian_shape: Union[jax.ShapeDtypeStruct, PyTree[jax.ShapeDtypeStruct]]
    brownian_tolerance: float
    adaptive_controller: diffrax.AbstractStepSizeController
    gamma: Union[float, PyTree[float]]
    momentum_init_range: Tuple[float, float]
    xshape: Union[Tuple[int], PyTree, None]
    num_chains: Union[int, None]
    minval: Union[float, None]
    maxval: Union[float, None]

    def __init__(
        self,
        xshape: Union[Tuple[int], PyTree[jax.ShapeDtypeStruct]],
        num_chains: Union[int, None],
        minval: Union[float, None],
        maxval: Union[float, None],
        initial_stepsize: Union[float, None],
        sigma: Union[float, PyTree[float]],
        num_samples_per_chain: int,
        max_solver_steps: int,
        solver: diffrax.AbstractSolver,
        initial_time: float,
        burn_in_time: float,
        final_time: float,
        brownian_shape: Union[Tuple[int], PyTree[jax.ShapeDtypeStruct]],
        gamma: Union[float, PyTree[float]],
        brownian_tolerance: float = 1e-4,
        adaptive_controller: Optional[
            diffrax.AbstractAdaptiveStepSizeController
        ] = None,
        momentum_init_range: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Initialize member variables."""
        self.xshape = xshape
        self.num_chains = num_chains
        self.minval = minval
        self.maxval = maxval
        self.initial_stepsize = initial_stepsize
        self.sigma = sigma
        self.max_solver_steps = max_solver_steps
        self.solver = solver
        self.initial_time = initial_time
        self.final_time = final_time
        if num_samples_per_chain == -1:
            self.saves = diffrax.SaveAt(t1=True)
        else:
            self.saves = diffrax.SaveAt(
                ts=jnp.linspace(burn_in_time, final_time, num_samples_per_chain)
            )
        if isinstance(brownian_shape, tuple):
            self.brownian_shape = jax.ShapeDtypeStruct(brownian_shape, "float32")
        else:
            self.brownian_shape = brownian_shape
        self.brownian_tolerance = brownian_tolerance
        self.adaptive_controller = (
            adaptive_controller
            if adaptive_controller is not None
            else diffrax.ConstantStepSize()
        )
        self.gamma = gamma
        self.momentum_init_range = (
            momentum_init_range if momentum_init_range is not None else (0.0, 0.0)
        )

    def step(
        self, model: AbstractModel, state: PyTree, key: PRNGKeyArray, **kwargs: Any
    ) -> PyTree:
        """Take a single step of the chain."""
        raise NotImplementedError

    def run_chain(
        self,
        model: AbstractModel,
        state: Union[PyTree, None],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict[str, PyTree]:
        """
        Run one chain of Langevin Hamiltonian sampling.

        Note: does not return an energy.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the random key to use

        Returns:
            - the results of one chain
        """
        if state is None:
            key, subkey = jax.random.split(key)
            state = self._random_initial_state(subkey)

        def _coupled_drift_term(
            t: float, y: Tuple[PyTree, PyTree], args: PyTree
        ) -> Tuple[PyTree, PyTree]:
            x, p = y
            dpos = p
            if isinstance(self.gamma, float):
                friction = jax.tree_util.tree_map(lambda a: -self.gamma * a, p)
            else:
                friction = jax.tree_util.tree_map(lambda a, b: -a * b, self.gamma, p)
            dmom = jax.tree_util.tree_map(
                lambda a, b: a + b, friction, model.score(x, **args)
            )
            return dpos, dmom

        def _diffusion(
            t: float, y: Tuple[PyTree, PyTree], args: PyTree
        ) -> Tuple[PyTree, PyTree]:
            if isinstance(self.sigma, float):
                return jax.tree_util.tree_map(
                    lambda i: jnp.zeros_like(i, dtype="float32"), y[0]
                ), jax.tree_util.tree_map(lambda i: self.sigma * jnp.ones_like(i), y[1])
            else:
                return jax.tree_util.tree_map(lambda _: 0.0, y[0]), self.sigma

        key, subkey = jax.random.split(key)
        brownian_motion = diffrax.VirtualBrownianTree(
            self.initial_time,
            self.final_time,
            tol=self.brownian_tolerance,
            shape=(self.brownian_shape, self.brownian_shape),
            key=key,
        )
        terms: diffrax.MultiTerm = diffrax.MultiTerm(
            diffrax.ODETerm(_coupled_drift_term),  # type: ignore
            diffrax.WeaklyDiagonalControlTerm(_diffusion, brownian_motion),  # type: ignore
        )

        y0 = (
            state,
            random_uniform_pytree(
                subkey, state, self.momentum_init_range[0], self.momentum_init_range[1]
            ),
        )
        sol = diffrax.diffeqsolve(
            terms,
            self.solver,
            self.initial_time,
            self.final_time,
            args=kwargs,
            dt0=self.initial_stepsize,
            y0=y0,
            saveat=self.saves,
            max_steps=self.max_solver_steps,
            stepsize_controller=self.adaptive_controller,
        )
        assert sol.ys is not None
        sde_samples = sol.ys[0]
        if self.saves.subs.t1:
            sde_samples = jax.tree_map(lambda x: jnp.squeeze(x, axis=0), sde_samples)
        return {"position": sde_samples, "energy": None}


class AnnealedLangevinSampler(AbstractContinuousSampler, strict=True):
    """
    Annealed multilevel langevin sampling.

    This is used in noise conditioned score matching when sampling. This takes
    the highest noise level (with a larger step size) and does langevin sampling,
    then uses the next highest noise level (sigma) and continues this process. Basically,
    decreasing the stepsize and added noise over each trained noise level.

    For more information see:
        - https://arxiv.org/abs/1907.05600 (algo 1)
        - https://arxiv.org/abs/2006.09011 (algo 1)

    Attributes:
        - sigmas: the array of noise levels
        - num_langevin_steps: the number of langevin steps to do per noise level
        - epsilon: the scaling of the stepsize
        - sample_clip: how much to clip generated samples by each step
        - grad_clip: how much to clip the gradients by each step (or the score)
        - score_based: whether or not the input is an EBM or a score network
        - final_denoise: whether or not to do the final denoising step
        - return_full_chain: whether or no to return the full chain rather than just
            the last sample
    """

    sigmas: jnp.ndarray
    num_langevin_steps: int
    epsilon: float
    score_based: bool
    final_denoise: bool
    sample_clip: Optional[Tuple[float, float]]
    grad_clip: Optional[float]
    return_full_chain: bool
    xshape: Union[Tuple[int], PyTree, None]
    num_chains: Union[int, None]
    minval: Union[float, None]
    maxval: Union[float, None]

    def __init__(
        self,
        xshape: Union[Tuple[int], None],
        num_chains: Union[int, None],
        minval: Union[float, None],
        maxval: Union[float, None],
        sigmas: Float[Array, "num_sigmas"],
        num_langevin_steps: int,
        epsilon: float,
        score_based: bool = False,
        final_denoise: bool = True,
        sample_clip: Optional[Union[Tuple[float, float], float]] = None,
        grad_clip: Optional[float] = None,
        return_full_chain: bool = False,
    ) -> None:
        """Initialize member variables."""
        self.epsilon = epsilon
        self.sigmas = sigmas
        if isinstance(sample_clip, float):
            self.sample_clip = (-sample_clip, sample_clip)
        else:
            self.sample_clip = sample_clip
        self.grad_clip = grad_clip
        self.num_langevin_steps = num_langevin_steps
        self.score_based = score_based
        self.final_denoise = final_denoise
        self.xshape = xshape
        self.num_chains = num_chains
        self.minval = minval
        self.maxval = maxval
        self.return_full_chain = return_full_chain

    def step(
        self,
        model: AbstractModel,
        state: Union[PyTree[Float[Array, "..."]], Float[Array, "xshape"]],
        key: PRNGKeyArray,
        stepsize: Optional[Float[Array, ""]] = None,
        score_sigma: Optional[Float[Array, ""]] = None,
        **kwargs: Any,
    ) -> Float[Array, "xshape"]:
        """
        Take a single step of the chain.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the random key to use
            - stepsize: the stepsize to use for the annealed step
            - score_sigma: the noise level to use for the score network

        Returns:
            - the next state
        """
        if stepsize is None or score_sigma is None:
            raise ValueError("stepsize and score_sigma must be defined!")
        variance = jnp.sqrt(2 * stepsize)
        current_position = state
        key, subkey = jax.random.split(key)
        next_position, _, _ = langevin_step(
            model,
            stepsize,
            current_position,
            variance,
            subkey,
            self.grad_clip,
            self.sample_clip,
            self.score_based,
            score_sigma,
            **kwargs,
        )
        return next_position

    def run_chain(
        self,
        model: AbstractModel,
        state: Union[PyTree[Float[Array, "..."]], Float[Array, "xshape"], None],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict:
        """
        Run an annealed chain.

        This anneals over each of the sigmas to generate a final result.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the random key to use

        Returns:
            - the final chain state
        """
        if state is None:
            if self.xshape is None or self.minval is None or self.maxval is None:
                raise ValueError(
                    "xshape and minval and maxval cannot be None if state is not provided!"
                )
            key, subkey = jax.random.split(key, 2)
            initial_state = jax.random.uniform(
                subkey, minval=self.minval, maxval=self.maxval, shape=self.xshape
            )
        else:
            initial_state = state

        def inner(
            carry: Float[Array, "xshape"], x: Float[Array, ""]
        ) -> Tuple[Float[Array, "xshape"], Float[Array, "xshape"]]:
            current_state = carry
            sigma = x
            stepsize = self.epsilon * (sigma / self.sigmas[-1]) ** 2
            one_step = scan_wrapper(
                lambda state, key: self.step(
                    model, state, key, stepsize, sigma, **kwargs
                ),
                self.return_full_chain,
            )
            keys = jax.random.split(key, self.num_langevin_steps)
            current_state, all_states = jax.lax.scan(one_step, current_state, keys)
            return current_state, all_states

        final_state, all_states = jax.lax.scan(inner, initial_state, self.sigmas)

        if self.return_full_chain:
            if self.final_denoise:
                all_states = all_states.at[-1].add(
                    self.sigmas[-1] ** 2
                    * model.score(all_states[-1], sigma=self.sigmas[-1])
                )
            return {"position": all_states, "energy": None}

        if self.final_denoise:
            final_state = final_state + self.sigmas[-1] ** 2 * model.score(
                final_state, sigma=self.sigmas[-1]
            )

        return {"position": final_state, "energy": None}


class HMCSampler(AbstractContinuousSampler, strict=True):
    """
    Run a Hamiltonian Monte Carlo (HMC) chain to sample from the provided model.

    Hamiltonian Monte Carlo (HMC) is a Markov Chain Monte Carlo (MCMC) method that leverages
    the geometry of the target distribution to propose distant moves in the state space,
    reducing the autocorrelation between samples. HMC introduces auxiliary momentum variables
    and simulates Hamiltonian dynamics to generate proposals

    As with the other sampling methods, this model assumes the output of the model
    is E(x) and not -E(x).

    Attributes:
        - num_hmc_steps: The number of HMC steps to perform.
        - step_size: The step size for the leapfrog integrator in the HMC algorithm.
        - num_integration_steps: The number of integration steps for the integrator in
            the HMC algorithm.
        - inverse_mass_matrix: The inverse mass matrix for the HMC algorithm. The shape should
            match the dimensions of the model's input.
        - return_full_chain: whether or no to return the full chain rather than just
            the last sample
    """

    num_hmc_steps: int
    stepsize: float
    num_integration_steps: int
    inverse_mass_matrix: Float[Array, "dims"]
    return_full_chain: bool
    xshape: Union[Tuple[int], PyTree, None]
    num_chains: Union[int, None]
    minval: Union[float, None]
    maxval: Union[float, None]

    def __init__(
        self,
        xshape: Union[Tuple[int], PyTree],
        num_chains: Union[int, None],
        minval: Union[float, None],
        maxval: Union[float, None],
        num_hmc_steps: int,
        stepsize: float,
        num_integration_steps: int,
        inverse_mass_matrix: Float[Array, "dims"],
        return_full_chain: bool = False,
    ) -> None:
        """Initialize member variables."""
        self.num_hmc_steps = num_hmc_steps
        self.stepsize = stepsize
        self.num_integration_steps = num_integration_steps
        self.inverse_mass_matrix = inverse_mass_matrix
        self.xshape = xshape
        self.num_chains = num_chains
        self.minval = minval
        self.maxval = maxval
        self.return_full_chain = return_full_chain

    def step(
        self, model: AbstractModel, state: PyTree, key: PRNGKeyArray, **kwargs: Any
    ) -> PyTree:
        """Take a single step of the chain."""
        raise NotImplementedError

    def run_chain(
        self,
        model: AbstractModel,
        state: Union[PyTree[Float[Array, "..."]], Float[Array, "xshape"], None],
        key: PRNGKeyArray,
        **kwargs: Any,
    ) -> Dict:
        """
        Run one chain of HMC.

        Args:
            - model: the EBM to sample from
            - state: the current state
            - key: the random key to use

        Returns:
            - the result of one chain
        """
        if state is None:
            key, subkey = jax.random.split(key, 2)
            in_state = self._random_initial_state(subkey)
        else:
            in_state = state

        x_0, unflat = ravel_pytree(in_state)

        hmc = blackjax.hmc(
            lambda x: -1
            * model.energy_function(
                unflat(x), **kwargs
            ),  # blackjax hmc wants logprob, but E(x) = -logprob(x)
            self.stepsize,
            self.inverse_mass_matrix,
            self.num_integration_steps,
        )

        initial_state = hmc.init(x_0)
        step_fn = eqx.filter_jit(hmc.step)

        one_step = scan_wrapper(
            lambda state, key: step_fn(key, state)[0], self.return_full_chain
        )  # ignore the info return value

        keys = jax.random.split(key, self.num_hmc_steps)
        final_state, all_states = jax.lax.scan(one_step, initial_state, keys)

        if self.return_full_chain:
            return {
                "position": all_states.position,
                "energy": (
                    final_state.potential_energy
                    if hasattr(final_state, "potential_energy")
                    else final_state.logdensity
                ),  # we have to do this because it was changed in 1.0.0
            }

        return {
            "position": unflat(final_state.position),
            "energy": (
                final_state.potential_energy
                if hasattr(final_state, "potential_energy")
                else final_state.logdensity
            ),  # we have to do this because it was changed in 1.0.0
        }
