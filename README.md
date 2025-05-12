# Energy-jax and Factor-graphs

A comprehensive toolkit for Energy-Based Models (EBMs) and Factor Graphs in JAX.

Implemented in Jax using equinox leverging Jax's JIT (Just in time compiling).

## Overview

This repository contains two main packages:

### Energy-jax (March 2025)
A JAX-based implementation of classical Energy-Based Models (EBMs) that provides various EBM architectures and sampling methods.

### Factor-graphs (May 2025)
A flexible framework for building and working with factor graphs, with support for energy-based models and efficient inference methods.

```
# Install both packages
pip install -e .
```

## key features

### Energy-jax
- EBM implementations (Discrete and Continuous)
- Neural network architectures (MLP, CNN, GNN, Transformer)
- Discrete and continuous sampling methods
- Natural Gradient Descent optimization
- Various loss functions implemented

### Factor-graphs
- Flexible factor graph construction and manipulation
- Support for both discrete and continuous variables
- Efficient Gibbs sampling implementation
- Integration with energy-jax EBMs
- Visualization tools for factor graphs
- Support for evidence-based inference

## Dependencies

Both packages require:
- jax>=0.4.16
- jaxlib>=0.4.16
- jaxtyping>=0.2.23
- equinox>=0.11.2
- numpy>=1.10.0

Additional dependencies for factor-graphs:
- networkx>=2.8.8
- matplotlib>=3.7.1
- hypernetx>=2.0.4

## Usage
Usage examples are provided in fac_graphs_examples and energy_jax_examples

## Documentation

