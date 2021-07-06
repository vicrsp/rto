# Real-time Optimization

Real-time optimization is a well known process optimization technique. It uses a process model, usually in the form of differential and/or algebraic equations, along with a numerical optimization algorithm that calculates optimal setpoints according to given cost function and constraints.

This repository provides easy to extend (I hope!) tools for building and comparing RTO systems. It tries to leverage numpy and scipy capabilities, while also using a simple local SQLite database for storing experiment results.

## Installation
Just make these packages are installed: `numpy`, `pandas`, `matplotlib`, `scipy`, `bunch`, `sklearn` and `seaborn`.

## Package Organization

### RTO
The main RTO routine is located under the `src/rto/optimization` module, in the `rto.py` file, to be more specific. 

It implements a class RTO that receives all the necessary information about the system being optimized, such as the approximate and real process models, and the adaptation strategy.

The adaptation strategy can be any implementation of the methods available in the literature, respecting a pre-defined structure, of course.

### Adaptation Strategy
Adaptation strategies are located in the `src/rto/adaptation` module. The `ma_gaussian_processes.py` file should be used as example. One adaptation strategy needs to implement these methods:

1. adapt(u_k, data): receives the setpoints calculated at the k-$th$ iteration and measured data required for adaptation. 
2. get_modifiers(u_k): receives a setpoint and returns the cost and constraint modifiers.

*There is still some work necessary to better generalize this. It is currently built to work with modifier adaptation schemes*

### Model-Based Optimization 

Model-based Optimization routines are located in the `src/rto/optimization` module. The `optimizer.py` file should be used as example. The only requirement is to implement an `optimize` function that receives the approximate model, the adaptation strategy and general optimizer options, such as variable bounds and the starting point.

### Process Models
Process models are located in the `src/rto/models` module. There is currently implementations for the Williams-Otto CSTR and the acetoacetylation of pyrrole with dyketene reaction semi-batch system.

### Experimental Data

Utility functions for managing experiment results are under the `src/rto/experiment` module. 

## CCTA 2021
[You can find the notebook and scripts in the rto-examples repository](https://github.com/vicrsp/rto-examples)