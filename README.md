# Real-time Optimization

Real-time optimization is a well known process optimization technique. It uses a process model, usually in the form of differential and/or algebraic equations, along with a numerical optimization algorithm that calculates optimal setpoints according to given cost function and constraints.

This repository provides easy to extend (I hope!) tools for building and comparing RTO systems. It tries to leverage numpy and scipy capabilities, while also using a simple local SQLite database for storing experiment results.

## Installation
Just make these packages are installed: `numpy`, `pandas`, `matplotlib`, `scipy`, `seaborn` and `sqlite3`.

I have plans to make this a package, but it might still take quite some time.

## Package Organization

### RTO
The main RTO routine is located under the `src/optimization` module, in the `rto.py` file, to be more specific. 

It implements a class RTO that receives all the necessary information about the system being optimized, such as the approximate and real process models, and the adaptation strategy.

The adaptation strategy can be any implementation of the methods available in the literature, respecting a pre-defined structure, of course.

### Adaptation Strategy
Adaptation strategies are located in the `src/model/adaptation` module. The `ma_gaussian_processes.py` file should be used as example. One adapatation strategy needs to implement these methods:

1. adapt(u_k, data): receives the setpoints calculated at the k-$th$ iteration and measured data required for adaptation. 
2. get_modifiers(u_k): receives a setpoint and returns the cost and constraint modifiers.

*There is still some work necessary to better generalize this. It is currently built to work with modifier adaptation schemes*

### Model-Based Optimization 

Model-based Optimization routines are located in the `src/optimization` module. The `batch_profile_optimizer.py` file should be used as example. The only requirement is to implment and `optimize` that receives the approximate model, the adaptation strategy and general optimizer options, such as variable bounds and the starting point.

### Process Models
Process models are located in the `src/model/process` module. 

### Experimental Data

First, you will need to create a database. To do taht, just run the `src/create_database.py` script in the console:

```bash
python3 src/create_database.py -n mydb -f ~/data/rto 
```

## CCTA 2021
See the following notebook that was used to create the results displayed in the paper: `src/notebooks/MA_GaussianProcesses_CCTA_2021.ipynb`

If you you want to reproduce the results, create two databases using the script above and then run the file `src/magp_experiment.py`. Results might differ a bit due to the stochastic nature of the system.

Don't forget to adjust the file names in both the notebook and the script file. Don't hesitate to send me a message if you hav any troubles.

## References

Darby, M. L., Nikolaou, M., Jones, J., & Nicholson, D. (2011). RTO: An overview and assessment of current practice. *Journal of Process Control*, 21(6), 874-884.

A. Marchetti, B. Chachuat, and D. Bonvin (2009)  Modifier-adaptation methodology for real-time optimization *Industrial \& engineering chemistry research* vol. 48, no. 13, pp. 6022–6033, 2009

[3] de Avila Ferreira, T., Shukla, H. A., Faulwasser, T., Jones, C. N., & Bonvin, D. (2018, June). Real-time optimization of uncertain process systems via modifier adaptation and Gaussian processes. In *2018 European Control Conference (ECC)* (pp. 465-470). IEEE.