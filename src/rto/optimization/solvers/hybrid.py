from scipy.optimize import differential_evolution, minimize, OptimizeResult


def hybrid_minimize(func, x0, bounds, max_iter_init=50, atol_init=1e-2, **solver_params):
    """A hybrid algorith that is initialized with DE and polished with Nelder-Mead

    Args:
        func (_type_): _description_
        bounds (_type_): _description_
        max_iter_init (int, optional): _description_. Defaults to 50.
        atol_init (int, optional): _description_. Defaults to 1e-2.

    Returns:
        _type_: _description_
    """
    result_initialization = differential_evolution(
        func, bounds, maxiter=max_iter_init, polish=False, atol=atol_init, **solver_params)
    if result_initialization.success is True:
        result_polish = minimize(func, result_initialization.x, method='Nelder-Mead',
                                 bounds=bounds, options={'disp': False, 'fatol': 1e-6, 'maxiter': 1000})
    else:
        result_polish = minimize(func, x0, method='Nelder-Mead',
                                 bounds=bounds, options={'disp': False, 'fatol': 1e-6, 'maxiter': 1000})
    result_polish.nfev = result_polish.nfev + result_initialization.nfev
    return result_polish
