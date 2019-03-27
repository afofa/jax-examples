from typing import Callable, Tuple

from jax import grad
import jax.numpy as np
import numpy as onp

def gradient_descent_step(f:Callable[[float], float], x:float, lr:float) -> Tuple[float, Tuple[float, float]]:
    # gradient function
    df = grad(f)
    # f evaluated at x
    fx = f(x)
    # gradient of f evaluated at x
    dfx = df(x)

    # new x
    x_new = x - lr*dfx

    return x_new, (fx, dfx)

def gradient_descent(f:Callable[[float], float], x:float, lr:float, it:int):
    xs = onp.zeros((it+1))
    fxs = onp.zeros((it))
    dfxs = onp.zeros((it))

    xs[0] = x
    for i in range(it):
        x, (fx, dfx) = gradient_descent_step(f, x, lr)
        xs[i+1] = x
        fxs[i] = fx
        dfxs[i] = dfx

    return xs, fxs, dfxs