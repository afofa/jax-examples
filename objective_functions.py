import numpy as np

def obj_func1(x:float):
    return -x * np.exp(-x/5)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    from gradient_descent import *
    from jax import grad

    xs = np.linspace(-5, 25, 101)
    ys = list(map(obj_func1, xs))

    plt.plot(xs, ys)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Objective Function 1')
    plt.show()

    print(gradient_descent(obj_func1, 0, 3.0, 20))