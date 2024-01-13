import numpy as np
from matplotlib import pyplot as plt


def solve_quad(k, f, h, x):
    N = len(f)
    x[0] = f[0]
    for i in range(1, N):
        s = np.sum(K[i, 1:i] * x[1:i])
        x[i] = (f[i] + h / 2 * k[i, 0] + h * s) / (1 - h / 2 * k[i, i])
    return x


def error(exact, solution):
    return np.linalg.norm((exact - solution) / solution)


def iter(y, h, x, n, k, f):
    yk = y.copy()
    for i in range(n):
        yk[i] = 0
        for j in range(i):
            yk[i] = yk[i] + 2 * k(x[i], x[j]) * y[j]
        yk[i] = yk[i] - k(x[i], x[0]) * y[0] - k(x[i], x[i]) * y[i]
        yk[i] = f(x[i]) + yk[i] * h / 2
    return yk


def solve_iter(k, f, x, h, eps=1e-1, max_iter=1000):
    n = len(x)
    y = f(x)
    yk = iter(y, h, x, n, k, f)
    i = 0
    err = np.linalg.norm(y - yk) / np.linalg.norm(y)

    while err > eps:
        y = yk.copy()
        yk = iter(y.copy(), h, x, n, k, f)
        err = error(y, yk)
        i += 1
        if i > max_iter:
            break
    return yk, i


def report_fredholm(f_loc, t_loc):
    x_sol_quad = solve_quad(K, f_loc, h, np.zeros_like(f_loc))
    x_ex = [2 * np.exp(t[i]) - 1 for i in range(N)]

    plt.plot(t_loc, x_ex, label="Exact")
    plt.plot(t_loc, x_sol_quad, label="Numerical")
    plt.legend()
    plt.title("Quadratic formulae")
    plt.show()

    for j in range(len(f_loc)):
        f_loc[j] += np.random.random(1)*10000
    x_sol_quad_noisy = solve_quad(K, f_loc, h, np.zeros_like(f_loc))

    plt.plot(t_loc, x_ex, label="Exact")
    plt.plot(t_loc, x_sol_quad_noisy, label="Numerical")
    plt.legend()
    plt.title("Quadratic formulae noisy")
    plt.show()

    res, iteration_n = solve_iter(k_func, f_func, t_loc, h, eps=1e-5)
    plt.plot(t, x_ex, label='Exact solution')
    plt.scatter(t, res, label=f"Numerical {iteration_n} iterations")
    plt.scatter(t, x_sol_quad, label="Quadratic solution")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.title("Iterative methods")
    plt.show()

    res, iteration_n = solve_iter(k_func, f_func_noisy, t_loc, h, eps=1e-5)
    plt.plot(t, x_ex, label='Exact solution')
    plt.plot(t, res, label=f"Numerical {iteration_n} iterations")
    # plt.scatter(t, x_sol_quad, label="Quadratic solution")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.title("Iterative methods for noisy function")
    plt.show()


k_func = lambda x, t: np.exp(l*(x-t))
f_func = lambda t: np.exp(beta*t)
f_func_noisy = lambda t: np.exp(beta*t)+np.random.random(1)*10000

a = 0
b = 10
h = 1e-1
l = -1
N = int((b - a) / h)
t = np.linspace(a, b, N)
K = np.empty((N, N))
beta = 1

for i in range(N):
    for j in range(i + 1):
        K[i, j] = np.exp(l * (t[i] - t[j]))

f = np.array([np.exp(beta * t[i]) for i in range(N)])
# report_fredholm(f, t)