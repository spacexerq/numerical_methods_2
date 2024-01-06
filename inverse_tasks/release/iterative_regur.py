import numpy as np
import matplotlib.pyplot as plt
from copy import *

A = 0.5
delta_t = 1e-3
mu = 8 * 1e5
f0 = 300
beta = 3 * 1e3
a = 0
b = 0.005
N = 1000
h = (b - a) / N
t = np.linspace(a, b, N)


def core(t_l, tau):
    result = np.exp(-beta * t_l) + A * np.exp(-beta * t_l - delta_t)
    # result = np.exp(-beta * t_l)
    return result


def x_var(var):
    result = np.cos(2 * np.pi * f0 * var + mu * var * var)
    return result


def exact_solution(t_l, h_l):
    x_exact = x_var(t_l)
    core_loc = core(t_l, 0)
    # plt.plot(t_l, x_exact, label="Exact x")
    # plt.show()
    f_sol = np.convolve(core_loc, x_exact)
    n_c = len(f_sol)
    b_c = h_l * n_c
    t_for_convolve = np.linspace(a, b_c, n_c)
    # plt.plot(t_for_convolve, f_sol)
    # plt.show()
    return f_sol, t_for_convolve


def make_noisy(f_l, t_l, sigma_l=2e-1, mu_l=0):
    f_l += np.random.normal(mu_l, sigma_l, size=f_l.shape)
    return f_l


def iteration_step(x_prev_step, f, k, alpha=1.0):
    fd = f[:k.size]
    x_step = x_prev_step + alpha * (fd - np.convolve(k, x_prev_step)[:k.size])
    return x_step


def solve_iterative(f_noise, core_loc, alpha=1.0, m_max=100, eps=1e-2):
    x_0 = np.zeros_like(f_noise[:core_loc.size])
    x_current = deepcopy(x_0)
    x_new = iteration_step(x_current, f_noise, core_loc, alpha)
    xs = [x_new]
    m = 0
    for i in range(m_max):
        x_current = deepcopy(x_new)
        x_new = iteration_step(x_current, f_noise, core_loc, alpha)
        xs.append(x_new)
        if np.linalg.norm(x_current - x_new) / np.linalg.norm(x_new) <= eps:
            m = i
            return x_new, xs, m
    return x_new, xs, m


def iterative_solver(num_iterations=1e2, epsilon=1e-2, plot_default=True):
    x_exact_plt = x_var(t)
    if plot_default:
        plt.plot(t, x_exact_plt, label="Exact x")
        plt.legend()
        plt.show()

    # noisy function
    f1, t1 = exact_solution(t, h)
    f1_n = make_noisy(f1, t1, sigma_l=0.05)
    if plot_default:
        plt.plot(t1, f1_n)
        plt.title("Noisy function")
        plt.show()
    x_solution, xs, m = solve_iterative(f1_n, core(t, 0), alpha=0.01, m_max=int(num_iterations), eps=epsilon)

    plt.plot(t, x_var(t))
    plt.plot(t, x_solution[:t.shape[0]], '--')
    plt.title("Iterative regularization result")
    plt.show()

    # calculation relative error
    rel_error = [np.linalg.norm(x_i - x_exact_plt) / np.linalg.norm(x_exact_plt) for x_i in xs]
    plt.semilogy()
    plt.plot(rel_error)
    plt.text(0.1, 1e-1, "epsilon =" + str(epsilon), bbox=dict(boxstyle="round",
                                                              ec=(1., 0.5, 0.5),
                                                              fc=(1., 0.8, 0.8),
                                                              ))
    plt.xlabel("Number of steps")
    plt.title("Relative error")
    plt.show()


# iterative_solver(num_iterations=1e3, epsilon=1e-2, plot_default=True)
