import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft, ifft
from copy import deepcopy

data = np.load("data_v2.npy")
data = data/max(data)
# Расположу всю выборку между -1 и 1


def coef_legendre(n, x):
    polynom = sp.special.legendre(n)
    sum_res = 0
    for x_l in x:
        sum_res += polynom(x_l)
    return sum_res


def a(x, k=24):
    a = np.zeros((k,))
    length = len(x)
    for i in range(k):
        coef = (2 * i + 1) / (2 * length)
        a[i] = coef * coef_legendre(i, x)
    return a


a = a(data, k=10)

x = np.linspace(-1, 1, 200)
f = np.empty_like(x)

for i in range(len(x)):
    result = 0
    for k in range(len(a)):
        polynom = sp.special.legendre(k)
        var = float(a[k]*polynom(x[i]))
        result += var
    # print(res)
    f[i] += result

plt.plot(x, f)
plt.hist(data, bins=100, density=True)
plt.title("Плотность по всей выборке")
plt.show()
#
# small_data = data[:2000]
# np.random.shuffle(small_data)
# a_small, b_small = a_b(small_data, k=24)
#
# x_small = np.linspace(min(small_data), max(small_data), 100)
# f_small = np.ones_like(x_small) * a_small[0] / np.pi / 2
#
# for k in range(1, len(a_small)):
#     f_small += 1 / np.pi * (a_small[k] * np.cos(k * x_small) + b_small[k] * np.sin(k * x_small))
#
# plt.title("Плотность без регуляризации")
#
# plt.plot(x_small, f_small)
# res = plt.hist(small_data, bins=100, density=True)
# plt.show()
#
#
# def a_b_reg(x, k=24, alp=1e-1):
#     a, b = np.zeros((k,)), np.zeros((k,))
#     for i in range(k):
#         a[i] = np.sum(np.cos(i * x)) / (1 + alp * i ** 2.1)
#         b[i] = np.sum(np.sin(i * x)) / (1 + alp * i ** 2.1)
#     return a / len(x), b / len(x)
#
#
# small_data = data[:800]
# np.random.shuffle(small_data)
# a_small, b_small = a_b_reg(small_data, k=24, alp=1e-2)
#
# x_small = np.linspace(min(small_data), max(small_data), 100)
# f_small_reg = np.ones_like(x_small) * a_small[0] / np.pi / 2
#
# for k in range(1, len(a_small)):
#     f_small_reg += 1 / np.pi * (a_small[k] * np.cos(k * x_small) + b_small[k] * np.sin(k * x_small))
#
# plt.title("Плотность с регуляризацией")
#
# plt.plot(x_small, f_small_reg)
# res = plt.hist(small_data, bins=100, density=True)
# plt.show()
#
# # plt.figure(figsize=(15, 8))
# plt.plot(x_small, f_small, label='плотность по малой выборке без регуляризации')
# plt.plot(x_small, f_small_reg, label='плотность по малой выборке с регуляризацией')
# plt.plot(x, f, label='плотность по всей')
# plt.legend(loc=2)
# plt.show()
