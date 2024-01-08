import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

data = np.load("data_v2.npy")
data = data/max(data)
# Расположу всю выборку между -1 и 1


def coef_legendre(n, x):
    polynom = sp.special.legendre(n)
    sum_res = 0
    for x_l in x:
        sum_res += polynom(x_l)
    return sum_res


def a_coef(x, k):
    a = np.zeros((k,))
    length = len(x)
    for i in range(k):
        coef = (2 * i + 1) / (2 * length)
        a[i] = coef * coef_legendre(i, x)
    return a


def a_coef_reg(x,k, alpha=2e-2):
    a = np.zeros((k,))
    length = len(x)
    for i in range(k):
        coef = (2 * i + 1) / (2 * length)
        a[i] = coef * coef_legendre(i, x) / (1 + alpha * i ** 2.1)
    return a


def report_whole_data_density():
    a = a_coef(data, 10)
    x = np.linspace(-1, 1, 200)
    f = [0]*len(x)
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


def report_slice_data_no_reg_dens():
    np.random.shuffle(data)
    small_data = data[:500]
    a_slice = a_coef(small_data, 10)
    x_slice = np.linspace(-1, 1, 100)
    f_slice = [0]*len(x_slice)

    for i in range(len(x_slice)):
        result = 0
        for k in range(len(a_slice)):
            polynom = sp.special.legendre(k)
            var = float(a_slice[k]*polynom(x_slice[i]))
            result += var
        f_slice[i] += result

    plt.title("Плотность без регуляризации")
    plt.plot(x_slice, f_slice)
    plt.hist(small_data, bins=100, density=True)
    plt.show()


def report_slice_data_with_reg_dens():
    np.random.shuffle(data)
    small_data = data[:500]
    a_slice = a_coef_reg(small_data, 10)
    x_slice = np.linspace(-1, 1, 100)
    f_slice_reg = [0]*len(x_slice)

    for i in range(len(x_slice)):
        result = 0
        for k in range(len(a_slice)):
            polynom = sp.special.legendre(k)
            var = float(a_slice[k]*polynom(x_slice[i]))
            result += var
        f_slice_reg[i] += result

    plt.title("Плотность с регуляризацией")
    plt.plot(x_slice, f_slice_reg)
    plt.hist(small_data, bins=100, density=True)
    plt.show()

# plt.plot(x_small, f_small, label='Плотность среза данных без регуляризации')
# plt.plot(x_small, f_small_reg, label='Плотность среза данных с регуляризацией')
# plt.plot(x, f, label='плотность по всей')
# plt.legend()
# plt.show()


# report_whole_data_density()
# report_slice_data_no_reg_dens()
# report_slice_data_with_reg_dens()
