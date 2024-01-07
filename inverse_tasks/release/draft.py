import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft, ifft
# variant 8

def f1(t: np.ndarray): return pow(np.cos(200 * np.pi * t), 3) + np.cos(10 * np.pi * t + np.pi / 4)


def f2(t: np.ndarray): return np.power(t, 4) - 5 * np.power(t, 2)


def rect(t: np.ndarray, a: float = 1) -> np.ndarray:
    """функция прямоугольника

    Args:
        t (np.ndarray): значения, для которых считать функцию
        a (float, optional): граница, после которой функцию будет давать ноль. Defaults to 1.

    Returns:
        np.ndarray: значение функции на t
    """

    res = np.ones_like(t)
    for i in range(len(res)):
        if abs(t[i]) > a:
            res[i] = 0.
    return res


def method_2(f, dt: float = 1e-5, t_min: float = 0., t_max: float = 0.001):
    t = np.linspace(t_min, t_max, int((t_max - t_min) / dt), endpoint=True)
    f_t = f(t)
    # делаем преобразование фурье
    F = fft(f_t)
    # находим частоты, на которых посчитано F
    omega = 2 * np.pi * fftfreq(f_t.shape[0], d=t[1] - t[0])
    # считаем для частот функцию прямоугольника (значение границ подобрано ана глаз)
    m = rect(omega, 2 * np.pi * 120)
    m = 2 * np.pi * 120
    # производная в к-пространстве -- это домножение на 1j*omega, прямоугольник для того, чтоб погасить шум на бесконечности
    F_res = 1j * omega * F * m
    return t, np.real(ifft(F_res))


sigma = 5e-1

t_res, res = method_2(lambda t: f1(t) + 0, 1e-5, 0, 0.05)

plt.plot(t_res[1:-1], res[1:-1], label="method_2")
plt.plot(t_res, -10 * np.pi * (60 * pow(np.cos(200 * np.pi * t_res), 2) * np.sin(200 * np.pi * t_res) + np.sin(np.pi / 4 + 10 * np.pi * t_res))
, '--',
         label="real")
plt.title(f"Применение для первой функции с шумом ($\sigma$={sigma}) метода с фурье")
plt.show()
