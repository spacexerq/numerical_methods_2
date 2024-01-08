import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq, fft, ifft


def func1(t):
    return pow(np.cos(200 * np.pi * t), 3) + np.cos(10 * np.pi * t + np.pi / 4)


def func2(t):
    return pow(t, 2)


def der_func1(t):
    return -10 * np.pi * (
            60 * pow(np.cos(200 * np.pi * t), 2) * np.sin(200 * np.pi * t) + np.sin(np.pi / 4 + 10 * np.pi * t))


def der_func2(t):
    return 2 * t


def make_noisy(f_l, t_l, sigma_l=2e-1, mu_l=0):
    f_l += np.random.normal(mu_l, sigma_l, size=f_l.shape)
    return f_l


def finite_diff(local_f: callable, t_loc, dt_loc=1e-5, noisy=True, sigma_l=1e-2):
    derivative = np.empty_like(t_loc)
    if noisy:
        for i in range(len(t_loc)):
            derivative[i] = (make_noisy(local_f(t_loc[i] + dt_loc), t_loc[i], sigma_l=sigma_l) - make_noisy(
                local_f(t_loc[i] - dt_loc), t_loc[i], sigma_l=sigma_l)) / (2 * dt_loc)
    else:
        for i in range(len(t_loc)):
            derivative[i] = (local_f(t_loc[i] + dt_loc) - local_f(t_loc[i] - dt_loc)) / (2 * dt_loc)
    return derivative, t_loc


def report_finite_diff(func_type=1, sigma=1e-2, noisy_val=False):
    a = 0
    b = 0.02
    dt = 1e-5
    N = int((b - a) / dt)
    time = np.linspace(a, b, N)
    if func_type == 1:
        der, t1 = finite_diff(func1, time, dt_loc=dt, noisy=noisy_val, sigma_l=sigma)
        exact_der = der_func1(time)
    else:
        der, t1 = finite_diff(func2, time, dt_loc=dt, noisy=noisy_val, sigma_l=sigma)
        exact_der = der_func2(time)
    plt.plot(time, exact_der, label="Exact solution")
    plt.plot(t1, der, "--", label="FD solution")
    plt.legend()
    plt.text(0, -300, "dt=" + str(dt), bbox=dict(boxstyle="round",
                                                 ec=(1., 0.5, 0.5),
                                                 fc=(1., 0.8, 0.8),
                                                 ))
    if noisy_val:
        plt.title("FD with noise")
        plt.text(0, 0, "sigma=" + str(sigma), bbox=dict(boxstyle="round",
                                                        ec=(1., 0.5, 0.5),
                                                        fc=(1., 0.8, 0.8),
                                                        ))
    else:
        plt.title("FD no noise")
    plt.show()


def rect(t, border):
    res = np.ones_like(t)
    for i in range(len(res)):
        if abs(t[i]) > border:
            res[i] = 0.
    return res


def fourier_derivation(f: callable, t_loc, dt_loc, rect_border=3000, noisy=False, sigma_l=1e-2):
    f_t = f(t_loc)
    if noisy:
        f_t = make_noisy(f_t, t_loc, sigma_l=sigma_l)
    F = fft(f_t)
    omega = 2 * np.pi * fftfreq(f_t.shape[0], d=dt_loc)
    # plt.plot(omega)
    # plt.show()
    m = rect(omega, rect_border)
    F_res = 1j * omega * F * m
    derivative = np.real(ifft(F_res))
    return derivative, t_loc


def report_fourier(func_type=1, noisy_val=False, sigma=1e-2):
    a = 0
    b = 0.02
    dt = 1e-5
    N = int((b - a) / dt)
    time = np.linspace(a, b, N)
    if func_type == 1:
        der, t1 = fourier_derivation(func1, time, dt, noisy=noisy_val, sigma_l=sigma)
        exact_der = der_func1(time)
    else:
        der, t1 = fourier_derivation(func2, time, dt, rect_border=1000, noisy=noisy_val, sigma_l=sigma)
        exact_der = der_func2(time)
    plt.plot(time, exact_der, label="Exact solution")
    plt.plot(t1, der, "--", label="Fourier solution")
    plt.legend()
    if noisy_val:
        plt.text(0, 0, "sigma=" + str(sigma), bbox=dict(boxstyle="round",
                                                           ec=(1., 0.5, 0.5),
                                                           fc=(1., 0.8, 0.8),
                                                           ))
    plt.title("Fourier derivation")
    plt.show()


def report_least_squares(sigma=1e-2):
    a = 0
    b = 0.1
    dt = 1e-5
    N = int((b - a) / dt)
    time = np.linspace(a, b, N)
    f2 = make_noisy(func2(time), time, sigma_l=sigma)
    poly = np.polyfit(time, f2, deg=2)
    exact_der = der_func2(time)
    plt.plot(time, exact_der, label="Exact solution")
    plt.plot(time, 2*time*poly[0]+poly[1], "--", label="MNK solution")
    plt.text(0, 0, "sigma=" + str(sigma), bbox=dict(boxstyle="round",
                                                       ec=(1., 0.5, 0.5),
                                                       fc=(1., 0.8, 0.8),
                                                       ))
    plt.legend()
    plt.title("МНК")
    plt.show()


report_finite_diff(func_type=2, noisy_val=False, sigma=1e-2)
# report_finite_diff(func_type=2, noisy_val=True, sigma=1e-3)
# report_finite_diff(func_type=2, noisy_val=True, sigma=1e-2)
# report_fourier(func_type=1, noisy_val=True, sigma=1e-2)
report_fourier(func_type=2, noisy_val=False, sigma=1e-2)
# report_fourier(func_type=2, noisy_val=True, sigma=1e-2)
# report_least_squares(sigma=1)
# report_least_squares(sigma=1e-1)
# report_least_squares(sigma=1e-2)
