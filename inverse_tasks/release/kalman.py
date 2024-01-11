import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def kalman_filter(x, dt):
    s1 = 0.1
    s2 = 0.9
    res = [0] * len(x)
    res[0] = kalman_step(x[0], 0, s1, s2)
    for i in range(1, len(x)):
        res[i] = kalman_step(x[i], dt, res[i - 1])
    return res


def kalman_step(x_step, dt, prev_step, s1=0.1, s2=0.9):
    x_model = prev_step + 2 * dt
    result = s1 ** 2 / (s1 ** 2 + s2 ** 2) * x_step + s2 ** 2 / (s1 ** 2 + s2 ** 2) * x_model
    return result


def make_noisy(data, probability=0.):
    outer_value = max(data)
    sample = deepcopy(data)
    noise = np.array([(np.random.rand() > (1 - probability)) * np.random.random() * outer_value for _ in range(len(data))])
    sample += noise
    return sample


def report_kalman(prob_outer=0.1):
    data = np.load("var2_s1_0x01_s2_5_v_2.npy")
    t, x = data

    # print(t[1]-t[0])
    # dt=0.005

    signal = x
    clean_result = kalman_filter(signal, 0.005)
    signal_noised = make_noisy(signal, probability=prob_outer)
    noisy_result = kalman_filter(signal_noised, 0.005)

    plt.plot(t, signal_noised, linewidth=2, label='Noised data')
    plt.plot(t, signal, "--", linewidth=1, label='Given data')
    plt.title("Sample")
    plt.text(0, 8, "Outer prob. = " + str(prob_outer), bbox=dict(boxstyle="round",
                                                                 ec=(1., 0.5, 0.5),
                                                                 fc=(1., 0.8, 0.8),
                                                                 ))
    plt.legend()
    plt.show()
    plt.scatter(t, signal_noised, label='Given data', color="black", s=5)
    plt.scatter(t, signal, label='Given data', color="orange", s=8)
    plt.plot(t, clean_result, label="Filter result", linewidth=1, color="brown")
    plt.plot(t, noisy_result, label="Noisy result", linewidth=1, color="green")
    plt.title("Filtering result")
    plt.legend()
    plt.show()
    plt.plot(t, clean_result, label="Filter result", linewidth=1, color="brown")
    plt.plot(t, noisy_result, label="Noisy result", linewidth=1, color="green")
    plt.text(0, 8, "Outer prob. = " + str(prob_outer), bbox=dict(boxstyle="round",
                                                                 ec=(1., 0.5, 0.5),
                                                                 fc=(1., 0.8, 0.8),
                                                                 ))
    plt.title("Filtering result only functions")
    plt.legend()
    plt.show()


# report_kalman(prob_outer=0.1)
# report_kalman(prob_outer=0.05)
# report_kalman(prob_outer=0.01)
