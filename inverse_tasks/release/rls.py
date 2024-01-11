import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf


def adaptrls(x_loc, d_loc, p_loc, lmbd=1., alpha=1e-4):
    N_loc = len(x_loc)
    w = np.zeros(p_loc)
    R = np.eye(p_loc) * alpha
    # toeplitz matrix
    e = np.zeros(N_loc)
    for i in range(p_loc, N_loc):
        var_position = max(0, i - p_loc + 1)
        x_i = np.flip(x_loc[var_position:i + 1], axis=0)
        y_i = np.dot(w, x_i)
        e[i] = d_loc[i] - y_i
        k = np.dot(R, x_i) / (lmbd + np.dot(np.dot(x_i, R), x_i))
        w += k * e[i]
        R = (R - np.outer(k, np.dot(x_i, R))) / lmbd
        # tensor product with a result as matrix [u_i v_j]
    return w, e


def report_rls_diff_p(sigma=1., h_len=15, p=15, l_loc=1., show_error=False, show_signal=False, show_result=False):
    h = np.random.random((h_len,))
    x = np.random.random((1000,))
    d0 = signal.lfilter(h, 1, x)
    d = d0 + np.random.normal(0, sigma, len(x))
    if show_signal:
        plt.plot(d0[:200], linewidth=3, label="True signal")
        plt.plot(d[:200], linewidth=1, label="Signal with noise")
        plt.title("Signal form [0:200]")
        plt.text(0, 0, "sigma = " + str(sigma), bbox=dict(boxstyle="round",
                                                          ec=(1., 0.5, 0.5),
                                                          fc=(1., 0.8, 0.8),
                                                          ))
        plt.legend()
        plt.show()
    w, e = adaptrls(x, d, p, lmbd=l_loc)

    if show_error:
        plt.scatter(range(len(e)), e, label="Ошибка по шагам")
        plt.xlabel("t")
        plt.semilogy()
        plt.title("p=" + str(p) + " h=" + str(p))
        plt.legend()
        plt.show()

    w_min, e_min = adaptrls(x, d, p - 5, lmbd=l_loc)

    if show_error:
        plt.scatter(range(len(e_min)), e_min, label="Ошибка по шагам")
        plt.xlabel("t")
        plt.semilogy()
        plt.title("p=" + str(p) + " h=" + str(p - 5))
        plt.legend()
        plt.show()

    w_max, e_max = adaptrls(x, d, p + 5, lmbd=l_loc)
    if show_error:
        plt.scatter(range(len(e_max)), e_max, label="Ошибка по шагам")
        plt.xlabel("t")
        plt.semilogy()
        plt.title("p=" + str(p) + " h=" + str(p + 5))
        plt.legend()
        plt.show()

    if show_result:
        plt.plot(h, label="Given sample", linewidth=3, color='black')
        plt.plot(w, label="Model, p=" + str(p), color='red')
        plt.plot(w_min, label="Model, p=" + str(p - 5), color='green')
        plt.plot(w_max, label="Model, p=" + str(p + 5), color='blue')
        plt.title("Comparing filter coefficients for order " + str(p))
        plt.legend()
        plt.show()


def report_rls_diff_l(sigma=1., h_len=15, l_num=6, show_error=False, show_signal=False, show_result=False):
    p = h_len
    h = np.random.random((h_len,))
    x = np.random.random((1000,))
    d0 = signal.lfilter(h, 1, x)
    d = d0 + np.random.normal(0, sigma, len(x))
    if show_signal:
        plt.plot(d0[:200], linewidth=3, label="True signal")
        plt.plot(d[:200], linewidth=1, label="Signal with noise")
        plt.title("Signal form [0:200]")
        plt.text(0, 0, "sigma = " + str(sigma), bbox=dict(boxstyle="round",
                                                          ec=(1., 0.5, 0.5),
                                                          fc=(1., 0.8, 0.8),
                                                          ))
        plt.legend()
        plt.show()
    l_sample = np.linspace(0.9, 1.0, l_num)
    e_sample = []
    for l_loc in l_sample:
        w, e = adaptrls(x, d, p, lmbd=l_loc)
        e_sample.append(e)
        plt.scatter(range(len(e)), e, label="lambda = " + str(round(l_loc, 2)), s=2)
        plt.plot(np.poly1d(np.polyfit(range(len(e)), e, deg=0))(range(len(e))), linewidth=2.5)
    plt.legend()
    plt.semilogy()
    plt.show()


def report_memory_of_model(l_loc=0.98):
    p = 32
    sigma = 1e-3
    h1 = np.random.random((16,))
    h = np.concatenate([h1, h1])
    lmbd = l_loc
    x = np.random.random((1000,))
    d = [*signal.lfilter(h1, 1, x[:len(x) // 2]), *signal.lfilter(h1, 1, x[len(x) // 2:])] + np.random.normal(0, sigma,
                                                                                                              len(x))

    w, e = adaptrls(x, d, p, lmbd, 1e5)
    plt.scatter(range(len(e)), e, label="Ошибка по шагам")
    plt.xlabel("t")
    plt.yscale("log")
    plt.title(f"p={p}, $\sigma$={sigma}, $\lambda$={lmbd}")
    plt.ylim(1e-5, 1e1)
    plt.show()
    plt.plot(h, label='h')
    plt.plot(w, label='w')
    plt.legend()
    plt.show()


# report_rls_diff_p(sigma=1e-1, h_len=15, p=15, l_loc=0.5, show_error=True, show_signal=False, show_result=True)
# report_rls_diff_l(sigma=1e-1, h_len=15, show_error=True, show_signal=False, show_result=True)

# report_memory_of_model(l_loc=1)
# report_memory_of_model(l_loc=0.95)


def report_audio():
    filename = 'audio.wav'
    x, Fs = sf.read(filename, dtype='float32')
    plt.plot(range(len(x)), x)
    plt.title("Given audio")
    plt.show()
    noise = np.random.normal(0, 1, len(x))
    noise2 = signal.lfilter(np.array([1, -1]), 1, noise) # ФВЧ
    sound_noise = x + noise2
    plt.plot(range(len(sound_noise)), sound_noise)
    plt.title("Noised sound")
    plt.show()
    w, e = adaptrls(sound_noise, noise2, 15, lmbd=0.9, alpha=1e-4)
    plt.plot(range(len(e)), e)
    plt.title("Reconstructed noise")
    plt.show()
    plt.plot(range(1000, len(e)), e[1000:],linewidth=2, label="Model")
    plt.plot(range(len(x)), x, "--", linewidth=1, label="Given")
    plt.title("Reconstructed noise without first steps")
    plt.legend()
    plt.show()


# report_audio()
