import numpy as np
from scipy import signal

from TotalActivation.filters.cons import cons


def bold_parameters():
    eps = 0.54
    ts = 1.54
    tf = 2.46
    t0 = 0.98
    alpha = 0.33
    E0 = 0.34
    V0 = 1
    k1 = 7 * E0
    k2 = 2
    k3 = 2 * E0 - 0.2
    c = (1 + (1 - E0) * np.log(1 - E0) / E0) / t0

    a1 = -1 / t0
    a2 = -1 / (alpha * t0)
    a3 = -(1 + 1j * np.sqrt(4 * np.power(ts, 2) / tf - 1)) / (2 * ts)
    a4 = -(1 - 1j * np.sqrt(4 * np.power(ts, 2) / tf - 1)) / (2 * ts)

    psi = -((k1 + k2) * ((1 - alpha) / alpha / t0 - c / alpha) - (k3 - k2) / t0) / (-(k1 + k2) * c * t0 - k3 + k2)
    a = np.array([a1, a2, a3, a4])

    return a, psi


def spmhrf_parameters():
    a1 = -0.27
    a2 = -0.27
    a3 = -0.4347 - 1j * 0.3497
    a4 = -0.4347 + 1j * 0.3497

    a = np.array([a1, a2, a3, a4])
    psi = -0.1336

    return a, psi


def calculate_common_part(a, psi, t_r):
    fil_poles = np.array([psi * t_r])
    hden = cons(fil_poles)
    causal = np.array([x for x in fil_poles if np.real(x) < 0])
    n_causal = np.array([x for x in fil_poles if np.real(x) > 0])
    h_dc = cons(causal)
    h_dnc = cons(n_causal)
    reconstruct = {'num': cons(a * t_r), 'den': np.array([h_dc, h_dnc])}
    return reconstruct, hden


def spike_filter(a, psi, t_r):
    reconstruct, hden = calculate_common_part(a, psi, t_r)
    hnum = cons(a * t_r)
    _, h = signal.freqz(hnum, hden, 1024)
    maxeig = np.max(np.power(np.abs(h), 2))
    return reconstruct, reconstruct, maxeig


def block_filter(a, psi, t_r):
    reconstruct, hden = calculate_common_part(a, psi, t_r)
    hnum = cons(np.append(a * t_r, 0))
    _, h = signal.freqz(hnum, hden, 1024)
    maxeig = np.max(np.power(np.abs(h), 2))
    return {'num': hnum, 'den': reconstruct['den']}, reconstruct, maxeig
