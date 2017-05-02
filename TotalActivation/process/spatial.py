import numpy as np
from numpy.fft import fftn, ifftn


def tikhonov(d1, d2, masker, mode='3D', iter=10, l=1, mu=0.01):
    """
    Tikhonov spatial regularization.

    :param d1:
    :param d2:
    :param mode:
    :param iter:
    :param l:
    :param mu:
    :return: Regularized time series
    """

    data1 = masker.inverse_transform(d1)
    data2 = masker.inverse_transform(d2)
    d1 = data1.get_data()
    d2 = data2.get_data()

    if mode == "2D":
        h = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        H = np.fft.fft2(h, d1.shape[0:2])
        H = H * np.conj(H)

        k = 0
        while k < iter:
            d1 = (1 - mu) * d1 + mu * d2 - mu * l * np.fft.ifft2(
                H[:, :, np.newaxis, np.newaxis] * np.fft.fft2(d1, axes=(0, 1)))
            k += 1

    elif mode == "3D":
        h = np.zeros((3, 3, 3))
        h[:, :, 0] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        h[:, :, 1] = np.array([[0, 1, 0], [1, -6, 1], [0, 1, 0]])
        h[:, :, 2] = h[:, :, 0]

        H = np.fft.fftn(h, d1.shape[0:3])
        H = H * np.conj(H)

        k = 0
        while k < iter:
            d1 = (1 - mu) * d1 + mu * d2 - mu * l * np.fft.ifftn(
                H[:, :, :, np.newaxis] * np.fft.fftn(d1, axes=(0, 1, 2)))
            k += 1

    else:
        print("Mode has to be either 2D or 3D")

    d1 = np.real(d1)

    return masker.transform(data1)


def get_h(dim):
    h = np.zeros((3, 3, 3))
    h[:, :, 0] = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    h[:, :, 1] = np.array([[0, 1, 0], [1, -6, 1], [0, 1, 0]])
    h[:, :, 2] = h[:, :, 0]
    return h, fftn(h, dim)


def strspr_mat(y, atlas, Lambda, Nit, IND, data_shape):
    assert len(data_shape) == 4
    h, _ = get_h(data_shape[0:3])
    y_vol = np.zeros(data_shape)

    for j in range(IND.shape[0]):
        y_vol[IND[j][0], IND[j][1], IND[j][2], :] = y[:, j]

    max_eig = 144.0
    z = np.zeros(y_vol.shape, dtype=complex)
    x_vol = np.zeros(data_shape, dtype=complex)
    for i in range(data_shape[3]):
        for k in range(1, Nit):
            z0 = z[:, :, :, i]
            z1 = 1.0 / (Lambda * max_eig)
            z2 = atlas_filter(y_vol[:, :, :, i], h)
            z3 = atlas_filter_cconj(z[:, :, :, i], h)
            c = clip(z0 + z1 * z2 - z3 / max_eig, atlas)
            z[:, :, :, i] = c
        x_vol[:, :, :, i] = y_vol[:, :, :, i] - Lambda * atlas_filter_conj(z[:, :, :, i], h)

    x_out = np.zeros(y.shape, dtype=complex)
    for i in range(IND.shape[0]):
        x_out[:, i] = x_vol[IND[i, 0], IND[i, 1], IND[i, 2], :]

    return x_out


def strspr(y, atlas, Lambda, Nit, IND, data_shape):
    assert len(data_shape) == 4
    _, H = get_h(data_shape[0:3])

    y_vol = np.zeros(data_shape)
    for j in range(IND.shape[0]):
        y_vol[IND[j][0], IND[j][1], IND[j][2], :] = y[:, j]

    max_eig = 144.0
    z = np.zeros(y_vol.shape, dtype=complex)
    x_vol = np.zeros(data_shape, dtype=complex)
    for i in range(data_shape[3]):
        for k in range(1, Nit):
            z0 = z[:, :, :, i]
            z1 = 1.0 / (Lambda * max_eig)
            z2 = ifftn(H * fftn(y_vol[:, :, :, i], data_shape[0:3]))
            z3 = ifftn(H * np.conj(H) * fftn(z[:, :, :, i], data_shape[0:3]))
            z[:, :, :, i] = clip(z0 + z1 * z2 - z3 / max_eig, atlas)
        x_vol[:, :, :, i] = y_vol[:, :, :, i] - Lambda * ifftn(np.conj(H) * fftn(z[:, :, :, i], data_shape[0:3]))

    x_out = np.zeros(y.shape, dtype=complex)
    for i in range(IND.shape[0]):
        x_out[:, i] = x_vol[IND[i, 0], IND[i, 1], IND[i, 2], :]

    return x_out


def clip(in_, atlas):
    out = np.zeros(in_.shape, dtype=complex)
    for a in np.arange(1, np.max(atlas) + 1):
        ind = atlas == a
        norm = np.linalg.norm(in_[ind], 2)
        if norm > 1:
            out[ind] = in_[ind] / norm
        else:
            out[ind] = in_[ind]
    return out


def atlas_filter(in_, h):
    atlas_out = np.zeros(in_.shape, dtype=complex)
    r1 = in_[0:3, 0:10, 0:10]
    r2 = in_[3:10, 0:3, 0:10]
    r3 = in_[3:10, 3:10, 0:5]
    r4 = in_[3:10, 3:10, 5:10]
    atlas_out[0:3, 0:10, 0:10] = ifftn(fftn(h, r1.shape) * fftn(r1))
    atlas_out[3:10, 0:3, 0:10] = ifftn(fftn(h, r2.shape) * fftn(r2))
    atlas_out[3:10, 3:10, 0:5] = ifftn(fftn(h, r3.shape) * fftn(r3))
    atlas_out[3:10, 3:10, 5:10] = ifftn(fftn(h, r4.shape) * fftn(r4))
    return atlas_out


def atlas_filter_conj(in_, h):
    atlas_out = np.zeros(in_.shape, dtype=complex)
    r1 = in_[0:3, 0:10, 0:10]
    r2 = in_[3:10, 0:3, 0:10]
    r3 = in_[3:10, 3:10, 0:5]
    r4 = in_[3:10, 3:10, 5:10]
    atlas_out[0:3, 0:10, 0:10] = ifftn(np.conj(fftn(h, r1.shape)) * fftn(r1))
    atlas_out[3:10, 0:3, 0:10] = ifftn(np.conj(fftn(h, r2.shape)) * fftn(r2))
    atlas_out[3:10, 3:10, 0:5] = ifftn(np.conj(fftn(h, r3.shape)) * fftn(r3))
    atlas_out[3:10, 3:10, 5:10] = ifftn(np.conj(fftn(h, r4.shape)) * fftn(r4))
    return atlas_out


def atlas_filter_cconj(in_, h):
    atlas_out = np.zeros(in_.shape, dtype=complex)
    r1 = in_[0:3, 0:10, 0:10]
    r2 = in_[3:10, 0:3, 0:10]
    r3 = in_[3:10, 3:10, 0:5]
    r4 = in_[3:10, 3:10, 5:10]
    atlas_out[0:3, 0:10, 0:10] = ifftn(fftn(h, r1.shape) * np.conj(fftn(h, r1.shape)) * fftn(r1))
    atlas_out[3:10, 0:3, 0:10] = ifftn(fftn(h, r2.shape) * np.conj(fftn(h, r2.shape)) * fftn(r2))
    atlas_out[3:10, 3:10, 0:5] = ifftn(fftn(h, r3.shape) * np.conj(fftn(h, r3.shape)) * fftn(r3))
    atlas_out[3:10, 3:10, 5:10] = ifftn(fftn(h, r4.shape) * np.conj(fftn(h, r4.shape)) * fftn(r4))
    return atlas_out
