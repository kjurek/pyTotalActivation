import numpy as np


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
