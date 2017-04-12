import scipy.io as sio
import numpy as np
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

class DetrendType(object):
    DCT, NORMALIZE = range(2)


class Config(object):
    detrend = DetrendType.NORMALIZE


class TotalActivationTool(object):
    def __init__(self):
        self.config = None
        self.voxels = None

    def load(self, data_path, atlas_path, config):
        self.config = config
        data = sio.loadmat(data_path)['data']
        atlas = sio.loadmat(atlas_path)['atlas']
        self.data_shape = data.shape
        self.dimension = len(data.shape)
        self.voxels = data[np.nonzero(atlas * np.ndarray.sum(data, axis=self.dimension - 1))]
        logging.debug('Dimension={}'.format(self.dimension))
        logging.debug('Voxels.shape={}'.format(self.voxels.shape))

    def detrend(self):
        if self.config.detrend == DetrendType.NORMALIZE:
            self.tcn = self.__detrend_normalize()
        elif self.config.detrend == DetrendType.DCT:
            self.tcn = self.__detrend_dct()

    def __detrend_normalize(self):
        tcn = np.zeros((self.voxels.shape[1], self.voxels.shape[0]))
        logging.debug('tcn.shape={}'.format(tcn.shape))
        for i in range(self.voxels.shape[0]):
            tcn[:, i] = self.voxels[i] / np.std(self.voxels[i])
        logging.debug('tcn.shape={}, tcn={}'.format(tcn.shape, tcn))

    def detrend_dct(self):
        pass

    def regularization(self):
        pass
