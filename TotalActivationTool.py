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
        # Method_time: 'B', 'S' or 'W'
        # Method_space: 'S', 'T', None
        # HRF: 'bold', 'spmhrf'
        self.config = {'Method_time' = 'B',
                       'Method_space' = None,
                       'HRF' = 'bold',
                       'Detrend': True,
                       'Standardize' : True,
                       'Highpass' : 0.01,
                       'Lowpass' : None,
                       'TR' : 2,
                       'Lambda' : 1/0.8095}
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

    def load_nifti(self, d, a):
        '''
        Basic function to load NIFTI time-series and flatten them to 2D array

        Imputs:
        d : data (4D NIFTI file)
        a : atlas (3D NIFTI file)
        '''

        import nibabel as nib
        from nilearn.input_data import NiftiMasker

        self.masker = NiftiMasker(mask_strategy='epi',
                           memory="nilearn_cache", memory_level=2,
                           standardize=self.config['Standardize'],
                           detrend=self.config['Detrend'],
                           high_pass = self.config['Highpass'],
                           low_pass = self.config['Lowpass'],
                           t_r = self.config['TR'])

        self.masker.fit(data)
        self.data = self.masker.transform(data)
        self.data_shape = self.data.shape
        self.dimension = len(self.data.shape)
        self.voxels = self.data.shape[1]
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
