
import numpy as np
from scipy import fft
from scipy.fftpack import next_fast_len
from pycsou.core.linop import LinearOperator

class Convolve2DRGB(LinearOperator):

    """
    Linear operator for 2D convolution with psf. Strong inspiration 
    taken from diffcam/recon.py and diffcam/admm.py files. 
    """

    def __init__(self, size, psf, lipschitz_cst = 1000, dtype=np.float32):

        #this p
        self._is_rgb = True if len(psf.shape) == 3 else False
        if self._is_rgb:
            self._psf = psf
            self._n_channels = 3
        else:
            self._psf = psf[:, :, np.newaxis]
            self._n_channels = 1
        self._psf_shape = np.array(self._psf.shape)

        if dtype:
            self._psf = self._psf.astype(dtype)
            self._dtype = dtype
        else:
            self._dtype = self._psf.dtype
        if self._dtype == np.float32 or dtype == "float32":
            self._complex_dtype = np.complex64
        elif self._dtype == np.float64 or dtype == "float64":
            self._complex_dtype = np.complex128
        else:
            raise ValueError(f"Unsupported dtype : {self._dtype}")

        # cropping / padding indices
        self._padded_shape = 2 * self._psf_shape[:2] - 1
        self._padded_shape = np.array([next_fast_len(i) for i in self._padded_shape])
        self._padded_shape = np.r_[self._padded_shape, [self._n_channels]]
        self._start_idx = (self._padded_shape[:2] - self._psf_shape[:2]) // 2
        self._end_idx = self._start_idx + self._psf_shape[:2]

        # pre-compute operators / outputs
        self.reset()

        super(Convolve2DRGB, self).__init__(shape=(size, size), lipschitz_cst = lipschitz_cst)

    def reset(self):
        # spatial frequency response
        self._H = fft.rfft2(self._pad(self._psf), axes=(0, 1)).astype(self._complex_dtype)

    def __call__(self, x):

        """Convolution with frequency response."""

        x = x.reshape(self._psf.shape)
        x_padded = self._pad(x)

        out = fft.ifftshift(
                    fft.irfft2(
                        fft.rfft2(x_padded, axes=(0, 1)) * self._H,
                        axes=(0, 1),
                    ),
                    axes=(0, 1),
                )

        out = self._crop(out)
        out = out.ravel()

        return out

    def adjoint(self, x):

        """adjoint of forward / convolution"""

        x = x.reshape(self._psf.shape) #reshape x into an image 
        x_padded = self._pad(x) #zero pad image 

        out = fft.ifftshift(
            fft.irfft2(fft.rfft2(x_padded, axes=(0, 1)) * np.conj(self._H), axes=(0, 1)),
            axes=(0, 1),
        )

        out = self._crop(out)
        out = out.ravel()

        return out

    def _pad(self, v):

        """adjoint of cropping"""

        vpad = np.zeros(self._padded_shape).astype(v.dtype)
        vpad[self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]] = v
        return vpad  

    def _crop(self, x):
        """crop"""
    
        return x[self._start_idx[0] : self._end_idx[0], self._start_idx[1] : self._end_idx[1]]

class finiteDifferenceRGB(LinearOperator):

    def __init__(self, data_shape, lipschitz_cst):
        self.data_shape = data_shape
        super(finiteDifferenceRGB, self).__init__(shape=(data_shape[0], data_shape[1]), lipschitz_cst = lipschitz_cst)

    def __call__(self, x):
        """Gradient of image estimate, approximated by finite difference. Space where image is assumed sparse."""
        x_im = x.reshape(self.data_shape)
        out = np.stack(
            (np.roll(x_im, 1, axis=0) - x_im, np.roll(x_im, 1, axis=1) - x_im),
            axis=len(x_im.shape),
        )
        out = out.ravel()
        return out

    def adjoint(self, x):
        x = x.reshape(self.data_shape)
        diff1 = np.roll(x[..., 0], -1, axis=0) - x[..., 0]
        diff2 = np.roll(x[..., 1], -1, axis=1) - x[..., 1]
        out = diff1 + diff2
        out = out.ravel()
        return out