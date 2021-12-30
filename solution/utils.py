
import numpy as np
from scipy import fft
from scipy.fftpack import next_fast_len
from pycsou.core.linop import LinearOperator
from scipy.fft import dctn, idctn
from pycsou.opt.proxalgs import AcceleratedProximalGradientDescent as APGD
from pycsou.opt.proxalgs import PDS
from diffcam.plot import plot_image
import matplotlib.pyplot as plt
import pathlib as plib

# for disp and save arguments
# apgd with intermediate plots and saves

class APGD_(APGD):
    def __init__(self, gamma, datashape, no_plot, save, *args, **kwargs):
        self.gamma = gamma
        self.datashape = datashape
        self.no_plot = no_plot
        self.ax = None
        self.save = save
        super().__init__(*args, **kwargs)

    def print_diagnostics(self):
        print(dict(self.diagnostics.loc[self.iter]))
        if not self.no_plot:
            self.ax = plot_image(self.iterand['iterand'].reshape(self.datashape), gamma=self.gamma, ax=self.ax)
            self.ax.set_title(dict(self.diagnostics.loc[self.iter]))
            plt.draw()
            plt.pause(0.2)
        if self.save:
            plt.savefig(plib.Path(self.save) / f"{self.iter}.png")
            print(f"Files saved to : {self.save}")


#PDS with intermediate plots and saves 
class PDS_(PDS):
    def __init__(self, gamma, datashape, no_plot, save, *args, **kwargs):
        self.gamma = gamma
        self.datashape = datashape
        self.no_plot = no_plot
        self.ax = None
        self.save = save
        super().__init__(*args, **kwargs)

    def print_diagnostics(self):
        print(dict(self.diagnostics.loc[self.iter]))
        if not self.no_plot:
            self.ax = plot_image(self.iterand['iterand'].reshape(self.datashape), gamma=self.gamma, ax=self.ax)
            self.ax.set_title(dict(self.diagnostics.loc[self.iter]))
            plt.draw()
            plt.pause(0.2)
        if self.save:
            plt.savefig(plib.Path(self.save) / f"{self.iter}.png")
            print(f"Files saved to : {self.save}")

class Convolve2DRGB(LinearOperator):

    """
    Linear operator for 2D convolution with psf, generalized for 3 channel images. Strong inspiration 
    taken from diffcam/recon.py and diffcam/admm.py files. 
    """

    def __init__(self, size, psf, dtype=np.float32):

        super(Convolve2DRGB, self).__init__(shape=(size, size))
        
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