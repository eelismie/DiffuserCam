import numpy as np


def autocorr2d(vals, pad_mode="constant"):
    """
    Compute 2-D autocorrelation of image via the FFT.

    Parameters
    ----------
    vals : py:class:`~numpy.ndarray`
        2-D image.
    pad_mode : str
        Desired padding. See NumPy documentation: https://numpy.org/doc/stable/reference/generated/numpy.pad.html

    Return
    ------
    autocorr : py:class:`~numpy.ndarray`
    """

    n, m = vals.shape
    padded_vals = np.pad(vals, ((0, n), (0, m)), mode=pad_mode)
    f_vals = np.fft.fft2(padded_vals)
    f_autocorr = (f_vals * np.conj(f_vals))
    auto = np.fft.fftshift(np.fft.ifft2(f_autocorr))

    return auto[n//2: n//2 + n, m//2: m//2 + m].real #[n//2: n//2 + n, m//2: m//2 + m].real #[n: n+n, m: m+m].real #[n//2: n//2 + n, m//2: m//2 + m].real
