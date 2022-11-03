import numpy
import numpy as np
import matplotlib.pyplot as plt
from lib.Pitching import Autcorr


def fourier_transform(signal):
    size = 2 ** numpy.ceil(numpy.log2(2 * len(signal) - 1)).astype('int')

    # Variance
    var = numpy.var(signal)

    # Normalized data
    ndata = signal - numpy.mean(signal)

    # Compute the FFT
    fft = numpy.fft.fft(ndata, size)

    # Get the power spectrum
    pwr = np.abs(fft) ** 2

    # Calculate the autocorrelation from inverse FFT of the power spectrum
    acorr = numpy.fft.ifft(pwr).real / var / len(signal)
    return acorr


def numpy_correlate(signal):
    x = np.array(signal)

    # Mean
    mean = numpy.mean(signal)

    # Variance
    var = numpy.var(signal)

    # Normalized data
    ndata = signal - mean

    acorr = numpy.correlate(ndata, ndata, 'full')[len(ndata) - 1:]
    acorr = acorr / var / len(ndata)

    return acorr


def autocorr(signal, m, N) -> tuple[float, float]:
    peak: float = 0
    lag: int = 0
    for i in np.arange(20, 150):
        autoc: float = 0
        for n in np.arange(m - N + 1, m):
            autoc = autoc + signal[n] * signal[n - i]
            if autoc > peak:
                peak = autoc
                lag = i
    return lag, peak


def show_autcorrelation(signal, Nframe) -> None:
    FS: int = 180
    result: list[Autcorr] = []
    for i in np.arange(0, Nframe):
        lag, peak = autocorr(signal[i * FS: (i + 1) * FS], 160, 40)
        result.append(Autcorr(
            lag=lag,
            peak=peak,
            range=((i * FS), (i + 1) * FS),
            SFS=150,
            FS=FS
        ))
    for index, item in enumerate(result):
        print(
            f'index = {index}, lag = {item.lag}, peak={item.peak}, range={item.range[0]}:{item.range[1]},'
            f'FSF and FS = {item.SFS, item.FS}'
        )
    plt.plot(signal[result[0].range[0]:result[0].range[1]])
    amplitude = max(signal[result[0].range[0]:result[0].range[1]])
    show_signal = amplitude * np.sin((2 * np.pi) * result[0].lag * np.arange(0, (1 / 8000) * result[0].FS, (1 / 8000)))
    plt.plot(show_signal)
    plt.show()
    """s = numpy_correlate(signal[:150])
    plt.plot(signal[:150])
    plt.plot(s)
    plt.show()"""
