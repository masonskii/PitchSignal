import numpy
from scipy.signal import lfilter

from lib.coeff import *
from numpy import real
from numpy.fft import fft, ifft
from numpy import ceil, log2
from statistics import median


def nextpow2(x):
    res = ceil(log2(x))
    return res.astype('int')  # we want integer values only but ceil gives float


def LEVINSON(r, order=None, allow_singularity=False):
    T0 = numpy.real(r[0])
    T = r[1:]
    M = len(T)

    if order is None:
        M = len(T)
    else:
        assert order <= M, 'order must be less than size of the input data'
        M = order

    realdata = numpy.isrealobj(r)
    if realdata is True:
        A = numpy.zeros(M, dtype=float)
        ref = numpy.zeros(M, dtype=float)
    else:
        A = numpy.zeros(M, dtype=complex)
        ref = numpy.zeros(M, dtype=complex)

    P = T0

    for k in range(0, M):
        save = T[k]
        if k == 0:
            temp = -save / P
        else:
            # save += sum([A[j]*T[k-j-1] for j in range(0,k)])
            for j in range(0, k):
                save = save + A[j] * T[k - j - 1]
            temp = -save / P
        if realdata:
            P = P * (1. - temp ** 2.)
        else:
            P = P * (1. - (temp.real ** 2 + temp.imag ** 2))
        if P <= 0 and allow_singularity == False:
            raise ValueError("singular matrix")
        A[k] = temp
        ref[k] = temp  # save reflection coeff at each step
        if k == 0:
            continue

        khalf = (k + 1) // 2
        if realdata is True:
            for j in range(0, khalf):
                kj = k - j - 1
                save = A[j]
                A[j] = save + temp * A[kj]
                if j != kj:
                    A[kj] += temp * save
        else:
            for j in range(0, khalf):
                kj = k - j - 1
                save = A[j]
                A[j] = save + temp * A[kj].conjugate()
                if j != kj:
                    A[kj] = A[kj] + temp * save.conjugate()

    return A, P, ref


def rlevinson(a, efinal):
    a = numpy.array(a)
    realdata = numpy.isrealobj(a)

    assert a[0] == 1, 'First coefficient of the prediction polynomial must be unity'

    p = len(a)

    if p < 2:
        raise ValueError('Polynomial should have at least two coefficients')

    if realdata == True:
        U = numpy.zeros((p, p))  # This matrix will have the prediction
        # polynomials of orders 1:p
    else:
        U = numpy.zeros((p, p), dtype=complex)
    U[:, p - 1] = numpy.conj(a[-1::-1])  # Prediction coefficients of order p

    p = p - 1
    e = numpy.zeros(p)

    # First we find the prediction coefficients of smaller orders and form the
    # Matrix U

    # Initialize the step down

    e[-1] = efinal  # Prediction error of order p

    # Step down
    for k in range(p - 1, 0, -1):
        [a, e[k - 1]] = levdown(a, e[k])
        U[:, k] = numpy.concatenate((numpy.conj(a[-1::-1].transpose()),
                                     [0] * (p - k)))

    e0 = e[0] / (1. - abs(a[1] ** 2))  # % Because a[1]=1 (true polynomial)
    U[0, 0] = 1  # % Prediction coefficient of zeroth order
    kr = numpy.conj(U[0, 1:])  # % The reflection coefficients
    kr = kr.transpose()  # % To make it into a column vector

    #   % Once we have the matrix U and the prediction error at various orders, we can
    #  % use this information to find the autocorrelation coefficients.

    R = numpy.zeros(1, dtype=complex)
    # % Initialize recursion
    k = 1
    R0 = e0  # To take care of the zero indexing problem
    R[0] = -numpy.conj(U[0, 1]) * R0  # R[1]=-a1[1]*R[0]

    # Actual recursion
    for k in range(1, p):
        r = -sum(numpy.conj(U[k - 1::-1, k]) * R[-1::-1]) - kr[k] * e[k - 1]
        R = numpy.insert(R, len(R), r)

    # Include R(0) and make it a column vector. Note the dot transpose

    # R = [R0 R].';
    R = numpy.insert(R, 0, e0)
    return R, U, kr, e


def levdown(anxt, enxt=None):

    if anxt[0] != 1:
        raise ValueError('At least one of the reflection coefficients is equal to one.')
    anxt = anxt[1:]  # Drop the leading 1, it is not needed
    #  in the step down

    # Extract the k+1'th reflection coefficient
    knxt = anxt[-1]
    if knxt == 1.0:
        raise ValueError('At least one of the reflection coefficients is equal to one.')

    # A Matrix formulation from Stoica is used to avoid looping
    acur = (anxt[0:-1] - knxt * numpy.conj(anxt[-2::-1])) / (1. - abs(knxt) ** 2)
    ecur = None
    if enxt is not None:
        ecur = enxt / (1. - numpy.dot(knxt.conj().transpose(), knxt))

    acur = numpy.insert(acur, 0, 1)

    return acur, ecur


def levup(acur, knxt, ecur=None):
    if acur[0] != 1:
        raise ValueError('At least one of the reflection coefficients is equal to one.')
    acur = acur[1:]  # Drop the leading 1, it is not needed

    # Matrix formulation from Stoica is used to avoid looping
    anxt = numpy.concatenate((acur, [0])) + knxt * numpy.concatenate((numpy.conj(acur[-1::-1]), [1]))

    enxt = None
    if ecur is not None:
        # matlab version enxt = (1-knxt'.*knxt)*ecur
        enxt = (1. - numpy.dot(numpy.conj(knxt), knxt)) * ecur

    anxt = numpy.insert(anxt, 0, 1)

    return anxt, enxt


def lpc(x, N=None):
    m = len(x)
    if N == None:
        N = m - 1  # default value if N is not provided
    elif N > m - 1:
        # disp('Warning: zero-padding short input sequence')
        x.resize(N + 1)
        # todo: check this zero-padding.

    X = fft(x, 2 ** nextpow2(2. * len(x) - 1))
    R = real(ifft(abs(X) ** 2))
    R = R / (m - 1.)  # Biased autocorrelation estimate
    a, e, ref = LEVINSON(R, N)
    return a, e


def filter_5b(sig_in, state_b, state_e):
    try:
        import numpy as np
        from scipy.signal import lfilter

        bands = np.zeros((5, 180))
        envelopes = np.zeros((4, 180))
        b = np.asarray(butt_bp_num).reshape(5, 7)
        a = np.asarray(butt_bp_den).reshape(5, 7)
        for i in np.arange(5):  # фильтрация в каждой из 5 полос
            bands[i, :], state_b[i, :] = lfilter(b[i, :], a[i, :], sig_in, zi=state_b[i, :])
        temp1 = np.abs(bands[0:4, :])  # абсолютные значения полосовых (1-4) сигналов
        a = np.asarray(smooth_den).reshape(1, 3)
        b = np.asarray(smooth_num).reshape(1, 3)
        for i in np.arange(4):
            envelopes[i, :], state_e[i, :] = lfilter(b[0, :], a[0, :], temp1[i, :],
                                                     zi=state_e[i, :])  # сглаживающий фильтр

        return bands, state_b, envelopes, state_e
    except Exception as e:
        raise e


def filter_lpc(s):
    try:
        import numpy as np

        a = np.zeros(11)
        f = np.zeros(10)
        v = s * np.conj(np.transpose(np.hamming(200)))
        a, _ = lpc(v, N=10)  # N = order
        f = a
        return a
    except Exception as e:
        raise e


def filter_APU(p3, rp3, G2, buffer):

    if rp3 > 0.8 and G2 > 30:
        buffer[0:2] = buffer[1:3]
        buffer[2] = p3
    else:
        buffer = [b * 0.95 + 2.5 for b in buffer]

    pavg = median(buffer)
    return pavg, buffer


def filter_gain(s, vbp1, p2):
    k = 1
    Ltmp = p2
    Lfr = p2

    if vbp1 > 0.6:
        while Ltmp < 180:  # определение целого числа периодов к
            k = k + 1
            Lfr = Ltmp
            Ltmp = p2 * k
    else:
        Lfr = 120  # длительность сигнала, включающего целое число периодов
    HL = np.int32(np.around(Lfr / 2))
    Lfr = np.int32(HL * 2)
    G = np.zeros(2)
    # Усиление для первого подкадра
    G[0] = 10 * np.log10(0.01 + np.matmul(s[90 - HL:90 + HL], np.conj(np.transpose(s[90 - HL:90 + HL]) / Lfr)))
    G[1] = 10 * np.log10(0.01 + np.matmul(s[180 - HL:180 + HL], np.conj(np.transpose(s[180 - HL:180 + HL]) / Lfr)))

    for i in np.arange(1):
        if G[i] < 0:
            G[i] = 0
    return G


def filter_lpc_residual(lpcs, sig_in):
    try:
        exc = lfilter(np.append(1, lpcs), 1, sig_in)
        return exc[10:]
    except Exception as e:
        raise e
