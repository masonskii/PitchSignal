import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.signal import lfilter
from lib.coeff import *
from lib.pitch import *
from lib.filters import *
from lib.Pitching import Pitching
from lib.autocorrelation import autocorr, show_autcorrelation


def open_file_signal(filename='output_sound.wav', FRL=180, fdtype='uint8', adtype=float) -> tuple[ndarray, int]:
    """
    :param filename:   file with signal
    :param adtype:  Тип данных для массива
    :param fdtype: Тип данных для файла
    :param FRL: Длинна кадра
    :return:
        s - массив сигнала
        NFrame - Число кадров
    """
    try:
        import numpy as np

        if adtype == float:
            adtype = np.float64
        s = np.transpose(np.fromfile(filename, dtype=fdtype))
        s = np.array(s[7 * 8 + 2:], dtype=adtype)
        s = s - 128
        s = s / 128
        s = s * 32767
        Nframe = np.int32(np.fix(len(s) / FRL))
        return s, Nframe
    except Exception as e:
        raise e


def pitch_coding() -> list[Pitching]:
    """
    Инициализация
    """
    try:
        FS = 180  # Длина кадра
        INPUT_SIGNAL = np.zeros(FS * 2)  # входной сигнал
        FNCHSIGNAL_1000 = np.zeros(FS * 2)  # сигнал на входе ФНЧ (1000 Гц)
        BANDS = np.zeros((5, (FS * 2)))  # полосовые сигналы

        # начальные состояния
        CFILTER = np.zeros(4)  # chebichev filter
        BFILTER = np.zeros(6)  # buttervorter
        BSTATE = np.zeros((5, 6))
        ESTATE = np.zeros((4, 2))
        ENV = np.zeros((4, (FS * 2)))  # огибающие полосовых сигналов
        BUFFER = [50, 50, 50]  # буфер медианного фильтра ОТ
        AVG = 50  # значение ОТ для случая низкой корреляции значений ОТ,
        s, Nframe = open_file_signal()
        global Qpitch
        RESULT: list[Pitching] = []
    except Exception as e:
        raise f'Initialization error{e}'
    for i in np.arange(Nframe - 1):
        V = np.zeros(5)
        Qpitch = 0.0
        INPUT_SIGNAL[:FS] = INPUT_SIGNAL[FS: FS * 2]
        FNCHSIGNAL_1000[:FS] = FNCHSIGNAL_1000[FS: FS * 2]
        BANDS[:, :FS] = BANDS[:, FS:FS * 2]
        ENV[:, :FS] = ENV[:, FS:FS * 2]

        """ Takes new frame speech"""
        new_frame_speech = s[i * FS: (i + 1) * FS]
        INPUT_SIGNAL[FS:FS * 2], CFILTER = lfilter(dcr_num, dcr_den, new_frame_speech, zi=CFILTER)
        FNCHSIGNAL_1000[FS: FS * 2], BFILTER = lfilter(butt_1000num, butt_1000den, INPUT_SIGNAL[FS:FS * 2],
                                                       zi=BFILTER)
        CURRENT_INTEGER_PITCH = intpitch(FNCHSIGNAL_1000, 160, 40)
        BANDS[:, FS:FS * 2], BSTATE, ENV[:, FS:FS * 2], ESTATE = filter_5b(INPUT_SIGNAL[FS: FS * 2], BSTATE, ESTATE)
        P2, V[0] = pitch2(BANDS[0, :], CURRENT_INTEGER_PITCH)
        koefficients = filter_lpc(INPUT_SIGNAL[(FS - 100):(FS + 100)])
        koefficients = koefficients * np.power(0.994, np.arange(2, 12))
        RESID = filter_lpc_residual(koefficients, INPUT_SIGNAL)
        peak = np.sqrt(np.matmul(RESID[105:265], np.conj(np.transpose(RESID[105:265]))) / 160) / (
                np.sum(np.abs(RESID[105:265])) / 160)
        if peak > 1.34:
            V[0] = 1
        temp = np.zeros(6)
        filtered_resid, temp = lfilter(butt_1000num, butt_1000den, RESID, zi=temp)
        temp = np.reshape(temp, (-1, 1))
        filtered_resid = np.append((0, 0, 0, 0, 0), filtered_resid)
        filtered_resid = np.append(filtered_resid, (0, 0, 0, 0, 0))
        P3, R3 = pitch3(INPUT_SIGNAL, filtered_resid, P2, AVG)
        G = filter_gain(INPUT_SIGNAL, V[0], P2)
        AVG, BUFFER = filter_APU(P3, R3, G[1], BUFFER)
        AVG = int(AVG)
        if V[0] > 0.6:
            RESULT.append(Pitching(P3, ((i * FS, (i + 1) * FS)), max(new_frame_speech)))
        else:
            RESULT.append(Pitching(0.0, ((i * FS, (i + 1) * FS)), max(new_frame_speech)))
    return RESULT


def show_structured(struct: list[Pitching]) -> None:
    for index in range(len(struct)):
        print(
            f'Struct #{index}\n Pitch = {struct[index].pitch}\n In range = {struct[index].range}\n'
            f'with amplitude={struct[index].amplitude} \nand sampling = {struct[index].sampling}'
        )


def plot_frame(pitch, amplitude, range, sampling) -> None:
    signal, nframe = open_file_signal()
    # plt.plot(signal[0 * range[0]:(0 + 1) * range[1]])
    plt.plot(signal[range[0]:range[1]])
    show_signal = amplitude * np.sin((2 * np.pi) * pitch * np.arange(0, sampling * 180, sampling))
    plt.plot(show_signal)
    # plt.plot(amplidute * np.sin(2 * np.pi) * pitch * np.array((0, ((1 / 8000) * len(signal)), (1 / 8000))))
    # plt.plot(amplidute * np.sin(2 * np.pi * pitch * 1 / 8000))
    plt.show()


"""gibrid-integer pitch method"""
res = pitch_coding()
show_structured(res)
plot_frame(res[0].pitch, res[0].amplitude, res[0].range, res[0].sampling)

"""Autocorrelation method"""
"""
s, frame = open_file_signal()
show_autcorrelation(s, frame)"""
