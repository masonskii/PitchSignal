import numpy as np
from numpy import ndarray, fix
import cmath

__all__ = ['fpr', 'double_ck', 'double_ver', 'intpitch', 'pitch3', 'pitch2']


def fpr(sig, T):
    try:
        T = int(T)
        k = int(T / 2)
        # Автокорреляция
        c0_tm1 = int(np.matmul(sig[100 - k:259 - k],
                               np.conj(np.transpose(np.reshape(sig[100 - k + T - 1:259 - k + T - 1], (1, -1))))))
        c0_t1 = int(np.matmul(sig[100 - k:259 - k], np.conj(
            np.transpose(np.reshape(sig[100 - k + T + 1:259 - k + T + 1], (1, -1))))))
        c0_t = int(np.matmul(sig[100 - k:259 - k],
                             np.conj(np.transpose(np.reshape(sig[100 - k + T:259 - k + T], (1, -1))))))
        if c0_tm1 > c0_t1:  # Оценка диапазона fp
            c0_t1 = c0_t
            c0_t = c0_tm1
            T = T - 1

        ct_t = int(np.matmul(sig[100 - k + T:259 - k + T],
                             np.conj(np.transpose(np.reshape(sig[100 - k + T:259 - k + T], (1, -1))))))
        c0_0 = int(np.matmul(sig[100 - k:259 - k], np.conj(np.transpose(np.reshape(sig[100 - k:259 - k], (1, -1))))))

        ct_t1 = int(np.matmul(sig[100 - k + T:259 - k + T],
                              np.conj(np.transpose(np.reshape(sig[100 - k + T + 1:259 - k + T + 1], (1, -1))))))
        ct1_t1 = int(np.matmul(sig[100 - k + T + 1:259 - k + T + 1],
                               np.conj(np.transpose(np.reshape(sig[100 - k + T + 1:259 - k + T + 1], (1, -1))))))

        # Параметр корреляции
        den = c0_t1 * (ct_t - ct_t1) + c0_t * (ct1_t1 - ct_t1)  # Знаменатель delta
        if np.abs(den) > 0.01:
            delta = (c0_t1 * ct_t - c0_t * ct_t1) / den
        if np.abs(den) < 0.01:
            delta = 0.5

        # Проверка граничных условий для параметра корреляции

        if delta < -1:
            delta = -1
        if delta > 2:
            delta = 2

        fp = T + delta  # Добавление смещения к целочисленному значению к

        # Отчет соответствующей корреляции
        den = c0_0 * (ct_t * float(np.power((1 - delta), 2)) + 2 * delta * (1 - delta) * ct_t1 + float(
            np.power(delta, 2)) * ct1_t1)
        den = cmath.sqrt(den).real
        if den > 0.01:
            fr = ((1 - delta) * c0_t + delta * c0_t1) / den
        else:
            fr = 0

        # проверка граничных условий для задержки ќ
        if fp < 20:
            fp = 20
        if fp > 160:
            fp = 160

        return fp, fr

    except Exception as e:
        raise e


def double_ver(sig_in, pp, cor_p):

    try:
        np, ncor_p = fpr(sig_in, round(2 * pp))  # вычисление корреляции для удвоенного значения К
        if ncor_p < cor_p:  # если рассчитанное значение корреляции меньше входной,
            cor_p = ncor_p  # то оно присваивается входной
        return cor_p

    except Exception as e:
        raise e


def double_ck(sig_in, p, Dth):
    try:
        pmin = 20  # Минимальное значение ОТ
        pc, cor_pc = fpr(sig_in, np.around(p))  # улучшение дробного значения ОТ
        for i in np.arange(7):  # поиск лучшего ОТ
            k = 9 - (i + 1)  # значение делителя
            temp_pit = np.around(pc / k)

            if temp_pit >= pmin:
                temp_pit, temp_cor = fpr(sig_in, temp_pit)  # улучшение дробного значение ОТ
                if temp_pit < 30:
                    temp_cor = double_ver(sig_in, temp_pit, temp_cor)  # удаление низкого ОТ
                if temp_cor > Dth * cor_pc:
                    pc, cor_pc = fpr(sig_in, round(temp_pit))  # улучшение дробного значения ОТ
                    break
        if pc < 30:
            cor_pc = double_ver(sig_in, pc, cor_pc)  # удаление низкого ОТ

        return pc, cor_pc
    except Exception as e:
        raise e


def intpitch(ss, ipmax, ipmin):
    try:
        r = 0  # максимальное значение нормированной функции автокорреляции
        T = 80  # начальное значение задержки ОТ
        r_new = 0  # текущее значение нормированной функции автокорреляции (НФАК)
        ipmax, ipmin = int(ipmax), int(ipmin)
        for tao in np.arange(ipmin, ipmax):
            k = np.int32(np.fix(tao / 2))
            c0_t = np.matmul(ss[100 - k:259 - k],
                                 np.transpose(np.reshape(ss[100 - k + tao: 259 - k + tao], (1, -1))))[0]  # числитель НФАК
            c0_0 = np.matmul(ss[100 - k:259 - k], np.transpose(np.reshape(ss[100 - k:259 - k], (1, -1))))[0]
            ct_t = np.matmul(ss[100 - k + tao:259 - k + tao],
                                 np.transpose(np.reshape(ss[100 - k + tao:259 - k + tao], (1, -1))))[0]
            den = c0_0 * ct_t  # знаменатель НФАК
            if den > 0:
                r_new = c0_t * c0_t / den  # расчет НФАК
            if r_new > r:
                r = r_new  # определение максимального значения НФАК
                T = tao  # и соответствующей задержки ОТ

        return T
    except Exception as e:
        raise e


def pitch2(sig, intp):
    try:
        low = intp - 5  # определение нижней границы поиска задержки ОТ (не ниже 20)
        if low < 20:
            low = 20

        up = intp + 5  # определение верхней границы поиска задержки ОТ (не выше 160)
        if up > 160:
            up = 160

        p = intpitch(sig, up, low)  # определение значения ОТ в указанных пределах

        p, r = fpr(sig, p)  # улучшение дробного значения ОТ

        return p, r
    except Exception as e:
        raise e


def pitch3(sig_in, resid, p2, pavg):
    try:

        p2 = int(np.around(p2))  # округление дробного значения ОТ
        p3, rp3 = pitch2(resid, p2)  # вычисление дробного значения ОТ по сигналу остатка предсказания

        if rp3 >= 0.6:
            Dth = 0.5  # пороговое значение
            if p3 <= 100:
                Dth = 0.75  # пороговое значение

            p3, rp3 = double_ck(resid, p3, Dth)  # Определение ОТ с удвоенной точностью
        else:
            p3, rp3 = fpr(sig_in, p2)  # улучшение дробного значения
            if rp3 < 0.55:
                p3 = pavg
            else:
                Dth = 0.7
                if p3 <= 100:
                    Dth = 0.9

                p3, rp3 = double_ck(sig_in, p3, Dth)
        return p3, rp3
    except Exception as e:
        raise e
