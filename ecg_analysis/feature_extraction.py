import json
from pathlib import Path
import numpy as np
import pandas as pd
import neurokit2 as nk
from math import atan2, degrees, floor
from scipy.signal import find_peaks 
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def load_ecg_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    fs = int(data.get("DiscretizationFrequency", 500))
    leads = {}
    for lead in data['Leads']:
        name = lead['Name']
        x = np.array([s['X'] for s in lead['Samples']], dtype=float)
        y = np.array([s['Y'] for s in lead['Samples']], dtype=float)
        leads[name] = {'X': x, 'Y': y}
    return leads, fs 


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k,v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj 


def estimate_u_wave(signal, t_offset_idx, next_p_onset_idx, fs=500, min_prominence=0.02):
    start = t_offset_idx + 1
    end = next_p_onset_idx - 1 if next_p_onset_idx is not None else min(len(signal)-1, t_offset_idx + int(400/1000*fs))
    if start >= end:
        return None  # нет пространства
    seg = signal[start:end+1]
    # ищем небольшой пик в том же направлении, что и T (если T положител., ищем положит.)
    # настроим prominence относительно амплитуды T
    peaks, props = find_peaks(seg, prominence=min_prominence)
    if peaks.size == 0:
        # пробуем обратную полярность
        peaks, props = find_peaks(-seg, prominence=min_prominence)
        if peaks.size == 0:
            return None
        else:
            return start + int(peaks[0])
    return start + int(peaks[0])


def estimate_j_point(signal, qrs_offset_idx, fs=500, search_ms=60):
    """
    Ищем в окне [qrs_offset, qrs_offset + search_ms] первую точку, где производная близка к нулю
    или где уровень стабилизируется относительно baseline.
    """
    n = len(signal)
    end = min(n-1, qrs_offset_idx + int(search_ms/1000*fs))
    if qrs_offset_idx >= end:
        return qrs_offset_idx
    seg = signal[qrs_offset_idx:end+1]
    # вычислим первую точку, где наклон небольшой (меньше некоторого порога)
    deriv = np.diff(seg)
    if len(deriv)==0:
        return qrs_offset_idx
    # порог как 10% от макс(|deriv|)
    dth = 0.1 * max(1e-9, np.max(np.abs(deriv)))
    for i, d in enumerate(deriv):
        if abs(d) < dth:
            return qrs_offset_idx + i
    # fallback - конец окна
    return end 


def estimate_onset_offset(signal, peak_idx, baseline, side='left', fs=500, threshold_frac=0.05, max_ms=120):
    """
    Оценка начала (onset) или конца (offset) зубца относительно базовой линии.

    signal: np.array или list - ЭКГ сигнал
    peak_idx: int - индекс пика зубца
    baseline: float - значение базовой линии (TP segment)
    side: 'left' -> onset, 'right' -> offset
    fs: частота дискретизации
    threshold_frac: доля амплитуды от пика относительно baseline для определения окончания зубца
    max_ms: максимальный поиск в миллисекундах
    """
    n = len(signal)
    peak_amp = abs(signal[peak_idx] - baseline)  # амплитуда относительно baseline
    thresh = threshold_frac * peak_amp
    max_samples = int(max_ms / 1000 * fs)

    if side == 'left':
        for step in range(1, min(max_samples, peak_idx) + 1):
            cur = peak_idx - step
            if abs(signal[cur] - baseline) < thresh:
                return cur
        return max(0, peak_idx - max_samples)
    else:
        for step in range(1, min(max_samples, n - peak_idx - 1) + 1):
            cur = peak_idx + step
            if abs(signal[cur] - baseline) < thresh:
                return cur
        return min(n - 1, peak_idx + max_samples)



def calculate_eos(lead_I, lead_aVF, q_peaks_I, r_peaks_I, s_peaks_I, q_peaks_aVF, r_peaks_aVF, s_peaks_aVF):
    """ Расчёт электрической оси сердца (ЭОС) """

    def peak_amplitude(signal, peaks):
        return np.mean([signal[p] for p in peaks]) if len(peaks) else 0

    # амплитуды зубцов
    A_Q_I = peak_amplitude(lead_I, q_peaks_I)
    A_R_I = peak_amplitude(lead_I, r_peaks_I)
    A_S_I = peak_amplitude(lead_I, s_peaks_I)

    A_Q_aVF = peak_amplitude(lead_aVF, q_peaks_aVF)
    A_R_aVF = peak_amplitude(lead_aVF, r_peaks_aVF)
    A_S_aVF = peak_amplitude(lead_aVF, s_peaks_aVF)

    # алгебраическая сумма амплитуд
    A_I = A_R_I + A_Q_I + A_S_I
    A_aVF = A_R_aVF + A_Q_aVF + A_S_aVF

    # вычисление угла
    alpha = np.degrees(np.arctan2(A_aVF, A_I))

    is_normal_eos = False 
    # классификация
    if 0 <= alpha <= 90:
        is_normal_eos = True 
        if 0 <= alpha <= 30: 
            interpretation = 'ЭОС горизонтально'
        elif 30 <= alpha <= 70:
            interpretation = 'ЭОС в норме'
        else:
            interpretation = 'ЭОС вертикально'
    elif -30 <= alpha <= 0:
        interpretation = 'ЭОС отклонена влево'
    elif -90 <= alpha <= -30:
        interpretation = 'ЭОС резко отклонена влево'    
    elif 90 <= alpha <= 120:
        interpretation = 'ЭОС отклонена вправо'
    else:
        interpretation = 'ЭОС резко отклонена вправо'

    return alpha, is_normal_eos, interpretation


def identify_rhythm(p_peaks_dict, qrs_count, is_rr_regular):
    ''' Синусоидный или эктопический ритм '''
    p_peaks_I = p_peaks_dict['I']
    p_peaks_II = p_peaks_dict['II']
    p_peaks_aVF = p_peaks_dict['aVF']
    p_peaks_V2 = p_peaks_dict['V2']
    p_peaks_V3 = p_peaks_dict['V3']
    p_peaks_V4 = p_peaks_dict['V4']
    p_peaks_V5 = p_peaks_dict['V5']
    p_peaks_V6 = p_peaks_dict['V6']
    p_peaks_aVR = p_peaks_dict['aVR']
    p_peaks_aVR = p_peaks_dict['aVF']

    is_sin = False 

    # критерий синусового ритма: зубцы в отведениях I-II, V2-V6 и aVF положительны, в aVR отрицательны
    # в остальных могут быть разными 
    is_positive_required_peaks = all(map(lambda i: i > 0,
                                        p_peaks_I, p_peaks_II, p_peaks_aVF, p_peaks_V2,
                                        p_peaks_V3, p_peaks_V4, p_peaks_V5, p_peaks_V6))
    is_negative_required_peaks = all(map(lambda i: i > 0, p_peaks_aVR)) 

    # зубцы P перед каждым QRS
    is_p_count_normal = len(p_peaks_II) == qrs_count

    # интервал RR или PP постоянный (10%)
    if is_p_count_normal and is_rr_regular and is_positive_required_peaks and is_negative_required_peaks:
        is_sin = True

    return is_sin


def calculate_baseline(t_offsets, p_onsets, lead_ii, fs=500):
    """
    Базовая линия = медиана по всем TP-сегментам (от T_offset до ближайшего следующего P_onset),
    """
    tp_segments = []

    for t in t_offsets:
        # ближайший P после данного T
        next_ps = [p for p in p_onsets if p > t]
        if not next_ps:
            continue
        p = next_ps[0]

        dur_samples = p - t
        dur_ms = dur_samples / fs * 1000.0

        seg = lead_ii[t:p]
        if len(seg) > 0:
            tp_segments.append(seg)

    if not tp_segments:
        # фоллбек — медиана по всему сигналу
        return float(np.median(lead_ii))

    all_tp = np.concatenate(tp_segments)
    return float(np.median(all_tp))


def calculate_intervals(r_peaks, q_peaks, p_peaks, fs=500):
    ''' Вычисляет интервалы RR, PP, PQ (PR если Q нет) в милисекундах
    '''
    
    rr_intervals_ms = np.diff(np.array(r_peaks)) / fs * 1000.0 if len(r_peaks) >= 2 else np.array([])
    pp_intervals_ms = np.diff(np.array(p_peaks)) / fs * 1000.0 if len(p_peaks) >= 2 else np.array([])

    pq_intervals_ms = []
    for p in p_peaks:
        qs_after = [q for q in q_peaks if q > p]
        if qs_after:
            q_index = qs_after[0]
        else:
            rs_after = [r for r in r_peaks if r > p]
            q_index = rs_after[0] if rs_after else None
        if q_index:
            pq_intervals_ms.append((q_index - p)/fs*1000.0)
    pq_intervals_ms = np.array(pq_intervals_ms)
    return rr_intervals_ms, pp_intervals_ms, pq_intervals_ms


def identify_rhythm_regularity(intervals):
    ''' Определяет регулярный ритм или нет '''
    if len(intervals) == 0:
        return False
    mean = float(np.mean(intervals))
    if mean == 0 or np.isnan(mean):
        return False
    std = float(np.std(intervals))
    cv = std / mean
    return cv <= 0.10 # разброс не больше 0.1, можно поставить до 0.15


def process_file(json_path, patients_csv=None, method='neurokit'):
    """
    Полный процесс обработки одного файла:
    добавить длины сегментов + амплитуды зубцов + синусоидность ритма + форму QRS
    """
    json_path = Path(json_path)
    leads_data, fs = load_ecg_from_json(json_path)
    lead_names = list(leads_data.keys())

    result = {
        'filename': str(json_path.name),
        'FS': int(fs)
    }

    # информация о пациенте - сейчас считывается из файла
    if patients_csv:
        df = pd.read_csv(patients_csv, sep=';')
        fname = json_path.stem  # "00001_hr"
        match = df[df['filename_hr'].str.endswith(fname, na=False)]
        if not match.empty:
            patient_info = match.iloc[0].to_dict()
        else:
            patient_info = {"filename_hr": fname, "note": "Данных пациента нет"}
    else:
        patient_info = {"filename_hr": json_path.name, "note": "Информация о пациентах не передана"}

    # считаем изоэлектрическую линию + ЧСС + регулярность интервалов для всех отведений по отведению II
    lead_ii = leads_data['II']['Y']
    signals_ii, _ = nk.ecg_process(lead_ii, sampling_rate=fs, method=method)
    clean_lead_ii = signals_ii['ECG_Clean'].tolist()

    def mask_to_idx(series):
        arr = np.array(series)
        return np.where(arr == 1)[0]

    r_peaks = mask_to_idx(signals_ii['ECG_R_Peaks'])
    p_peaks = mask_to_idx(signals_ii['ECG_P_Peaks'])
    s_peaks = mask_to_idx(signals_ii['ECG_S_Peaks'])
    q_peaks = mask_to_idx(signals_ii['ECG_Q_Peaks'])
    p_onsets = mask_to_idx(signals_ii['ECG_P_Onsets'])
    t_offsets = mask_to_idx(signals_ii['ECG_T_Offsets'])

    baseline = calculate_baseline(t_offsets, p_onsets, clean_lead_ii)
    heart_rate = signals_ii['ECG_Rate'].tolist()
    rr_intervals_ms, pp_intervals_ms, pq_intervals_ms = calculate_intervals(r_peaks, q_peaks, p_peaks)
    is_rr_regular = identify_rhythm_regularity(rr_intervals_ms)
    is_pp_regular = identify_rhythm_regularity(pp_intervals_ms)
    is_pq_regular = identify_rhythm_regularity(pq_intervals_ms)

    result = {
        'filename': str(json_path.name),
        'FS': int(fs),
        'patient_info': patient_info,
        'baseline': baseline,
        'heart_rate': heart_rate,
        'rr_intervals_ms': rr_intervals_ms,
        'pp_intervals_ms': pp_intervals_ms,
        'pq_intervals_ms': pq_intervals_ms,
        'is_rr_regular': is_rr_regular,
        'is_pp_regular': is_pp_regular,
        'is_pq_regular': is_pq_regular
    }

    for lead_name in lead_names:
        lead = leads_data[lead_name]['Y']

        # ecg_process: возвращает (signals_df, rpeaks) или (signals_df, info)
        signals, _ = nk.ecg_process(lead, sampling_rate=fs, method=method)
        r_peaks = mask_to_idx(signals['ECG_R_Peaks'])
        p_peaks = mask_to_idx(signals['ECG_P_Peaks'])
        s_peaks = mask_to_idx(signals['ECG_S_Peaks'])
        q_peaks = mask_to_idx(signals['ECG_Q_Peaks'])
        t_offsets = mask_to_idx(signals['ECG_T_Offsets'])
        p_onsets = mask_to_idx(signals['ECG_P_Onsets'])

        # если onsets/offsets не найдены — оценим по пикам
        # сделаем оценочные on/off для QRS: onset = left of Q or left of R, offset = right of S or right of R
        # для этого ищем ближайшие пики q, r, s рядом с каждым R
        qrs_onsets = []
        qrs_offsets = []
        for rp in r_peaks:
            q_left = [q for q in q_peaks if q <= rp]
            q_idx = q_left[-1] if q_left else rp
            on = estimate_onset_offset(lead, q_idx, baseline, side='left', fs=fs)

            s_right = [s for s in s_peaks if s >= rp]
            s_idx = s_right[0] if s_right else rp
            off = estimate_onset_offset(lead, s_idx, baseline, side='right', fs=fs)

            qrs_onsets.append(int(on))
            qrs_offsets.append(int(off))

        # j_point - начало зубца T (конец сегмента ST)
        j_points = mask_to_idx(signals['ECG_T_Onsets'])

        # U-wave estimation per beat: для каждого beat берем T_offset and next P_onset
        u_peaks = []
        # prepare next P_onset map
        p_onset_sorted = sorted(p_onsets)
        for i, off in enumerate(t_offsets):
            # find next P_onset after t_offset
            next_p = None
            for p_on in p_onset_sorted:
                if p_on > off:
                    next_p = p_on
                    break
            u = estimate_u_wave(lead, off, next_p, fs=fs, min_prominence=0.02)
            u_peaks.append(int(u) if u is not None else None)

        # амплитуда j относительно бейзлайна
        st_at_j = []
        for j in j_points:
            if j is None:
                st_at_j.append(None)
            else:
                st_at_j.append(float(lead[j] - baseline))

        # QRS durations (ms)
        qrs_durations_ms = []
        for on, off in zip(qrs_onsets, qrs_offsets):
            qrs_durations_ms.append((off - on)/fs*1000.0)

        # агрегируем результат по текущему отведению
        lead_dict = {
            'raw_signal': lead.tolist(),
            'clean_signal': signals['ECG_Clean'].tolist(),
            'timeline_x': leads_data[lead_name]['X'],
            'r_peaks': r_peaks,
            'p_peaks': p_peaks,
            'q_peaks': q_peaks,
            's_peaks': s_peaks,
            't_peaks': mask_to_idx(signals['ECG_T_Peaks']),
            't_onsets': mask_to_idx(signals['ECG_T_Onsets']),
            't_offsets': t_offsets,
            'p_onsets': p_onsets,
            'p_offsets': mask_to_idx(signals['ECG_P_Offsets']),
            'r_onsets': mask_to_idx(signals['ECG_R_Onsets']),
            'r_offsets': mask_to_idx(signals['ECG_R_Offsets']),
            "u_peaks": [int(u) if u is not None else None for u in u_peaks],
            "qrs_onsets": qrs_onsets,
            "qrs_offsets": qrs_offsets,
            "j_points": j_points,
            "qrs_durations_ms": qrs_durations_ms,
            "st_at_j_mv": st_at_j,
            'cycles_count': len(r_peaks)
        }
        result[lead_name] = lead_dict 

    # считаем ЭОС
    clean_lead_I = result['I']['clean_signal']
    clean_lead_aVF = result['aVF']['clean_signal']
    q_peaks_I = result['I']['q_peaks']
    r_peaks_I = result['I']['r_peaks']
    s_peaks_I = result['I']['s_peaks']
    q_peaks_aVF = result['aVF']['q_peaks']
    r_peaks_aVF = result['aVF']['r_peaks']
    s_peaks_aVF = result['aVF']['s_peaks']
    eos_degree, is_normal_eos, eos_interpretation = calculate_eos(clean_lead_I, clean_lead_aVF, 
                                                                    q_peaks_I, r_peaks_I, s_peaks_I,
                                                                    q_peaks_aVF, r_peaks_aVF, s_peaks_aVF)
    result['eos_angle'] = eos_degree
    result['is_normal_eos'] = is_normal_eos
    result['eos_interpretation'] = eos_interpretation

    return to_serializable(result)


def visualize_lead_with_marks(lead_result, baseline, fs=500, max_seconds=6, save_path=None):
    """
    Визуализирует сигнал конкретного отведения с отметками пиков и интервалов.
    """
    x = np.array(lead_result['timeline_x'])
    y = np.array(lead_result['clean_signal'])

    if max_seconds:
        max_samples = int(max_seconds * fs)
        x = x[:max_samples]
        y = y[:max_samples]
    

    # 1мм по оси X = 0.02 при fs=500, 0.04 при fs=250, 0.01 при fs=1000 (шаг по X)
    x_tick = 0.02 * 500 / fs
    # 1мм по оси Y = 0,1 мВ
    y_tick = 0.1

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(x, y, color='black', linewidth=1.0, label=f"Отведение II")

    # сохраняем соотношение осей (делает секции квадратными)
    ax.set_aspect(x_tick / y_tick, adjustable='box') 

    # диапазон значений на графике
    ax.set_xlim(-1, max_seconds + 1) 
    ax.set_ylim(-2, 2)

    # параметры сетки 1х1 мм
    ax.xaxis.set_minor_locator(MultipleLocator(x_tick))
    ax.yaxis.set_minor_locator(MultipleLocator(y_tick))
    ax.grid(which='minor', color='lightcoral', linewidth=0.4, alpha=0.6)

    # сетка 5х5 мм
    ax.xaxis.set_major_locator(MultipleLocator(x_tick * 5))
    ax.yaxis.set_major_locator(MultipleLocator(y_tick * 5))
    ax.grid(which='major', color='lightcoral', linewidth=0.8, alpha=0.9)

    ax.axhline(y=baseline, linestyle='--', linewidth=1, label='Baseline (TP)')

    colors = {
        'p_peaks': 'purple',
        'q_peaks': 'green',
        'r_peaks': 'red',
        's_peaks': 'blue',
        't_peaks': 'orange',
        'u_peaks': 'magenta',
        'j_points': 'black',
    }
    for key, color in colors.items():
        if key in lead_result:
            peaks = [p for p in lead_result[key] if p is not None and p < len(y)]
            if peaks:
                ax.scatter(np.array(peaks)/fs, y[np.array(peaks)], s=10, color=color, label=key)

    intervals = {
        "QRS": (lead_result.get("qrs_onsets"), lead_result.get("qrs_offsets"), "cyan", "QRS"),
        "ST": (lead_result.get("qrs_offsets"), lead_result.get("j_points"), "pink", "ST"),
    }
    for key, (starts, ends, color, label) in intervals.items():
        if starts and ends:
            for s, e in zip(starts, ends):
                if s and e and e < len(y) and s < e:
                    ax.axvspan(s/fs, e/fs, color=color, alpha=0.3, label=label)

    ax.set_title(f"ЭКГ по отведению II — пиковые точки и интервалы", fontsize=6)
    ax.set_xlabel('Время (с)', fontsize=6)
    ax.set_ylabel('Амплитуда (мВ)', fontsize=6)
    ax.legend(loc='upper left', fontsize=6, bbox_to_anchor=(1, 1))
    ax.tick_params(axis='both', labelsize=4)

    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    folder_path = Path("test_data")
    output_folder = Path("results")
    output_folder.mkdir(exist_ok=True)
    plots_folder = output_folder / "plots"
    plots_folder.mkdir(exist_ok=True)
    patients_csv = folder_path / "PatientInfo.csv"

    json_files = sorted(folder_path.glob("*.json"))

    for file in json_files:
        print(f"Обработка {file.name}")
        try:
            result = process_file(file, patients_csv, method='neurokit')

            # Сохраняем результат
            out_path = output_folder / f"{file.stem}_processed.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"Результат сохранен: {out_path}")

            # Визуализация
            fs = result.get("FS", 500)
            plot_path = plots_folder / f"{file.stem}_leadII.png"
            visualize_lead_with_marks(result['II'], result['baseline'], fs=fs, max_seconds=6, save_path=plot_path) 

            # Текстовое заключение
            hr_values = np.array(result.get("heart_rate", []))
            is_rr_regular = result.get("is_rr_regular", False)
            is_normal_eos = result.get("is_normal_eos", False)
            eos_text = result.get("eos_interpretation", "").capitalize()

            # Частота сердечных сокращений
            if is_rr_regular:
                hr_mean = int(round(np.nanmean(hr_values)))
                hr_text = f"ЧСС {hr_mean}/мин"
            else:
                hr_min = int(round(np.nanmin(hr_values)))
                hr_max = int(round(np.nanmax(hr_values)))
                hr_text = f"ЧСС {hr_min}-{hr_max}/мин"

            # Ритм
            rhythm_text = "Ритм регулярный" if is_rr_regular else "Ритм нерегулярный"

            # ЭОС
            eos_status = "ЭОС не отклонена" if is_normal_eos else "ЭОС отклонена"
            eos_summary = f"{eos_status}. {eos_text}"

            conclusion = f"{rhythm_text}, {hr_text}. {eos_summary}."

            # Сохраняем заключение в txt
            txt_path = output_folder / f"{file.stem}_summary.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(conclusion)
            print(f"{conclusion}")

        except Exception as e:
            print(f"Ошибка при обработке {file.name}: {e}")


if __name__ == "__main__":
    main()