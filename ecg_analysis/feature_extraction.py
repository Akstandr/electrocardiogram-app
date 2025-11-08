import json
from pathlib import Path
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import neurokit2 as nk
from math import atan2, degrees, floor
from scipy.signal import find_peaks


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


def estimate_u_wave(signal, baseline, t_offset_idx, next_p_onset_idx, fs=500):
    ''' Находит зубец U (при наличии) '''

    # амплитуда зубца u - 0.1-0.33мВ
    min_height = 0.1 + baseline 
    max_height = 0.33 + baseline

    # продолжительность зубца u - 0.08-0.24c
    min_width = 500 / fs * 4 
    max_width = 500 / fs * 12

    start = t_offset_idx # зубец u может быть наложен на зубец t

    if next_p_onset_idx is not None:
        end = next_p_onset_idx - 1 
    else:
        end = len(signal)-1 # в конце отведения нет зубца p
    seg = signal[start:end+1]

    # ищем небольшой пик в том же направлении, что и T (если T положител., ищем положит.)
    peaks, props = find_peaks(seg, height=[min_height, max_height], width=[min_width, max_width])
    if peaks.size == 0:
        # пробуем обратную полярность
        peaks, props = find_peaks(-seg, height=[-min_height, -max_height], width=[min_width, max_width])
        if peaks.size == 0:
            return None
        else:
            return start + int(peaks[0])
    return start + int(peaks[0])


def estimate_onset_offset(signal, peak_idx, baseline, border_idx=None, side='left', fs=500, threshold_frac=0.05, max_s=0.08):
    """
    Оценка начала (onset) или конца (offset) зубца относительно базовой линии.

    signal: np.array или list - ЭКГ сигнал
    peak_idx: int - индекс пика зубца 
    border_idx - индекс начала пика следующего за текущим при наличии
    (для комплекса qrs - колнец p при проходе влево, начало t - при проходе вправо)
    baseline: float - значение базовой линии (TP segment)
    side: 'left' -> onset, 'right' -> offset
    fs: частота дискретизации
    threshold_frac: доля амплитуды от пика относительно baseline для определения окончания зубца
    max_s: максимальный поиск в секундах 
    """
    n = len(signal)
    peak_amp = abs(signal[peak_idx] - baseline)  # амплитуда относительно baseline 
    
    thresh = threshold_frac * peak_amp # доля амплитуды пика, которую мы считаем достаточным приближением к baseline для окончания поиска
    max_samples = int(max_s * fs) # количество индексов для обхода по заданному максимуму секунд
    if border_idx is not None:
        max_samples = min(abs(peak_idx - border_idx), max_samples) # ограничиваем обход по началу следующего пика
    if side == 'left':
        for step in range(1, max_samples + 1): 
            cur = peak_idx - step
            # когда сигнал становится близок к baseline возвращаем индекс
            if abs(signal[cur] - baseline) < thresh:
                return cur
        return max(0, peak_idx - max_samples)
    else:
        for step in range(1, max_samples + 1):
            cur = peak_idx + step
            if abs(signal[cur] - baseline) < thresh:
                return cur
        return min(n - 1, peak_idx + max_samples)


def calculate_eos(lead_I, lead_aVF, q_peaks, r_peaks, s_peaks):
    """ Расчёт электрической оси сердца (ЭОС) """

    def peak_amplitude(signal, peaks):
        return np.mean([signal[p] for p in peaks]) if len(peaks) else 0

    # амплитуды зубцов
    A_Q_I = peak_amplitude(lead_I, q_peaks)
    A_R_I = peak_amplitude(lead_I, r_peaks)
    A_S_I = peak_amplitude(lead_I, s_peaks)

    A_Q_aVF = peak_amplitude(lead_aVF, q_peaks)
    A_R_aVF = peak_amplitude(lead_aVF, r_peaks)
    A_S_aVF = peak_amplitude(lead_aVF, s_peaks)

    # алгебраическая сумма амплитуд
    A_I = A_R_I + A_Q_I + A_S_I
    A_aVF = A_R_aVF + A_Q_aVF + A_S_aVF

    # вычисление угла
    alpha = np.degrees(np.arctan2(A_aVF, A_I))

    is_normal_eos = False 
    # классификация
    if 0 <= alpha <= 90:
        is_normal_eos = True 
        interpretation = 'ЭОС в норме'
    elif -30 <= alpha <= 0:
        interpretation = 'ЭОС отклонена влево'
    elif -90 <= alpha <= -30:
        interpretation = 'ЭОС резко отклонена влево'    
    elif 90 <= alpha <= 120:
        interpretation = 'ЭОС отклонена вправо'
    else:
        interpretation = 'ЭОС резко отклонена вправо'

    return alpha, is_normal_eos, interpretation


def identify_rhythm(p_peaks_dict, qrs_count, is_rr_regular, baseline):
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
    p_peaks_aVF = p_peaks_dict['aVF']

    is_sin = False 

    # критерий синусового ритма: зубцы в отведениях I-II, V2-V6 и aVF положительны, в aVR отрицательны
    # в остальных могут быть разными 
    all_peaks = np.concatenate([p_peaks_I, p_peaks_II, p_peaks_aVF, p_peaks_V2,
                           p_peaks_V3, p_peaks_V4, p_peaks_V5, p_peaks_V6]) - baseline 
    is_positive_required_peaks = np.all(all_peaks > 0)

    is_negative_required_peaks = np.all(np.array(p_peaks_aVR) - baseline < 0) 

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
        # медиана по всему сигналу
        return float(np.median(lead_ii))

    all_tp = np.concatenate(tp_segments)
    return float(np.median(all_tp))


def calculate_intervals(r_peaks, q_peaks, p_peaks, t_peaks, fs=500):
    """
    Вычисляет интервалы RR, PP, PQ (PR если Q нет), QT, TP в миллисекундах
    и возвращает словарь словарей с их началами, концами, длительностями
    """
    def to_ms(samples):
        return samples / fs * 1000.0

    result = {
        "RR": {"onsets": [], "offsets": [], "durations_ms": []},
        "PP": {"onsets": [], "offsets": [], "durations_ms": []},
        "PQ": {"onsets": [], "offsets": [], "durations_ms": []},
        "QT": {"onsets": [], "offsets": [], "durations_ms": []},
        "TP": {"onsets": [], "offsets": [], "durations_ms": []},
    }

    # RR интервалы
    if len(r_peaks) >= 2:
        onsets = r_peaks[:-1]
        offsets = r_peaks[1:]
        durations = to_ms(np.diff(r_peaks))
        result["RR"]["onsets"] = onsets.tolist()
        result["RR"]["offsets"] = offsets.tolist()
        result["RR"]["durations_ms"] = durations.tolist()

    # PP 
    if len(p_peaks) >= 2:
        onsets = p_peaks[:-1]
        offsets = p_peaks[1:]
        durations = to_ms(np.diff(p_peaks))
        result["PP"]["onsets"] = onsets.tolist()
        result["PP"]["offsets"] = offsets.tolist()
        result["PP"]["durations_ms"] = durations.tolist()

    # PQ (PR если Q нет)
    for p in p_peaks:
        qs_after = [q for q in q_peaks if q > p]
        if qs_after:
            q_index = qs_after[0]
        else:
            rs_after = [r for r in r_peaks if r > p]
            q_index = rs_after[0] if rs_after else None
        if q_index:
            result["PQ"]["onsets"].append(p)
            result["PQ"]["offsets"].append(q_index)
            result["PQ"]["durations_ms"].append((q_index - p) / fs * 1000.0)

    # QT
    for q in q_peaks:
        ts_after = [t for t in t_peaks if t > q]
        if ts_after:
            t_index = ts_after[0]
            result["QT"]["onsets"].append(q)
            result["QT"]["offsets"].append(t_index)
            result["QT"]["durations_ms"].append((t_index - q) / fs * 1000.0)

    # TP
    for t in t_peaks:
        ps_after = [p for p in p_peaks if p > t]
        if ps_after:
            p_index = ps_after[0]
            result["TP"]["onsets"].append(t)
            result["TP"]["offsets"].append(p_index)
            result["TP"]["durations_ms"].append((p_index - t) / fs * 1000.0)

    return result


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


def process_signals(signal, fs, correct_artifacts=False, find_hr=False):
    ''' Процесс нахождения пиков и границ зубцов на сигнале ЭКГ
    input: signal - сигнал ЭКГ в минивольтах, 
    '''
    
    cleaned_signal = nk.ecg_clean(signal, sampling_rate=fs, method='neurokit')
    signals, info = nk.ecg_peaks(cleaned_signal, sampling_rate=fs, method='khamis2016', correct_artifacts=correct_artifacts)
    r_peaks = signals['ECG_R_Peaks']
    r_peaks_info = info['ECG_R_Peaks']
    waves, signals = nk.ecg_delineate(cleaned_signal, r_peaks_info, sampling_rate=fs, method='prominence', check=False)

    result = {
        'q_peaks': waves['ECG_Q_Peaks'],
        'r_peaks': r_peaks,
        'p_onsets': waves['ECG_P_Onsets'],
        'p_peaks': waves['ECG_P_Peaks'],
        's_peaks': waves['ECG_S_Peaks'],
        't_peaks': waves['ECG_T_Peaks'],
        't_offsets': waves['ECG_T_Offsets'],
        'p_onsets': waves['ECG_P_Onsets']
    }

    if find_hr:
        heart_rate = nk.ecg_rate(info['ECG_R_Peaks'], sampling_rate=fs, interpolation_method='monotone_cubic')
        return heart_rate, cleaned_signal, result
    return cleaned_signal, result



def process_file(json_path, patients_csv=None, method='neurokit'):
    """
    Полный процесс обработки одного файла:
    добавить длины сегментов + амплитуды зубцов + синусоидность ритма + форму QRS
    """
    def mask_to_idx(series):
        arr = np.array(series)
        return np.where(arr == 1)[0]
    
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
    signal = leads_data['II']['Y']
    heart_rate, cleaned_signal, waves = process_signals(signal, fs, correct_artifacts=False, find_hr=True)

    r_peaks = mask_to_idx(waves['r_peaks'])
    p_peaks = mask_to_idx(waves['p_peaks'])
    s_peaks = mask_to_idx(waves['s_peaks'])
    q_peaks = mask_to_idx(waves['q_peaks'])
    p_onsets = mask_to_idx(waves['p_onsets'])
    t_peaks = mask_to_idx(waves['t_peaks'])
    t_offsets =  mask_to_idx(waves['t_offsets']) 

    baseline = calculate_baseline(t_offsets, p_onsets, cleaned_signal)

    intervals = calculate_intervals(r_peaks, q_peaks, p_peaks, t_peaks)
    rr_intervals_ms, pp_intervals_ms, pq_intervals_ms = intervals["RR"]["durations_ms"], intervals["PP"]["durations_ms"], intervals["PQ"]["durations_ms"]
    is_rr_regular = identify_rhythm_regularity(rr_intervals_ms)
    is_pp_regular = identify_rhythm_regularity(pp_intervals_ms)
    is_pq_regular = identify_rhythm_regularity(pq_intervals_ms)

    qrs_count = len(r_peaks)

    if is_rr_regular:
        heart_rate = int(round(np.nanmean(heart_rate)))
    else:
        hr_min = int(round(np.nanmin(heart_rate)))
        hr_max = int(round(np.nanmax(heart_rate)))
        heart_rate = [hr_min, hr_max]

    # если onsets/offsets не найдены — оценим по пикам
    qrs_onsets = []
    qrs_offsets = []

    t_onsets = []
    p_offsets = []

    for tp in t_peaks: 
        on = estimate_onset_offset(cleaned_signal, tp, baseline, side='left', fs=fs, threshold_frac=0.01)
        t_onsets.append(int(on))
        
    for pp in p_peaks:
        off = estimate_onset_offset(cleaned_signal, pp, baseline, side='right', fs=fs, threshold_frac=0.01)
        p_offsets.append(int(off))

    for rp in r_peaks:
        q_left = [q for q in q_peaks if q <= rp]
        q_idx = q_left[-1] if q_left else rp

        p_before = [p for p in p_offsets if p < rp]
        border_left = p_before[-1] if p_before else None

        on = estimate_onset_offset(cleaned_signal, q_idx, baseline, border_idx=border_left, side='left', fs=fs, threshold_frac=0.01)

        s_right = [s for s in s_peaks if s >= rp]
        s_idx = s_right[0] if s_right else rp 

        t_after = [t for t in t_onsets if t > rp]
        border_right = t_after[0] if t_after else None

        off = estimate_onset_offset(cleaned_signal, s_idx, baseline, border_idx=border_right, side='right', fs=fs, threshold_frac=0.01)

        qrs_onsets.append(int(on))
        qrs_offsets.append(int(off))

    # j_point - (начало сегмента ST)
    j_points = qrs_offsets

    # зубец U
    u_peaks = []
    for i, off in enumerate(t_offsets):
        # след p_onset после t_offset
        next_p = None
        for p_on in p_onsets:
            if p_on > off:
                next_p = p_on
                break
        u = estimate_u_wave(cleaned_signal, baseline, off, next_p, fs=fs)
        u_peaks.append(int(u) if u is not None else None) 
    
    # QRS продолжительности
    qrs_durations_ms = []
    for on, off in zip(qrs_onsets, qrs_offsets):
        qrs_durations_ms.append((off - on)/fs*1000.0)

    result = {
        'filename': str(json_path.name),
        'FS': int(fs),
        'patient_info': patient_info,
        'baseline': baseline,
        'heart_rate': heart_rate,
        'intervals': intervals,
        'is_rr_regular': is_rr_regular,
        'is_pp_regular': is_pp_regular,
        'is_pq_regular': is_pq_regular,
        'timeline_x': leads_data['II']['X'],
        'r_peaks': r_peaks,
        'p_peaks': p_peaks,
        'q_peaks': q_peaks,
        's_peaks': s_peaks,
        't_peaks': t_peaks,
        't_onsets': t_onsets,
        't_offsets': t_offsets,
        'p_onsets': p_onsets,
        'p_offsets': p_offsets,
        "u_peaks": [int(u) if u is not None else None for u in u_peaks],
        "qrs_onsets": qrs_onsets,
        "qrs_offsets": qrs_offsets,
        "j_points": j_points,
        "qrs_durations_ms": qrs_durations_ms,
        'cycles_count': len(r_peaks),
        'leads_results': {}

    }

    p_peaks_dict = {}

    for lead_name in lead_names:
        lead = leads_data[lead_name]['Y']
        cleaned_signal = nk.ecg_clean(lead, sampling_rate=fs, method='neurokit')
        p_peaks_dict[lead_name] = [cleaned_signal[int(idx)] for idx in p_peaks]
        # амплитуда j относительно бейзлайна
        st_at_j = []
        for j in j_points:
            if j is None:
                st_at_j.append(None)
            else:
                st_at_j.append(float(cleaned_signal[j] - baseline))

        # агрегируем результат по текущему отведению
        lead_dict = {
            'raw_signal': lead.tolist(),
            'clean_signal': cleaned_signal,
            'timeline_x': leads_data[lead_name]['X'],
            'st_at_j_mv': st_at_j,
        }
        result['leads_results'][lead_name] = lead_dict 

    # считаем ЭОС
    clean_lead_I = result['leads_results']['I']['clean_signal']
    clean_lead_aVF = result['leads_results']['aVF']['clean_signal']
    eos_degree, is_normal_eos, eos_interpretation = calculate_eos(clean_lead_I, clean_lead_aVF, 
                                                                    q_peaks, r_peaks, s_peaks)
    result['eos_angle'] = eos_degree
    result['is_normal_eos'] = is_normal_eos
    result['eos_interpretation'] = eos_interpretation 

    result['is_sin'] = identify_rhythm(p_peaks_dict, qrs_count, is_rr_regular, baseline)

    return to_serializable(result)


def visualize_ecg(result, fs=500, save_path='ecg.json'):
    """
    Интерактивная визуализация ЭКГ с миллиметровой сеткой
    """
    fig = go.Figure()

    time = result['timeline_x']

    x_tick = 0.02 * 500 / fs # 1мм по x
    y_tick = 0.1 # 1мм по y

    colors = {'p_peaks': 'purple',
              'q_peaks': 'green',
              'r_peaks': 'red',
              's_peaks': 'blue',
              't_peaks': 'orange',
              'u_peaks': 'magenta',
              'j_points': 'black'}

    peaks = {
        'p_peaks': result['p_peaks'],
        'q_peaks': result['q_peaks'],
        'r_peaks': result['r_peaks'],
        's_peaks': result['s_peaks'],
        't_peaks': result['t_peaks'],
        'u_peaks': result['u_peaks'],
        'j_points': result['j_points']
    }

    baseline = result['baseline']

    for i, (lead_name, lead_result) in enumerate(result['leads_results'].items()):
        offset = -i * 2 # расстояние между отведениями
        y = np.array(lead_result['clean_signal']) + offset
        fig.add_trace(go.Scatter(
            x=time,
            y=y,
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ))

        for key, color in colors.items():
            lead_peaks = [p/fs for p in peaks[key] if p is not None]
            fig.add_trace(go.Scatter(
                x=lead_peaks,
                y=[y[int(p*fs)] for p in lead_peaks if int(p*fs) < len(y)],
                mode='markers',
                marker=dict(color=color, size=6, symbol="circle"),
                name=f"{key}",
                showlegend=(i == 0)
            ))
        
        fig.add_hline(y=baseline + offset, line_color='gray', line_dash='dash',
                annotation_text="baseline", annotation_position="bottom left")
        
        fig.add_annotation(
            x=0,
            y=y[0] + 0.1,
            text=f"{lead_name}",
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
            font=dict(size=10)
        )

    for key, color in colors.items():
        lead_peaks = [p/fs for p in peaks[key] if p is not None]
        for peak in lead_peaks:
            fig.add_vline(x=peak, line_color=color)

    aspect_ratio = x_tick / y_tick

    fig.update_xaxes(
        showgrid=True,
        gridcolor="red",
        gridwidth=0.4,
        dtick=x_tick*5,
        minor=dict(dtick=x_tick, showgrid=True, gridcolor="lightcoral", gridwidth=0.2),
        zeroline=False,
        showticklabels=False
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="red",
        gridwidth=0.4,
        dtick=y_tick*5,
        minor=dict(dtick=y_tick, showgrid=True, gridcolor="lightcoral", gridwidth=0.2),
        zeroline=False,
        scaleanchor="x", 
        scaleratio=aspect_ratio,
        showticklabels=False
    )

    fig.update_layout(
        height=800,
        width=1200,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    graph_json = fig.to_json()

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(graph_json)

    return graph_json


def get_conclusion(result):
    hr = np.array(result.get("heart_rate", []))
    is_rr_regular = result.get("is_rr_regular", False)
    is_normal_eos = result.get("is_normal_eos", False)
    eos_text = result.get("eos_interpretation", "").capitalize()
    eos_angle = result.get("eos_angle")
    is_sin = result.get('is_sin')

    rhythm_text = "Ритм синусовый, " if is_sin else "Ритм эктопический (не синусовый), "
    if is_rr_regular:
        hr_text = f"ЧСС {hr}/мин"
        rhythm_text += "регулярный"
    else:
        hr_text = f"ЧСС {hr[0]}-{hr[1]}/мин"
        rhythm_text += "нерегулярный"

    eos_status = "ЭОС не отклонена" if is_normal_eos else "ЭОС отклонена"
    eos_summary = f"{eos_status}. {eos_text} (угол α = {round(eos_angle, 1)})"

    # зубцы
    wave_types = {
        "P": ("p_peaks", "p_onsets", "p_offsets"),
        "Q": ("q_peaks", "q_onsets", "q_offsets"),
        "R": ("r_peaks", "r_onsets", "r_offsets"),
        "S": ("s_peaks", "s_onsets", "s_offsets"),
        "T": ("t_peaks", "t_onsets", "t_offsets"),
        "U": ("u_peaks", "u_onsets", "u_offsets"),
        "QRS-комплекс": ("", "qrs_onsets", "qrs_offsets"),
    }

    clean_signal = result['leads_results']['II']['clean_signal']
    baseline = result['baseline']

    # таблица по зубцам
    waves_table = []

    for name, (peaks_key, onsets_key, offsets_key) in wave_types.items():
        peaks = result.get(peaks_key) or []
        onsets = result.get(onsets_key) or []
        offsets = result.get(offsets_key) or []

        if onsets and offsets:
            if peaks:
                amps = [round(clean_signal[idx] - baseline, 3) for idx in peaks]
                mean_amplitude = round(float(np.mean(amps)), 3)
            else:
                amps = '-'
                mean_amplitude = '-'
            durations = [round((offsets[i] - onsets[i]) * 0.002, 3) for i in range(len(onsets))]
            ranges = [f"{round(onsets[i] * 0.002, 3)}–{round(offsets[i] * 0.002, 3)}" for i in range(len(onsets))]

            waves_table.append({
                "wave": name,
                "amplitudes": amps,
                "mean_amplitude": mean_amplitude,
                "durations": durations,
                "mean_duration": round(float(np.mean(durations)), 3),
                "ranges": ranges
            })
        else:
            waves_table.append({
                "wave": name,
                "amplitudes": [],
                "mean_amplitude": 0,
                "durations": [],
                "mean_duration": 0,
                "ranges": []
            })

    # таблица по интервалам
    intervals_data = result.get("intervals", {})
    intervals_table = []

    for name, values in intervals_data.items():
        durations = values.get("durations_ms") or []
        onsets = values.get("onsets") or []
        offsets = values.get("offsets") or []

        ranges = [f"{round(onsets[i] * 0.002, 3)}–{round(offsets[i] * 0.002, 3)}" for i in range(len(onsets))]

        intervals_table.append({
            "interval": name,
            "durations_ms": durations,
            "mean_duration_ms": round(float(np.mean(durations)), 3),
            "ranges": ranges
        })

    # ST
    st_table = []
    for lead_name, lead_data in result['leads_results'].items():
        st_at_j = lead_data.get('st_at_j_mv') or []
        st_values = [round(v, 3) for v in st_at_j if v is not None]

        st_table.append({
            "lead": lead_name,
            "ST_elevations_at_J": st_values, 
            "mean_ST_elevation": round(float(np.mean(st_values)), 3) if st_values else 0
        })

    # длительность ST
    st_onsets = result.get("qrs_offsets") or []
    st_offsets = result.get("t_onsets") or []
    durations_ST = [round((st_offsets[i] - st_onsets[i]) * 0.002, 3) for i in range(len(st_onsets))] if st_onsets and st_offsets else []
    mean_duration_ST = float(np.mean(durations_ST)) if durations_ST else 0

    st_durations_text = ", ".join(str(d) for d in durations_ST) if durations_ST else "отсутствуют"

    # текст
    conclusion_text = (
        f"{hr_text}. {rhythm_text}. {eos_summary}.\n"
        + f"Сегмент ST в отведении II: продолжительности {st_durations_text} сек., "
        f"средняя продолжительность {mean_duration_ST:.3f} сек." 
    )

    return {
        "text": conclusion_text,
        "waves_table": waves_table,
        "intervals_table": intervals_table,
        "st_table": st_table
    }


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
            plot_path = plots_folder / f"{file.stem}_plot.json"
            plot_json = visualize_ecg(result, fs, plot_path)

            # Текстовое заключение
            conclusion = get_conclusion(result)['text']

            # Сохраняем заключение в txt
            txt_path = output_folder / f"{file.stem}_summary.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(conclusion)
            print(f"{conclusion}")

        except Exception as e:
            print(f"Ошибка при обработке {file.name}: {e}")


if __name__ == "__main__":
    main()