"""Данные ЗСБ аппаратуры «Цикл» (файлы *.pod).

Имя файла: «ПП-ННН<ток>_<дубль>[_2].pod», например 06-220б_1.pod:
профиль 06, пикет 220, «б» — большой ток, «м» — малый, 1 — номер дубля;
дополнительная «_2» — пикет переснят с соседней петли. Встречаются оба
порядка: 01-17б_3_2.pod и 06-225б_2_3.pod; номер дубля определяется как
та цифра, которая меняется между файлами одного пикета.

Формат — формат системы ПОДБОР (Захаркин, Тарло; см. описание «ПРОБА-WIN»):
    стр. 5:  A  lизм  разнос  угол — сторона генераторной петли, сторона
                                      измерительной петли, разнос, угол
    стр. 6:  I·W  A  B            — ампер-витки, размеры генераторной петли
    стр. 7:  Z X Y [Долг Шир Выс]  — координаты
    стр. 8:  число отсчётов, далее «время, мс   ЭДС, мВ»
Кажущееся сопротивление, как в «ПРОБЕ» (поздняя стадия, полупространство):
    ρτ = μ0/(4πt) · (2μ0·A²·lизм²·I·W / (5·t·ε))^(2/3).

Обработка зондирования (все файлы одного пикета):
  1) дубли одного тока складываются (медиана), ошибка — разброс дублей;
  2) сигнал нормируется на ток и площадь рамки: dB/dt/I = ЭДС / (I·S·n);
  3) у каждого тока отбрасываются ранние времена: «полка» насыщения и
     время выключения тока (до t_min, по умолчанию 2 × задержка из файла);
  4) кривые сшиваются как в «ПРОБЕ»: большой ток — главный, малый
     приводится к нему сшивающим коэффициентом на самом согласованном
     5-точечном участке перекрытия и используется до этого участка;
  5) поздние времена, где сигнал меньше snr_min ошибок, исключаются (маска),
     как и всё после первой такой точки.
"""

from __future__ import annotations

import os
import re
from collections import OrderedDict

import numpy as np

from .tem import TEMSounding

UNITS = {"uV": 1e-6, "mV": 1e-3, "V": 1.0}
_NAME_RE = re.compile(r"^(\d+)-(\d+)([бБмМbBmM])((?:_\d+)*)\s*$")


def parse_name(path):
    """(профиль, пикет, ток 'б'/'м', [числа после тока]) или None."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _NAME_RE.match(stem)
    if not m:
        return None
    cur = m.group(3).lower()
    cur = {"b": "б", "m": "м"}.get(cur, cur)
    nums = [int(x) for x in m.group(4).split("_") if x]
    return m.group(1), m.group(2), cur, nums


def _assign_dup_rep(items):
    """items: [(nums, rec)] одного пикета и тока -> [(повтор, rec)]."""
    two = [n for n, _ in items if len(n) >= 2]
    first_const = len({n[0] for n in two}) == 1 and len({n[1] for n in two}) > 1
    out = []
    for n, rec in items:
        if len(n) >= 2:
            rep = n[0] if first_const else n[1]
        else:
            rep = 1
        out.append((rep, rec))
    return out


def read_pod(path):
    raw = open(path, "rb").read()
    for enc in ("cp1251", "utf-8"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    lines = text.splitlines()

    def nums(line):
        out = []
        for p in line.replace(",", ".").split():
            try:
                out.append(float(p))
            except ValueError:
                break
        return out

    geo = nums(lines[4])     # Ltr Lrc r Angle
    cur = nums(lines[5])     # W*I La Lb
    pos = nums(lines[6])     # Z X Y Долг Шир Выс
    n = int(nums(lines[7])[0])
    data = np.array([nums(lines[8 + i])[:2] for i in range(n)], float)
    delay = 0.0
    t_comm = None
    t_recv = None
    for line in lines[8 + n:]:
        v = nums(line)
        if "Общая длительность задержки" in line:
            delay = v[0] * 1e-6 if v else 0.0
        elif "Задержка коммутатора" in line and v:
            t_comm = v[0] * 1e-6       # = половина фронта выключения тока
        elif "Инерц. приемника" in line and v:
            t_recv = v[0] * 1e-6
        m = re.search(r"Tfcommut=([\d.Ee+-]+)", line)
        if m and t_comm is None:
            t_comm = float(m.group(1))
    return dict(
        path=path, area=lines[0].split()[0] if lines[0].split() else "",
        profile=lines[2].split()[0], picket=lines[3].split()[0],
        ltr=geo[0], lrc=geo[1] if len(geo) > 1 else 0.0, r=geo[2] if len(geo) > 2 else 0.0,
        angle=geo[3] if len(geo) > 3 else 0.0, wi=cur[0],
        la=cur[1] if len(cur) > 1 else geo[0], lb=cur[2] if len(cur) > 2 else geo[0],
        z=pos[0] if pos else 0.0, x=pos[1] if len(pos) > 1 else np.nan,
        y=pos[2] if len(pos) > 2 else np.nan, lon=pos[3] if len(pos) > 3 else np.nan,
        lat=pos[4] if len(pos) > 4 else np.nan,
        times=data[:, 0] * 1e-3, values=data[:, 1], delay=delay, t_comm=t_comm, t_recv=t_recv)


def group_files(paths):
    """{(профиль, пикет, повтор): {'б': [rec...], 'м': [rec...]}} в порядке пикетов."""
    raw = {}
    skipped = []
    for p in paths:
        key = parse_name(p)
        try:
            rec = read_pod(p)
        except Exception as e:  # noqa: BLE001
            skipped.append(f"{os.path.basename(p)}: {e}")
            continue
        if key is None:
            # имя не по шаблону — профиль/пикет из шапки, ток определим по величине
            key = (rec["profile"], re.sub(r"\D", "", rec["picket"]), "?", [])
        prof, pk, cur, nums = key
        raw.setdefault((prof, pk, cur), []).append((nums, rec))
    groups = {}
    for (prof, pk, cur), items in raw.items():
        for rep, rec in _assign_dup_rep(items):
            groups.setdefault((prof, pk, rep), {}).setdefault(cur, []).append(rec)
    out = OrderedDict()
    for k in sorted(groups, key=lambda k: (k[0], int(k[1]) if k[1].isdigit() else 0, k[2])):
        g = groups[k]
        if "?" in g:  # разделяем по току
            for r in g.pop("?"):
                g.setdefault("б" if r["wi"] >= 2.0 else "м", []).append(r)
        out[k] = g
    return out, skipped


def _stack(recs, scale):
    """Медиана дублей и ошибка среднего (в единицах dB/dt/I)."""
    t = recs[0]["times"]
    V = np.array([r["values"] for r in recs if len(r["values"]) == len(t)])
    I = np.array([r["wi"] for r in recs if len(r["values"]) == len(t)])
    S = V / I[:, None] * scale
    med = np.median(S, axis=0)
    if len(S) > 1:
        err = np.std(S, axis=0, ddof=1) / np.sqrt(len(S))
    else:
        err = np.full(len(t), np.nan)
    return t, med, err


def _saturation_end(t, v):
    """Конец «полки» насыщения на ранних временах (или 0)."""
    a = np.abs(v)
    if len(a) < 3 or a[0] == 0:
        return 0.0
    k = 0
    while k + 1 < len(a) and abs(a[k + 1] - a[0]) <= 0.03 * a[0]:
        k += 1
    return float(t[k]) if k >= 2 else 0.0


def merge_currents(big, small, t_min=None, t_cross=None, snr_min=2.0, delay=0.0,
                   scale_small=True):
    """big/small = (t, v, err) или None. Возвращает t, v, err, mask, info.

    Сшивка как в «ПРОБЕ»: на перекрытии ищется окно из 5 точек, где кривые
    малого и большого тока наиболее согласованы (минимальный разброс их
    отношения); «главным» считается большой ток, малый умножается на
    сшивающий коэффициент (если scale_small) и используется до этого окна.
    """
    info = {}
    if big is None and small is None:
        raise ValueError("нет данных")
    ref = big if big is not None else small
    t = ref[0]
    # по умолчанию — двойная задержка (коммутатор + приёмник): раньше идёт выключение тока
    t_min = max(2 * delay, 10e-6) if t_min is None else t_min
    sat_b = _saturation_end(*big[:2]) if big is not None else 0.0
    sat_s = _saturation_end(*small[:2]) if small is not None else 0.0
    info.update(sat_big=sat_b, sat_small=sat_s, t_min=t_min, coef=1.0)
    big_start = max(t_min, 1.5 * sat_b) if sat_b > 0 else t_min
    small_start = max(t_min, 1.5 * sat_s) if sat_s > 0 else t_min

    if big is not None and small is not None:
        vb, vs = big[1], small[1]
        es = np.where(np.isfinite(small[2]), small[2], 0.0)
        both = (t >= big_start) & (t >= small_start) & (vs != 0) & (vb != 0) & \
            (np.sign(vb) == np.sign(vs)) & (np.abs(vs) > 3 * es)
        lr = np.where(both, np.log(np.abs(vb) / np.where(vs == 0, 1, np.abs(vs))), np.nan)
        coef = 1.0
        if t_cross is None:
            best = None
            for i in range(len(t) - 4):
                w = lr[i:i + 5]
                if np.all(np.isfinite(w)):
                    sd = float(np.std(w))
                    if best is None or sd < best[0]:
                        best = (sd, i, float(np.mean(w)))
            if best is not None:
                t_cross = float(t[best[1]])
                coef = float(np.exp(best[2]))
            else:
                t_cross = max(3 * sat_b, big_start)
        else:
            k = np.where(np.isfinite(lr) & (t >= t_cross))[0][:5]
            if len(k):
                coef = float(np.exp(np.mean(lr[k])))
        if not scale_small:
            coef = 1.0
        info["coef"] = coef
        use_small = t < t_cross
        v = np.where(use_small, coef * vs, vb)
        e = np.where(use_small, coef * small[2], big[2])
        valid = np.where(use_small, t >= small_start, t >= big_start)
    else:
        v, e = ref[1], ref[2]
        valid = t >= (big_start if big is not None else small_start)
    info["t_cross"] = t_cross

    # шум: |v| < snr_min*err; всё после первой «шумовой» точки — тоже в маску
    noisy = np.isfinite(e) & (np.abs(v) < snr_min * e)
    mask = ~valid.copy()
    first_noise = np.where(noisy & valid)[0]
    if len(first_noise):
        mask[first_noise[0]:] = True
    mask |= ~np.isfinite(v) | (v == 0)
    return t, v, e, mask, info


RX_MODES = {
    "loop": "Lrc — сторона квадратной приёмной рамки, м (площадь = Lrc² × витки)",
    "area": "Lrc — эффективная площадь датчика, м² (датчик — точка)",
}


def build_soundings(paths, units="mV", rx_turns=1.0, t_min=None, t_cross=None, snr_min=2.0,
                    ramp=0.0, rx_mode="loop", scale_small=True):
    """Список TEMSounding из файлов *.pod (по одному на пикет) и список пропусков.

    units — единицы ЭДС в файле (uV/mV/V); rx_mode — как понимать Lrc (RX_MODES).
    ramp — длительность фронта выключения тока, с: None — из файла для каждого тока
    (2 × «Задержка коммутатора»), время при этом отсчитывается от середины фронта;
    0 — мгновенное выключение.
    scale_small — подгонять малый ток к большому коэффициентом (как «ПРОБА»).
    По умолчанию — как в «ПРОБЕ»: мгновенное выключение и сшивающий коэффициент.
    """
    groups, skipped = group_files(paths)
    out = []
    for (prof, pk, rep), g in groups.items():
        recs = g.get("б", []) + g.get("м", [])
        if not recs:
            continue
        r0 = recs[0]
        if rx_mode == "area":
            rx_side = 0.0
            area = max(r0["lrc"], 1e-9) * rx_turns
        else:
            rx_side = r0["lrc"]
            area = rx_side ** 2 * rx_turns if rx_side > 0 else rx_turns
        scale = UNITS[units] / area
        big = _stack(g["б"], scale) if g.get("б") else None
        small = _stack(g["м"], scale) if g.get("м") else None
        t, v, e, mask, info = merge_currents(big, small, t_min, t_cross, snr_min, r0["delay"],
                                             scale_small)
        ramps = None
        if ramp is None:
            def front(recs_):
                tc = [r["t_comm"] for r in recs_ if r.get("t_comm")]
                return 2 * float(np.median(tc)) if tc else 0.0
            fb = front(g.get("б", [])) if g.get("б") else None
            fs = front(g.get("м", [])) if g.get("м") else None
            tcr = info.get("t_cross")
            if fb is not None and fs is not None and tcr is not None:
                ramps = np.where(t < tcr, fs, fb)
            else:
                ramps = np.full(len(t), fb if fb is not None else (fs or 0.0))
            # точки внутри фронта (t < фронт/2) моделировать нельзя
            mask = mask | (t <= ramps / 2 + 1e-6)
            info["fronts"] = dict(big=fb, small=fs)
        name = f"{prof}-{pk}" + (f"_{rep}" if rep > 1 else "")
        lat = float(np.median([r["lat"] for r in recs]))
        lon = float(np.median([r["lon"] for r in recs]))
        snd = TEMSounding(
            name=name, lat=lat, lon=lon, times=t, signal=v, err=e, tx_side=r0["la"] or r0["ltr"],
            tx_b=r0["lb"] or r0["ltr"], rx_side=rx_side, offset=r0["r"],
            offset_angle=r0["angle"], ramp=ramp or 0.0, elev=float(np.median([r["z"] for r in recs])),
            path=r0["path"], mask=mask,
            ramps=ramps,
            opts=dict(format="pod", files=[r["path"] for r in recs], units=units,
                      rx_turns=rx_turns, t_min=t_min, t_cross=t_cross, snr_min=snr_min,
                      ramp=ramp, rx_mode=rx_mode, scale_small=scale_small))
        snd.raw = dict(big=big, small=small, info=info,
                       currents={k: [r["wi"] for r in g.get(k, [])] for k in ("б", "м")},
                       n_dup={k: len(g.get(k, [])) for k in ("б", "м")})
        out.append(snd)
    return out, skipped
