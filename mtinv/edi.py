"""Чтение и запись EDI-файлов (SEG MT/EMAP).

Поддерживается:
  * импеданс  >ZXXR/>ZXXI/>ZXX.VAR ... (в том числе с суффиксами ROT=..., //n);
  * типпер    >TXR.EXP/>TXI.EXP/>TXVAR.EXP, >TYR.EXP ...;
  * только кажущиеся сопротивления и фазы >RHOXY/>PHSXY (+ .ERR) — импеданс
    восстанавливается из них;
  * координаты LAT/LONG/ELEV из >HEAD или REFLAT/REFLONG/REFELEV из
    >=DEFINEMEAS, как в градусах, так и в формате Г:М:С.
Блоки спектров (>SPECTRA) не поддерживаются.
"""

from __future__ import annotations

import os
import re

import numpy as np

from .mtdata import Station

_NODATA = 1.0e30


def _parse_angle(val: str) -> float | None:
    val = val.strip().strip('"').strip("'").strip()
    if not val:
        return None
    if ":" in val:
        parts = val.split(":")
        sign = -1.0 if parts[0].strip().startswith("-") else 1.0
        nums = [abs(float(p)) for p in parts if p.strip() != ""]
        deg = nums[0] + (nums[1] / 60.0 if len(nums) > 1 else 0.0) + (
            nums[2] / 3600.0 if len(nums) > 2 else 0.0)
        return sign * deg
    try:
        return float(val)
    except ValueError:
        return None


def _parse_float(val: str) -> float | None:
    try:
        return float(val.strip().strip('"').strip("'"))
    except ValueError:
        return None


_num_re = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")


def _split_blocks(text: str):
    """Разбивает файл на блоки (ключевое слово, строка заголовка, тело)."""
    blocks = []
    cur = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(">"):
            if cur is not None:
                blocks.append(cur)
            head = s[1:].strip()
            if head.startswith("!"):  # комментарий вида >!****!
                cur = None
                continue
            kw = re.split(r"[\s/]", head, maxsplit=1)[0].upper()
            cur = [kw, head, []]
        elif cur is not None:
            cur[2].append(line)
    if cur is not None:
        blocks.append(cur)
    return blocks


def _kv_pairs(lines):
    out = {}
    for line in lines:
        for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(\"[^\"]*\"|'[^']*'|\S+)", line):
            out[m.group(1).upper()] = m.group(2).strip('"').strip("'")
    return out


def _numbers(lines):
    txt = " ".join(lines).replace("D", "E").replace("d", "e")
    return np.array([float(x) for x in _num_re.findall(txt)], dtype=float)


def read_edi(path: str) -> Station:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    blocks = _split_blocks(text)

    header = {}
    arrays = {}
    for kw, head, body in blocks:
        if kw in ("HEAD", "INFO", "=DEFINEMEAS", "=MTSECT"):
            kv = _kv_pairs([head] + body if kw != "INFO" else [head])
            for k, v in kv.items():
                header.setdefault(k, v)
            continue
        if kw in ("END", "EMEAS", "HMEAS", "=SPECTRASECT", "SPECTRA"):
            continue
        arrays[kw] = _numbers(body)

    name = header.get("DATAID") or os.path.splitext(os.path.basename(path))[0]
    lat = _parse_angle(header.get("LAT", "")) if "LAT" in header else None
    lon = None
    for k in ("LONG", "LON"):
        if k in header:
            lon = _parse_angle(header[k])
            break
    if lat is None and "REFLAT" in header:
        lat = _parse_angle(header["REFLAT"])
    if lon is None:
        for k in ("REFLONG", "REFLON"):
            if k in header:
                lon = _parse_angle(header[k])
                break
    elev = None
    for k in ("ELEV", "REFELEV"):
        if k in header:
            elev = _parse_float(header[k])
            break

    if "FREQ" not in arrays:
        if "SPECTRA" in text.upper():
            raise ValueError("EDI содержит только спектры (>SPECTRA) — пока не поддерживается")
        raise ValueError("в файле нет блока >FREQ")
    freq = arrays["FREQ"]
    nf = len(freq)

    def get(key):
        a = arrays.get(key)
        if a is None or len(a) < nf:
            return None
        a = a[:nf].copy()
        a[np.abs(a) >= _NODATA] = np.nan
        return a

    Z = np.full((nf, 2, 2), np.nan + 1j * np.nan, dtype=complex)
    Zerr = np.full((nf, 2, 2), np.nan)
    comps = {"XX": (0, 0), "XY": (0, 1), "YX": (1, 0), "YY": (1, 1)}
    have_z = False
    for c, (i, j) in comps.items():
        re_ = get(f"Z{c}R")
        im_ = get(f"Z{c}I")
        if re_ is not None and im_ is not None:
            have_z = True
            Z[:, i, j] = re_ + 1j * im_
            var = get(f"Z{c}.VAR")
            if var is not None:
                Zerr[:, i, j] = np.sqrt(np.abs(var))
            else:
                std = get(f"Z{c}.STD") if f"Z{c}.STD" in arrays else None
                if std is not None:
                    Zerr[:, i, j] = np.abs(std)

    if not have_z:
        # восстанавливаем импеданс из rho/phase
        for c, (i, j) in comps.items():
            rho = get(f"RHO{c}")
            phs = get(f"PHS{c}")
            if rho is None or phs is None:
                continue
            have_z = True
            amp = np.sqrt(np.abs(rho) * 5.0 * freq)  # rho = 0.2 T |Z|^2
            Z[:, i, j] = amp * np.exp(1j * np.deg2rad(phs))
            rerr = get(f"RHO{c}.ERR")
            perr = get(f"PHS{c}.ERR")
            e = np.full(nf, np.nan)
            if rerr is not None:
                e = 0.5 * np.abs(rerr) / np.abs(rho) * amp
            elif perr is not None:
                e = np.deg2rad(np.abs(perr)) * amp
            Zerr[:, i, j] = e
    if not have_z:
        raise ValueError("не найдено ни импеданса (>ZXYR...), ни >RHOXY/>PHSXY")

    T = np.full((nf, 2), np.nan + 1j * np.nan, dtype=complex)
    Terr = np.full((nf, 2), np.nan)
    have_t = False
    for k, c in enumerate(("X", "Y")):
        re_ = get(f"T{c}R.EXP") if f"T{c}R.EXP" in arrays else get(f"T{c}R")
        im_ = get(f"T{c}I.EXP") if f"T{c}I.EXP" in arrays else get(f"T{c}I")
        if re_ is not None and im_ is not None:
            have_t = True
            T[:, k] = re_ + 1j * im_
            var = get(f"T{c}VAR.EXP") if f"T{c}VAR.EXP" in arrays else get(f"T{c}.VAR")
            if var is not None:
                Terr[:, k] = np.sqrt(np.abs(var))

    zrot = get("ZROT")
    st = Station(
        name=str(name),
        lat=lat if lat is not None else np.nan,
        lon=lon if lon is not None else np.nan,
        elev=elev if elev is not None else 0.0,
        freq=freq,
        Z=Z,
        Zerr=Zerr,
        T=T if have_t else None,
        Terr=Terr if have_t else None,
        path=path,
        zrot=float(np.nanmedian(zrot)) if zrot is not None and np.any(np.isfinite(zrot)) else 0.0,
    )
    st.sort()
    return st


def _fmt_block(values, per_line=6):
    vals = np.where(np.isfinite(values), values, 1.0e32)
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append(" ".join(f"{v: .6E}" for v in vals[i:i + per_line]))
    return "\n".join(lines)


def _dms(val: float) -> str:
    sign = "-" if val < 0 else ""
    tot = round(abs(val) * 3600 * 100)  # сотые доли секунды
    d, rem = divmod(tot, 3600 * 100)
    m, s = divmod(rem, 60 * 100)
    return f"{sign}{d}:{m:02d}:{s / 100:05.2f}"


def write_edi(st: Station, path: str):
    nf = len(st.freq)
    out = [">HEAD", f'  DATAID="{st.name}"', f"  LAT={_dms(st.lat)}",
           f"  LONG={_dms(st.lon)}", f"  ELEV={st.elev:.1f}", '  PROGVERS="MT-Inv"', "",
           ">INFO", "", ">=DEFINEMEAS", f"  REFLAT={_dms(st.lat)}", f"  REFLONG={_dms(st.lon)}",
           f"  REFELEV={st.elev:.1f}", "", ">=MTSECT", f"  NFREQ={nf}", "",
           f">FREQ //{nf}", _fmt_block(st.freq), f">ZROT //{nf}", _fmt_block(np.zeros(nf))]
    comps = {"XX": (0, 0), "XY": (0, 1), "YX": (1, 0), "YY": (1, 1)}
    for c, (i, j) in comps.items():
        out += [f">Z{c}R ROT=ZROT //{nf}", _fmt_block(st.Z[:, i, j].real),
                f">Z{c}I ROT=ZROT //{nf}", _fmt_block(st.Z[:, i, j].imag),
                f">Z{c}.VAR ROT=ZROT //{nf}", _fmt_block(st.Zerr[:, i, j] ** 2)]
    if st.T is not None:
        for k, c in enumerate(("X", "Y")):
            out += [f">T{c}R.EXP //{nf}", _fmt_block(st.T[:, k].real),
                    f">T{c}I.EXP //{nf}", _fmt_block(st.T[:, k].imag),
                    f">T{c}VAR.EXP //{nf}", _fmt_block(st.Terr[:, k] ** 2)]
    out.append(">END")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
