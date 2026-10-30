"""Генерация синтетических EDI (2D-модель) для проверки программы."""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mtinv.edi import write_edi  # noqa: E402
from mtinv.inv2d import Inv2DData, Inv2DSettings, build_mesh, forward_2d  # noqa: E402
from mtinv.mtdata import Station  # noqa: E402
from mtinv.profile import from_local  # noqa: E402
from mtinv.tem import TEMForward, TEMSounding, write_tem  # noqa: E402

# статический сдвиг (множитель ρa строк XY и YX), который «портит» часть станций
STATIC_SHIFT = {2: (0.35, 0.5), 6: (2.5, 1.8), 10: (0.5, 0.4), 12: (3.0, 2.2)}


def true_column(xj):
    """Истинная 1D-колонка под точкой профиля (для ЗСБ): кровли и ρ."""
    tops = [0.0, 300.0]
    rhos = [30.0, 100.0]
    if xj > 28000:
        tops += [800.0, 4000.0]
        rhos += [2000.0, 100.0]
    elif 14000 < xj < 24000:
        tops += [2000.0, 6000.0]
        rhos += [3.0, 100.0]
    tops.append(25000.0)
    rhos.append(1000.0)
    return np.array(rhos), np.diff(tops)


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    nst = 15
    x = np.linspace(0, 42000, nst) + rng.normal(0, 300, nst)
    x[0], x[-1] = 0, 42000
    freqs = np.logspace(np.log10(320), -3, 28)
    d = Inv2DData([str(i) for i in range(nst)], x, freqs, {"te": dict(rho=np.full((28, nst), 30.0))})
    mesh, _ = build_mesh(d, Inv2DSettings(z_first=5, cells_between=6), 100.0)
    cc = mesh.cell_centers
    X, Zc = cc[:, 0], -cc[:, 1]
    rho = np.full(mesh.n_cells, 100.0)
    rho[Zc < 300] = 30.0                                         # осадки
    rho[(X > 14000) & (X < 24000) & (Zc > 2000) & (Zc < 6000)] = 3.0   # проводник
    rho[(X > 28000) & (Zc > 800) & (Zc < 4000)] = 2000.0              # изолятор
    rho[Zc > 25000] = 1000.0
    rho[Zc < 0] = 1e9
    out = forward_2d(mesh, rho, x, freqs)
    lat0, lon0 = 72.30, 95.20  # Таймыр
    # профиль на восток-северо-восток, азимут 80°
    az = np.deg2rad(80)
    lat, lon = from_local(x * np.sin(az), x * np.cos(az), lat0, lon0)
    # простирание = азимут профиля - 90 = -10° -> поворачиваем данные обратно
    strike = -10.0
    for j in range(nst):
        Z = np.zeros((len(freqs), 2, 2), complex)
        amp_te = np.sqrt(out["te"]["rho"][:, j] * 5 * freqs)
        amp_tm = np.sqrt(out["tm"]["rho"][:, j] * 5 * freqs)
        Z[:, 0, 1] = amp_te * np.exp(1j * np.deg2rad(out["te"]["phs"][:, j]))
        Z[:, 1, 0] = -amp_tm * np.exp(1j * np.deg2rad(out["tm"]["phs"][:, j]))
        noise = 0.03 * np.abs(Z[:, 0, 1])[:, None, None] * (
            rng.normal(size=Z.shape) + 1j * rng.normal(size=Z.shape))
        Z[:, 0, 0] += noise[:, 0, 0] * 0.3
        Z[:, 1, 1] += noise[:, 1, 1] * 0.3
        Z[:, 0, 1] += noise[:, 0, 1]
        Z[:, 1, 0] += noise[:, 1, 0]
        Zerr = np.full(Z.shape, 0.03) * np.sqrt(np.abs(Z[:, 0, 1] * Z[:, 1, 0]))[:, None, None]
        st = Station(f"TMR{j + 1:03d}", float(lat[j]), float(lon[j]), 50.0 + 5 * j, freqs, Z, Zerr)
        st = st.rotated(-strike)  # из системы простирания в географическую
        st.zrot = 0.0
        if j in STATIC_SHIFT:  # гальванические искажения в системе измерения
            kx, ky = STATIC_SHIFT[j]
            st.Z[:, 0, :] *= np.sqrt(kx)
            st.Z[:, 1, :] *= np.sqrt(ky)
            st.Zerr[:, 0, :] *= np.sqrt(kx)
            st.Zerr[:, 1, :] *= np.sqrt(ky)
        write_edi(st, os.path.join(out_dir, f"TMR{j + 1:03d}.edi"))

    # ЗСБ рядом с каждой станцией: петля 200 м, центральная установка,
    # приёмник с эффективной площадью 10000 м², 10 мкс – 8 мс
    tem_dir = os.path.join(out_dir, "zsb")
    os.makedirs(tem_dir, exist_ok=True)
    times = np.logspace(-5, np.log10(8e-3), 30)
    fwd = TEMForward(times, 200.0)
    for j in range(nst):
        rho_c, thick = true_column(x[j])
        sig = fwd(rho_c, thick)
        err = 0.03 * sig + 2e-12   # 3 % + шум на поздних временах
        noisy = sig + err * rng.normal(size=len(sig))
        la, lo = from_local(x[j] * np.sin(az) - 150 * np.sin(az - np.pi / 2),
                            x[j] * np.cos(az) + 150, lat0, lon0)
        snd = TEMSounding(f"ZSB{j + 1:03d}", float(la), float(lo), times, np.abs(noisy), err,
                          tx_side=200.0)
        write_tem(snd, os.path.join(tem_dir, f"ZSB{j + 1:03d}.txt"), rx_area=10000.0)
    print("записано", nst, "EDI и", nst, "ЗСБ в", out_dir)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "demo_data"))
