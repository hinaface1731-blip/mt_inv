"""Подложки карт (тайлы XYZ) с кэшированием на диске."""

from __future__ import annotations

import io
import math
import os

import numpy as np

R = 6378137.0
PROVIDERS = {
    "Нет": None,
    "OpenStreetMap": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "OpenTopoMap (рельеф)": "https://a.tile.opentopomap.org/{z}/{x}/{y}.png",
    "Esri спутник": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Esri топо": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
}
CACHE = os.path.join(os.path.expanduser("~"), ".mtinv", "tiles")


def merc(lat, lon):
    lat = np.clip(np.asarray(lat, float), -85, 85)
    x = R * np.deg2rad(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.deg2rad(lat) / 2))
    return x, y


def inv_merc(x, y):
    lon = np.rad2deg(np.asarray(x) / R)
    lat = np.rad2deg(2 * np.arctan(np.exp(np.asarray(y) / R)) - np.pi / 2)
    return lat, lon


def _tile_xy(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lr = math.radians(lat)
    y = (1.0 - math.asinh(math.tan(lr)) / math.pi) / 2.0 * n
    return x, y


def _tile_bounds(tx, ty, z):
    n = 2 ** z
    lon0 = tx / n * 360 - 180
    lon1 = (tx + 1) / n * 360 - 180
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (ty + 1) / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * ty / n))))
    x0, y0 = merc(lat0, lon0)
    x1, y1 = merc(lat1, lon1)
    return float(x0), float(x1), float(y0), float(y1)


def _fetch(url, path):
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "MT-Inv/0.1 (geophysics GUI)"})
    with urllib.request.urlopen(req, timeout=8) as r:
        data = r.read()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return data


def get_tiles(provider, lat_min, lat_max, lon_min, lon_max, max_tiles=30):
    """Список (image ndarray, extent(x0,x1,y0,y1) в Меркаторе)."""
    tmpl = PROVIDERS.get(provider)
    if not tmpl:
        return []
    import matplotlib.image as mpimg

    z = 18
    while z > 1:
        x0, y1 = _tile_xy(lat_min, lon_min, z)
        x1, y0 = _tile_xy(lat_max, lon_max, z)
        nt = (int(x1) - int(x0) + 1) * (int(y1) - int(y0) + 1)
        if nt <= max_tiles:
            break
        z -= 1
    x0, y1 = _tile_xy(lat_min, lon_min, z)
    x1, y0 = _tile_xy(lat_max, lon_max, z)
    out = []
    key = "".join(c for c in provider if c.isalnum())
    for tx in range(int(x0), int(x1) + 1):
        for ty in range(int(y0), int(y1) + 1):
            path = os.path.join(CACHE, key, str(z), str(tx), f"{ty}.png")
            try:
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        data = f.read()
                else:
                    data = _fetch(tmpl.format(z=z, x=tx, y=ty), path)
                fmt = "jpg" if data[:3] == b"\xff\xd8\xff" else "png"
                img = mpimg.imread(io.BytesIO(data), format=fmt)
                out.append((img, _tile_bounds(tx, ty, z)))
            except Exception:
                continue
    return out
