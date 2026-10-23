"""Профили: проекция станций на линию, локальные координаты."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

R_EARTH = 6371000.0


def to_local(lat, lon, lat0, lon0):
    """Широта/долгота -> (восток, север) в метрах относительно (lat0, lon0)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    x = np.deg2rad(lon - lon0) * R_EARTH * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(lat - lat0) * R_EARTH
    return x, y


def from_local(x, y, lat0, lon0):
    lat = lat0 + np.rad2deg(np.asarray(y) / R_EARTH)
    lon = lon0 + np.rad2deg(np.asarray(x) / (R_EARTH * np.cos(np.deg2rad(lat0))))
    return lat, lon


@dataclass
class Profile:
    name: str
    stations: list            # имена станций
    lat0: float = 0.0         # начало профиля
    lon0: float = 0.0
    lat1: float = 0.0         # конец профиля
    lon1: float = 0.0
    extra: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, name, stations):
        """Профиль по МНК-прямой через станции (stations — объекты Station)."""
        lat = np.array([s.lat for s in stations])
        lon = np.array([s.lon for s in stations])
        la0, lo0 = lat.mean(), lon.mean()
        x, y = to_local(lat, lon, la0, lo0)
        P = np.c_[x, y]
        if len(stations) >= 2:
            _, _, vt = np.linalg.svd(P - P.mean(0))
            d = vt[0]
        else:
            d = np.array([1.0, 0.0])
        # направление: на восток (или на север, если профиль меридиональный)
        if d[0] < -1e-9 or (abs(d[0]) < 1e-9 and d[1] < 0):
            d = -d
        t = P @ d
        a = d * (t.min() - 0.0)
        b = d * (t.max() + 0.0)
        la_a, lo_a = from_local(a[0], a[1], la0, lo0)
        la_b, lo_b = from_local(b[0], b[1], la0, lo0)
        order = np.argsort(t)
        names = [stations[i].name for i in order]
        return cls(name, names, float(la_a), float(lo_a), float(la_b), float(lo_b))

    @property
    def azimuth(self):
        """Азимут профиля (град от севера по часовой)."""
        x, y = to_local(self.lat1, self.lon1, self.lat0, self.lon0)
        return float(np.mod(np.rad2deg(np.arctan2(x, y)), 180.0))

    @property
    def length(self):
        x, y = to_local(self.lat1, self.lon1, self.lat0, self.lon0)
        return float(np.hypot(x, y))

    def project(self, lat, lon):
        """Расстояние вдоль профиля и смещение от линии, м."""
        x, y = to_local(lat, lon, self.lat0, self.lon0)
        x1, y1 = to_local(self.lat1, self.lon1, self.lat0, self.lon0)
        L = max(np.hypot(x1, y1), 1e-9)
        ux, uy = x1 / L, y1 / L
        along = x * ux + y * uy
        off = -x * uy + y * ux
        return along, off

    def to_dict(self):
        return dict(name=self.name, stations=list(self.stations), lat0=self.lat0,
                    lon0=self.lon0, lat1=self.lat1, lon1=self.lon1)

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["stations"], d["lat0"], d["lon0"], d["lat1"], d["lon1"])
