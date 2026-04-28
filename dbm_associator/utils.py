import math
import re
from datetime import datetime, timezone

import matplotlib.dates as mdates
import numpy as np
import pytplot
import requests


def to_ts(s: str) -> float:
    return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp()


def to_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def epoch_to_mpl_array(ts_array):
    return mdates.date2num([datetime.fromtimestamp(t, tz=timezone.utc) for t in ts_array])


def epoch_to_mpl_scalar(t):
    return mdates.date2num(datetime.fromtimestamp(t, tz=timezone.utc))


def nearest_series(x_src, y_src, x_ref):
    x_src = np.asarray(x_src)
    y_src = np.asarray(y_src)
    x_ref = np.asarray(x_ref)
    if y_src.ndim == 1:
        y_src = y_src[:, None]
    y_out = np.full((len(x_ref), y_src.shape[1]), np.nan)
    j = 0
    for i, t in enumerate(x_ref):
        while j + 1 < len(x_src) and abs(x_src[j + 1] - t) < abs(x_src[j] - t):
            j += 1
        y_out[i, :] = y_src[j, :]
    return y_out.squeeze()


def mag_mag_components_from_tplot(base_name_candidates):
    names = pytplot.tplot_names()
    for base in base_name_candidates:
        if base in names:
            pytplot.split_vec(base)
            try:
                bx = pytplot.get_data(f"{base}_x")
                by = pytplot.get_data(f"{base}_y")
                bz = pytplot.get_data(f"{base}_z")
                if bx is not None and by is not None and bz is not None:
                    return bx.times, bx.y, by.y, bz.y
            except Exception:
                pass
    for nm in names:
        if nm.endswith('_x'):
            root = nm[:-2]
            if (root + '_y') in names and (root + '_z') in names:
                bx = pytplot.get_data(nm)
                by = pytplot.get_data(root + '_y')
                bz = pytplot.get_data(root + '_z')
                return bx.times, bx.y, by.y, bz.y
    return None, None, None, None


def first_existing(*candidates):
    names = set(pytplot.tplot_names())
    for c in candidates:
        if c in names:
            return c
    return None


def gaussian(x, mu, sigma, amplitude):
    return amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def pad_int(val, width):
    s = str(val).strip().replace('*', '')
    if s in ('-----', '-------'):
        return None
    try:
        return f"{int(float(s)):0{width}d}"
    except Exception:
        return None


def parse_catalog_line(line):
    line = line.strip()
    if not line or not re.match(r'^\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}', line):
        return None
    toks = line.split()
    if len(toks) < 12:
        return None
    try:
        d, t = toks[0], toks[1]
        cpa = toks[2]
        width = toks[3]
        linear = toks[4]
        init = toks[5]
        final = toks[6]
        v20r = toks[7]
        accel = toks[8]
        mpa = toks[11]
        if not re.match(r'^\d+$', mpa):
            for tok in reversed(toks):
                if re.match(r'^\d+$', tok):
                    mpa = tok
                    break
        dt = datetime.strptime(d + ' ' + t, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
        return {
            'dt': dt, 'date': d, 'time': t, 'cpa': cpa, 'width': width,
            'linear': linear, 'init': init, 'final': final, 'v20r': v20r,
            'accel': accel, 'mpa': mpa, 'raw': line,
        }
    except Exception:
        return None


def fetch_text(url, timeout=30):
    r = requests.get(url, headers={"User-Agent": "CME-ICME-GUI/1.1"}, timeout=timeout)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or 'utf-8'
    return r.text


def fetch_bytes(url, timeout=30):
    r = requests.get(url, headers={"User-Agent": "CME-ICME-GUI/1.1"}, timeout=timeout)
    r.raise_for_status()
    return r.content


def parse_yht_for_heights(text):
    heights, times = [], []
    for line in text.splitlines():
        m = re.match(r'^\s*(\d+(?:\.\d+)?)\s+(\d{4}/\d{2}/\d{2})\s+(\d{2}:\d{2}:\d{2})', line)
        if m:
            h = float(m.group(1))
            dt = datetime.strptime(m.group(2) + ' ' + m.group(3), "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc)
            heights.append(h)
            times.append(dt)
    if len(heights) < 3:
        return None, None, None
    t0 = times[0]
    t_sec = np.array([(ti - t0).total_seconds() for ti in times], dtype=float)
    h_arr = np.array(heights, dtype=float)
    return t_sec, h_arr, t0


def quadratic_time_at_height(t_sec, h_arr, target_h=20.0):
    if len(t_sec) < 3:
        return None
    try:
        a, b, c = np.polyfit(t_sec, h_arr, deg=2)
        disc = b * b - 4 * a * (c - target_h)
        if disc < 0:
            return None
        if abs(a) > 1e-14:
            roots = [(-b + math.sqrt(disc)) / (2 * a), (-b - math.sqrt(disc)) / (2 * a)]
        elif abs(b) > 1e-14:
            roots = [-(c - target_h) / b]
        else:
            roots = []
        roots = [r for r in roots if np.isfinite(r) and r >= 0]
        if not roots:
            return None
        t20 = min(roots)
        return t20 if t20 <= 7 * 24 * 3600 else None
    except Exception:
        return None
