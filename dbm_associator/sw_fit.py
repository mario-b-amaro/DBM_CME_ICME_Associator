import math

import numpy as np
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
)
from scipy.optimize import curve_fit

from .plotting import SimpleMplCanvas
from .utils import to_utc


def gaussian_mixture(x, *params):
    params = np.asarray(params).reshape(-1, 3)
    y = np.zeros_like(x, dtype=float)
    for mu, sigma, amplitude in params:
        sigma = abs(sigma) + 1e-6
        amplitude = max(amplitude, 0.0)
        y += amplitude * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return y


def mixture_moments(params):
    if params is None or len(params) == 0:
        return np.nan, np.nan
    p = np.asarray(params).reshape(-1, 3)
    mu = p[:, 0]
    sigma = np.abs(p[:, 1])
    amp = np.abs(p[:, 2])
    if not np.any(amp > 0):
        return np.nan, np.nan
    w = amp / amp.sum()
    mu_eff = float(np.sum(w * mu))
    var_eff = float(np.sum(w * (sigma ** 2 + (mu - mu_eff) ** 2)))
    return mu_eff, math.sqrt(max(var_eff, 0.0))


class SWFitDialog(QDialog):
    def __init__(self, parent, t_unix, Np, Vrad, sheath_t, sw_hours):
        super().__init__(parent)
        self.setWindowTitle("SW Histograms: Multi-Gaussian Fit")
        self.result_ready = False
        self.rho_sw = None
        self.sigma_rho_sw = None
        self.w_kms = None
        self.sigma_w_kms = None

        self.t_unix = np.asarray(t_unix)
        self.Np = np.asarray(Np)
        self.Vrad = np.asarray(Vrad)
        self.sw_start = sheath_t - sw_hours * 3600.0
        self.sw_end = sheath_t

        mask = (self.t_unix >= self.sw_start) & (self.t_unix < self.sw_end)
        self.n_sw_vals = self.Np[mask]
        self.v_sw_vals = self.Vrad[mask]
        self.n_sw_vals = self.n_sw_vals[np.isfinite(self.n_sw_vals)]
        self.v_sw_vals = self.v_sw_vals[np.isfinite(self.v_sw_vals)]

        if self.n_sw_vals.size < 20 or self.v_sw_vals.size < 20:
            raise RuntimeError("Not enough pre-sheath samples for interactive fit.")

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"Pre-sheath interval: {to_utc(self.sw_start)}  —  {to_utc(self.sw_end)}\n"
            f"Density samples: {len(self.n_sw_vals)}   Speed samples: {len(self.v_sw_vals)}"
        ))

        dens_header = QHBoxLayout()
        dens_header.addWidget(QLabel("Density histogram (Np, cm⁻³)"))
        dens_header.addStretch(1)
        dens_header.addWidget(QLabel("Gaussians:"))
        self.spnCompN = QSpinBox(); self.spnCompN.setRange(1, 5); self.spnCompN.setValue(1)
        dens_header.addWidget(self.spnCompN)
        layout.addLayout(dens_header)
        self.canvas_n = SimpleMplCanvas("Np [cm⁻³]", "Counts")
        layout.addWidget(self.canvas_n)

        speed_header = QHBoxLayout()
        speed_header.addWidget(QLabel("Speed histogram (v, km/s)"))
        speed_header.addStretch(1)
        speed_header.addWidget(QLabel("Gaussians:"))
        self.spnCompV = QSpinBox(); self.spnCompV.setRange(1, 5); self.spnCompV.setValue(1)
        speed_header.addWidget(self.spnCompV)
        layout.addLayout(speed_header)
        self.canvas_v = SimpleMplCanvas("v [km/s]", "Counts")
        layout.addWidget(self.canvas_v)

        self.txtSummary = QTextEdit(); self.txtSummary.setReadOnly(True)
        layout.addWidget(self.txtSummary)

        btn_box = QHBoxLayout()
        self.btnRefit = QPushButton("Refit")
        self.btnApply = QPushButton("Apply & Close")
        self.btnCancel = QPushButton("Cancel")
        btn_box.addWidget(self.btnRefit); btn_box.addStretch(1)
        btn_box.addWidget(self.btnApply); btn_box.addWidget(self.btnCancel)
        layout.addLayout(btn_box)

        self.btnRefit.clicked.connect(self.refit_all)
        self.btnApply.clicked.connect(self.apply_and_close)
        self.btnCancel.clicked.connect(self.reject)
        self.spnCompN.valueChanged.connect(self.refit_all)
        self.spnCompV.valueChanged.connect(self.refit_all)

        self.n_hist, self.n_edges = np.histogram(self.n_sw_vals, bins=80)
        self.v_hist, self.v_edges = np.histogram(self.v_sw_vals, bins=60)
        self.n_params = None
        self.v_params = None
        self.refit_all()

    def fit_mixture(self, x_cent, y_hist, n_comp):
        if n_comp <= 0 or np.all(y_hist <= 0):
            return None
        x_min, x_max = float(np.min(x_cent)), float(np.max(x_cent))
        n_bins = len(x_cent); seg_size = max(1, n_bins // n_comp)
        p0 = []; y_max = float(np.max(y_hist))
        for j in range(n_comp):
            i0 = j * seg_size; i1 = (j + 1) * seg_size if j < n_comp - 1 else n_bins
            xc_seg = x_cent[max(0, i0):max(i0 + 1, i1)]
            yc_seg = y_hist[max(0, i0):max(i0 + 1, i1)]
            mu_j = float(np.sum(xc_seg * yc_seg) / np.sum(yc_seg)) if np.any(yc_seg > 0) else x_min + (j + 0.5) * (x_max - x_min) / n_comp
            p0.extend([mu_j, 0.15 * (x_max - x_min), y_max / n_comp if y_max > 0 else 1.0])
        lower, upper = [], []
        for _ in range(n_comp):
            lower.extend([x_min, 1e-3, 0.0]); upper.extend([x_max, (x_max - x_min), np.inf])
        try:
            popt, _ = curve_fit(gaussian_mixture, x_cent, y_hist, p0=np.asarray(p0), bounds=(np.asarray(lower), np.asarray(upper)), maxfev=40000)
            return popt
        except Exception:
            return None

    def refit_all(self):
        x_cent_n = 0.5 * (self.n_edges[1:] + self.n_edges[:-1])
        x_cent_v = 0.5 * (self.v_edges[1:] + self.v_edges[:-1])
        self.n_params = self.fit_mixture(x_cent_n, self.n_hist, int(self.spnCompN.value()))
        self.v_params = self.fit_mixture(x_cent_v, self.v_hist, int(self.spnCompV.value()))

        axn = self.canvas_n.ax; axn.cla(); axn.grid(True, alpha=0.25); axn.set_xlabel("Np [cm⁻³]"); axn.set_ylabel("Counts")
        axn.bar(x_cent_n, self.n_hist, width=(self.n_edges[1] - self.n_edges[0]), alpha=0.5, align='center')
        if self.n_params is not None: axn.plot(x_cent_n, gaussian_mixture(x_cent_n, *self.n_params), lw=1.5)
        self.canvas_n.draw_idle()

        axv = self.canvas_v.ax; axv.cla(); axv.grid(True, alpha=0.25); axv.set_xlabel("v [km/s]"); axv.set_ylabel("Counts")
        axv.bar(x_cent_v, self.v_hist, width=(self.v_edges[1] - self.v_edges[0]), alpha=0.5, align='center')
        if self.v_params is not None: axv.plot(x_cent_v, gaussian_mixture(x_cent_v, *self.v_params), lw=1.5)
        self.canvas_v.draw_idle()

        mu_n, sig_n = mixture_moments(self.n_params)
        mu_v, sig_v = mixture_moments(self.v_params)
        self.rho_sw = mu_n * 1e6 * 1.67262192e-27 if np.isfinite(mu_n) else np.nan
        self.sigma_rho_sw = sig_n * 1e6 * 1.67262192e-27 if np.isfinite(sig_n) else np.nan
        self.w_kms = mu_v; self.sigma_w_kms = sig_v
        self.txtSummary.setPlainText(
            f"Effective Np: mu={mu_n:.2f} ± {sig_n:.2f} cm⁻³\n"
            f"→ rho_sw = {self.rho_sw:.3e} ± {self.sigma_rho_sw:.3e} kg/m³\n\n"
            f"Effective v: mu={mu_v:.2f} ± {sig_v:.2f} km/s"
        )

    def apply_and_close(self):
        if self.rho_sw is None or self.w_kms is None:
            QMessageBox.warning(self, "No result", "No valid fit results to apply.")
            return
        self.result_ready = True
        self.accept()
