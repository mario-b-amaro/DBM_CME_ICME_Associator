import math
import re
from datetime import datetime, timedelta, timezone

import matplotlib.dates as mdates
import numpy as np
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy.optimize import curve_fit, fsolve

from .data_loader import MissionLoader
from .plotting import MplPanel, ResultsDialog
from .sw_fit import SWFitDialog
from .utils import (
    epoch_to_mpl_array,
    epoch_to_mpl_scalar,
    fetch_bytes,
    fetch_text,
    gaussian,
    nearest_series,
    pad_int,
    parse_catalog_line,
    parse_yht_for_heights,
    quadratic_time_at_height,
    to_ts,
    to_utc,
)
from .version import __version__

class CMEGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"DBM CME–ICME Association Tool v{__version__}")

        self.loader = None
        self.data = None
        self.t_unix = None
        self.t_mpl = None

        self.sheath_t = np.nan
        self.narrow1_t = np.nan
        self.narrow2_t = np.nan
        self.wide1_t = np.nan
        self.wide2_t = np.nan

        self.last_dep_window = None
        self.last_v0_window = None

        ctrl = QGridLayout()

        ctrl.addWidget(QLabel("Mission:"), 0, 0)
        self.cmbMission = QComboBox()
        self.cmbMission.addItems(["PSP", "SOLO", "WIND", "ACE"])
        ctrl.addWidget(self.cmbMission, 0, 1)

        ctrl.addWidget(QLabel("Trange start (UTC):"), 0, 2)
        self.edStart = QLineEdit("2018-11-11 15:00:00")
        ctrl.addWidget(self.edStart, 0, 3)

        ctrl.addWidget(QLabel("Trange end (UTC):"), 0, 4)
        self.edEnd = QLineEdit("2018-11-12 08:00:00")
        ctrl.addWidget(self.edEnd, 0, 5)

        self.btnLoad = QPushButton("Load Data")
        self.btnLoad.clicked.connect(self.load_data)
        ctrl.addWidget(self.btnLoad, 0, 6)

        row = 1
        ctrl.addWidget(QLabel("Sheath start (UTC):"), row, 0)
        self.edSheath = QLineEdit("")
        ctrl.addWidget(self.edSheath, row, 1)

        ctrl.addWidget(QLabel("MO narrow start:"), row, 2)
        self.edNar1 = QLineEdit("")
        ctrl.addWidget(self.edNar1, row, 3)

        ctrl.addWidget(QLabel("MO narrow end:"), row, 4)
        self.edNar2 = QLineEdit("")
        ctrl.addWidget(self.edNar2, row, 5)

        row += 1
        ctrl.addWidget(QLabel("MO wide start:"), row, 0)
        self.edWide1 = QLineEdit("")
        ctrl.addWidget(self.edWide1, row, 1)

        ctrl.addWidget(QLabel("MO wide end:"), row, 2)
        self.edWide2 = QLineEdit("")
        ctrl.addWidget(self.edWide2, row, 3)

        self.btnUpdate = QPushButton("Update Boundaries")
        self.btnUpdate.clicked.connect(self.update_boundaries_clicked)
        ctrl.addWidget(self.btnUpdate, row, 5)

        row += 1
        ctrl.addWidget(QLabel("Hours before sheath for SW fit:"), row, 0)
        self.spnSWHours = QSpinBox()
        self.spnSWHours.setRange(1, 48)
        self.spnSWHours.setValue(24)
        ctrl.addWidget(self.spnSWHours, row, 1)

        ctrl.addWidget(QLabel("Cd (fixed):"), row, 2)
        self.spnCd = QDoubleSpinBox()
        self.spnCd.setRange(0.1, 5.0)
        self.spnCd.setSingleStep(0.1)
        self.spnCd.setValue(1.0)
        ctrl.addWidget(self.spnCd, row, 3)

        ctrl.addWidget(QLabel("Max fsolve iterations:"), row, 4)
        self.spnMaxIter = QSpinBox()
        self.spnMaxIter.setRange(100, 50000)
        self.spnMaxIter.setValue(10000)
        ctrl.addWidget(self.spnMaxIter, row, 5)

        ctrl.addWidget(QLabel("v0 guess [km/s]:"), row, 6)
        self.spnV0Guess = QDoubleSpinBox()
        self.spnV0Guess.setRange(0.0, 4000.0)
        self.spnV0Guess.setSingleStep(10.0)
        self.spnV0Guess.setValue(300.0)
        ctrl.addWidget(self.spnV0Guess, row, 7)

        row += 1
        ctrl.addWidget(QLabel("T guess [s]:"), row, 0)
        self.spnTGuess = QDoubleSpinBox()
        self.spnTGuess.setRange(0.0, 1.0e7)
        self.spnTGuess.setSingleStep(600.0)
        self.spnTGuess.setValue(50000)
        ctrl.addWidget(self.spnTGuess, row, 1)

        self.btnCompute = QPushButton("Compute Parameters")
        self.btnCompute.clicked.connect(self.compute_parameters_only)
        ctrl.addWidget(self.btnCompute, row, 6)

        self.btnSolve = QPushButton("Solve DBM")
        self.btnSolve.clicked.connect(self.solve_dbm_only)
        ctrl.addWidget(self.btnSolve, row, 7)

        row += 1
        self.btnLASCO = QPushButton("Search LASCO Catalogue")
        self.btnLASCO.clicked.connect(self.search_lasco_catalogue)
        ctrl.addWidget(self.btnLASCO, row, 0, 1, 3)

        self.btnSavePlots = QPushButton("Save All Plots as Image…")
        self.btnSavePlots.clicked.connect(self.save_all_plots_image)
        ctrl.addWidget(self.btnSavePlots, row, 3, 1, 3)

        self.btnSWFit = QPushButton("Fit SW Histograms…")
        self.btnSWFit.clicked.connect(self.open_sw_fit_dialog)
        ctrl.addWidget(self.btnSWFit, row, 6, 1, 2)

        row += 1
        param_box = QGridLayout()
        self.lbl_dt = QLabel("dt: –")
        self.lbl_v = QLabel("v: –")
        self.lbl_L = QLabel("L: –")
        self.lbl_rho = QLabel("ρ: –")
        self.lbl_rhosw = QLabel("ρ_sw: –")
        self.lbl_gamma = QLabel("γ: –")
        self.lbl_w = QLabel("w: –")

        param_box.addWidget(self.lbl_dt, 0, 0)
        param_box.addWidget(self.lbl_v, 1, 0)
        param_box.addWidget(self.lbl_L, 2, 0)
        param_box.addWidget(self.lbl_rho, 3, 0)
        param_box.addWidget(self.lbl_rhosw, 4, 0)
        param_box.addWidget(self.lbl_gamma, 5, 0)
        param_box.addWidget(self.lbl_w, 6, 0)

        self.lbl_v0win = QLabel("v0 window: –")
        self.lbl_Tiwin = QLabel("T_i window: –")
        self.lbl_depwin = QLabel("20 R☉ departure: –")

        param_box.addWidget(self.lbl_v0win, 0, 1)
        param_box.addWidget(self.lbl_Tiwin, 1, 1)
        param_box.addWidget(self.lbl_depwin, 2, 1)

        ctrl.addLayout(param_box, row, 0, 1, 9)
        
        self.pB = MplPanel("|B| and components", "B [nT]", logy=False)
        self.pT = MplPanel("Proton Temperature", "T [eV]", logy=False)
        self.pV = MplPanel("Flow Speed (radial or bulk)", "v [km s$^{-1}$]", logy=False)
        self.pN = MplPanel("Number Density", "n [cm$^{-3}$]", logy=False)
        self.pBeta = MplPanel("Plasma Beta", "β", logy=True)

        plots_container = QWidget()
        vplots = QVBoxLayout(plots_container)
        for w in (self.pB, self.pT, self.pV, self.pN, self.pBeta):
            vplots.addWidget(w)
        vplots.addStretch(1)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(plots_container)

        lay = QVBoxLayout(self)
        lay.addLayout(ctrl)
        lay.addWidget(self.scroll)

    # HELPERS FOR PLOTTING

    def _plot_mask(self):
        if self.t_unix is None:
            return np.full(0, False, dtype=bool)
        try:
            start_ts = to_ts(self.edStart.text())
            end_ts = to_ts(self.edEnd.text())
        except Exception:
            return np.full_like(self.t_unix, True, dtype=bool)
        return (self.t_unix >= start_ts) & (self.t_unix <= end_ts)

    def _plot_xlim(self):
        try:
            start_ts = to_ts(self.edStart.text())
            end_ts = to_ts(self.edEnd.text())
            return epoch_to_mpl_scalar(start_ts), epoch_to_mpl_scalar(end_ts)
        except Exception:
            return self.t_mpl[0], self.t_mpl[-1]

    # CORE ACTIONS

    def load_data(self):
        try:
            start_str = self.edStart.text().strip()
            end_str = self.edEnd.text().strip()
            start_dt = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            ext_start = start_dt - timedelta(hours=48)
            tr = [
                ext_start.strftime("%Y-%m-%d/%H:%M:%S"),
                end_dt.strftime("%Y-%m-%d/%H:%M:%S")
            ]

            self.loader = MissionLoader(self.cmbMission.currentText(), tr)
            mv = self.loader.load()
            self.data = mv

            self.t_unix = mv.t_mag.astype(float)
            self.t_mpl = epoch_to_mpl_array(self.t_unix)

            self.Br, self.Bt, self.Bn, self.Babs = mv.Br, mv.Bt, mv.Bn, mv.Bmag
            self.Vrad = nearest_series(mv.t_v, mv.Vrad, self.t_unix)
            self.Np = nearest_series(mv.t_n, mv.Np, self.t_unix)
            self.Tp = nearest_series(mv.t_T, mv.Tp, self.t_unix)

            Bsq = self.Babs ** 2
            self.beta = (self.Np * self.Tp) / Bsq * 3.5 * 10.0

            for edt, val in [
                (self.edSheath, self.edStart.text()),
                (self.edNar1, self.edStart.text()),
                (self.edNar2, self.edEnd.text()),
                (self.edWide1, self.edStart.text()),
                (self.edWide2, self.edEnd.text())
            ]:
                if not edt.text().strip():
                    edt.setText(val)

            self.update_boundary_values_from_fields()
            self.draw_all()
            QMessageBox.information(self, "Success", "Data loaded and plotted.")

        except Exception as e:
            QMessageBox.critical(self, "Load failed", f"{type(e).__name__}: {e}")

    def draw_all(self):
        mask = self._plot_mask()
        t_plot = self.t_mpl[mask]
        Babs = self.Babs[mask]
        Br = self.Br[mask]
        Bt = self.Bt[mask]
        Bn = self.Bn[mask]
        Tp = self.Tp[mask]
        Vrad = self.Vrad[mask]
        Np = self.Np[mask]
        beta = self.beta[mask]

        self.pB.clear_plot()
        self.pT.clear_plot()
        self.pV.clear_plot()
        self.pN.clear_plot()
        self.pBeta.clear_plot(logy=True)

        ax = self.pB.ax
        ax.plot(t_plot, Babs, lw=1.2, color='k', label='|B|')
        ax.plot(t_plot, Br, lw=0.9, color='tab:red', label='B_R')
        ax.plot(t_plot, Bt, lw=0.9, color='tab:green', label='B_T')
        ax.plot(t_plot, Bn, lw=0.9, color='tab:blue', label='B_N')
        ax.legend(loc='upper right', frameon=False)
        ax.set_ylabel('B [nT]')

        self.pT.ax.plot(t_plot, Tp, lw=1.2, color='black')
        self.pT.ax.set_ylabel('T [eV]')

        self.pV.ax.plot(t_plot, Vrad, lw=1.2, color='black')
        self.pV.ax.set_ylabel('v [km/s]')

        self.pN.ax.plot(t_plot, Np, lw=1.2, color='black')
        self.pN.ax.set_ylabel('n [cm$^{-3}$]')

        self.pBeta.ax.plot(t_plot, beta, lw=1.2, color='black')
        self.pBeta.ax.set_ylabel('β')

        self.refresh_boundary_lines()

        x0, x1 = self._plot_xlim()
        for p in (self.pB, self.pT, self.pV, self.pN, self.pBeta):
            p.ax.set_xlim(x0, x1)
            p.figure.canvas.draw_idle()

    def update_boundary_values_from_fields(self):
        def safe_parse(txt):
            txt = txt.strip()
            return to_ts(txt) if txt else np.nan
        self.sheath_t = safe_parse(self.edSheath.text())
        self.narrow1_t = safe_parse(self.edNar1.text())
        self.narrow2_t = safe_parse(self.edNar2.text())
        self.wide1_t = safe_parse(self.edWide1.text())
        self.wide2_t = safe_parse(self.edWide2.text())

    def refresh_boundary_lines(self):
        times_colors = [
            (self.sheath_t, 'red'),
            (self.narrow1_t, 'blue'),
            (self.narrow2_t, 'blue'),
            (self.wide1_t, 'green'),
            (self.wide2_t, 'green')
        ]
        for panel in (self.pB, self.pT, self.pV, self.pN, self.pBeta):
            panel.draw_verticals(times_and_colors=times_colors)

    def update_boundaries_clicked(self):
        try:
            self.update_boundary_values_from_fields()
            self.refresh_boundary_lines()
        except Exception as e:
            QMessageBox.critical(self, "Update failed", f"{type(e).__name__}: {e}")

    # COMPUTE PARAMETERS

    def compute_parameters_only(self):
        try:
            if self.data is None:
                raise RuntimeError("Load data first.")
            if not (np.isfinite(self.narrow1_t) and np.isfinite(self.narrow2_t) and
                    np.isfinite(self.wide1_t) and np.isfinite(self.wide2_t)):
                raise RuntimeError("Please set all MO boundaries (narrow & wide).")

            dt1 = abs(self.wide2_t - self.wide1_t)
            dt2 = abs(self.narrow2_t - self.narrow1_t)
            dt = 0.5 * (dt1 + dt2)
            sigma_dt = 0.5 * abs(dt1 - dt2)

            def avg_in(t1, t2, t_ref, y_ref):
                mask = (t_ref >= t1) & (t_ref <= t2)
                vals = y_ref[mask]
                return np.nanmean(vals) if vals.size else np.nan

            t_ref = self.t_unix
            v1_kms = avg_in(self.wide1_t, self.wide2_t, t_ref, self.Vrad)
            v2_kms = avg_in(self.narrow1_t, self.narrow2_t, t_ref, self.Vrad)
            if not (np.isfinite(v1_kms) and np.isfinite(v2_kms)):
                raise RuntimeError("No velocity points inside MO windows.")
            v_kms = 0.5 * (v1_kms + v2_kms)
            sigma_v_kms = 0.5 * abs(v1_kms - v2_kms)

            v_mps = v_kms * 1e3
            sv_mps = sigma_v_kms * 1e3
            L_m = dt * v_mps
            sigma_L_m = math.sqrt((dt ** 2) * (sv_mps ** 2) + (v_mps ** 2) * (sigma_dt ** 2))

            L_km = L_m / 1e3
            sigma_L_km = sigma_L_m / 1e3

            mp = 1.67262192e-27
            n_to_m3 = 1e6

            nmo1 = avg_in(self.wide1_t, self.wide2_t, t_ref, self.Np)
            nmo2 = avg_in(self.narrow1_t, self.narrow2_t, t_ref, self.Np)
            rho_1 = nmo1 * n_to_m3 * mp
            rho_2 = nmo2 * n_to_m3 * mp
            rho = 0.5 * (rho_1 + rho_2)
            sigma_rho = 0.5 * abs(rho_1 - rho_2)

            if not np.isfinite(self.sheath_t):
                raise RuntimeError("Please set sheath start.")

            sw_hours = int(self.spnSWHours.value())
            sw_start = self.sheath_t - sw_hours * 3600.0
            sw_end = self.sheath_t

            mask_sw_n = (t_ref >= sw_start) & (t_ref < sw_end)
            n_sw_vals = self.Np[mask_sw_n]
            n_sw_vals = n_sw_vals[np.isfinite(n_sw_vals)]
            if n_sw_vals.size < 20:
                raise RuntimeError("Not enough pre-sheath density samples for Gaussian fit.")
            y_hist, x_edges = np.histogram(n_sw_vals, bins=200)
            x_cent = 0.5 * (x_edges[1:] + x_edges[:-1])
            p0 = (np.nanmedian(n_sw_vals),
                  np.nanstd(n_sw_vals) if np.nanstd(n_sw_vals) > 0 else 1.0,
                  np.max(y_hist))
            popt, _ = curve_fit(gaussian, x_cent, y_hist, p0=p0, maxfev=20000)
            mu_n, sig_n, _A = popt
            rho_sw = mu_n * n_to_m3 * mp
            sigma_rho_sw = sig_n * n_to_m3 * mp

            mask_sw_v = (t_ref >= sw_start) & (t_ref < sw_end)
            v_sw_vals = self.Vrad[mask_sw_v]
            v_sw_vals = v_sw_vals[np.isfinite(v_sw_vals)]
            if v_sw_vals.size < 20:
                raise RuntimeError("Not enough pre-sheath speed samples for Gaussian fit.")
            yv, xv_edges = np.histogram(v_sw_vals, bins=75)
            xv = 0.5 * (xv_edges[1:] + xv_edges[:-1])
            p0v = (np.nanmedian(v_sw_vals),
                   np.nanstd(v_sw_vals) if np.nanstd(v_sw_vals) > 0 else 5.0,
                   np.max(yv))
            poptv, _ = curve_fit(gaussian, xv, yv, p0=p0v, maxfev=20000)
            w_kms = float(poptv[0])
            sigma_w_kms = abs(float(poptv[1]))

            Cd0 = float(self.spnCd.value())

            def gamma_from_inputs_km(Cd, L_km, sigma_L_km, rho, sigma_rho, rho_sw, sigma_rho_sw):
                rho_ratio = rho / rho_sw
                g = Cd / (L_km * (rho_ratio + 0.5))
                sigma_g = math.sqrt(
                    (Cd ** 2 * sigma_L_km ** 2) / (((rho_ratio + 0.5) ** 2) * (L_km ** 4)) +
                    (Cd ** 2 * sigma_rho ** 2) / (((rho_ratio + 0.5) ** 4) * (L_km ** 2) * (rho_sw ** 2)) +
                    (16 * rho ** 2 * Cd ** 2 * sigma_rho_sw ** 2) /
                    (((rho_sw + 2 * rho) ** 4) * (L_km ** 2))
                )
                return g, sigma_g

            g_km, sg_km = gamma_from_inputs_km(Cd0, L_km, sigma_L_km, rho, sigma_rho, rho_sw, sigma_rho_sw)

            self.lbl_dt.setText(f"dt: {dt:.1f} ± {sigma_dt:.1f} s")
            self.lbl_v.setText(f"v: {v_kms:.2f} ± {sigma_v_kms:.2f} km/s")
            self.lbl_L.setText(f"L: {L_km:.3f} ± {sigma_L_km:.3f} km")
            self.lbl_rho.setText(f"ρ: {rho:.3e} ± {sigma_rho:.3e} kg/m³")
            self.lbl_rhosw.setText(f"ρ_sw: {rho_sw:.3e} ± {sigma_rho_sw:.3e} kg/m³")
            self.lbl_gamma.setText(f"γ: {g_km:.3e} ± {sg_km:.3e} 1/km")
            self.lbl_w.setText(f"w: {w_kms:.2f} ± {sigma_w_kms:.2f} km/s")

            self._cache_params = dict(
                dt=dt, sdt=sigma_dt,
                v_kms=v_kms, sv_kms=sigma_v_kms,
                L_m=L_m, sL_m=sigma_L_m,
                L_km=L_km, sL_km=sigma_L_km,
                rho=rho, srho=sigma_rho,
                rho_sw=rho_sw, srho_sw=sigma_rho_sw,
                w_kms=w_kms, sw_kms=sigma_w_kms,
                g_km=g_km, sg_km=sg_km,
                Cd0=Cd0
            )

            self.lbl_v0win.setText("v0 window: –")
            self.lbl_Tiwin.setText("T_i window: –")
            self.lbl_depwin.setText("20 R☉ departure: –")

        except Exception as e:
            QMessageBox.critical(self, "Compute failed", f"{type(e).__name__}: {e}")

    # SW HISTOGRAM DIALOG INTEGRATION

    def open_sw_fit_dialog(self):
        try:
            if self.data is None:
                raise RuntimeError("Load data first.")
            if not hasattr(self, '_cache_params'):
                raise RuntimeError("Compute parameters first.")

            if not np.isfinite(self.sheath_t):
                raise RuntimeError("Please set sheath start.")

            sw_hours = int(self.spnSWHours.value())

            dlg = SWFitDialog(self, self.t_unix, self.Np, self.Vrad, self.sheath_t, sw_hours)
            if dlg.exec_() == QDialog.Accepted and dlg.result_ready:
                rho_sw = dlg.rho_sw
                sigma_rho_sw = dlg.sigma_rho_sw
                w_kms = dlg.w_kms
                sigma_w_kms = dlg.sigma_w_kms

                if not np.isfinite(rho_sw) or not np.isfinite(w_kms):
                    raise RuntimeError("Fitted SW parameters are not finite.")

                self.lbl_rhosw.setText(f"ρ_sw (multi-Gauss): {rho_sw:.3e} ± {sigma_rho_sw:.3e} kg/m³")
                self.lbl_w.setText(f"w (multi-Gauss): {w_kms:.2f} ± {sigma_w_kms:.2f} km/s")

                p = self._cache_params
                p['rho_sw'] = rho_sw
                p['srho_sw'] = sigma_rho_sw
                p['w_kms'] = w_kms
                p['sw_kms'] = sigma_w_kms

                Cd0 = p['Cd0']
                L_km = p['L_km']
                sigma_L_km = p['sL_km']
                rho = p['rho']
                sigma_rho = p['srho']

                def gamma_from_inputs_km(Cd, L_km, sigma_L_km, rho, sigma_rho, rho_sw, sigma_rho_sw):
                    rho_ratio = rho / rho_sw
                    g = Cd / (L_km * (rho_ratio + 0.5))  # 1/km
                    sigma_g = math.sqrt(
                        (Cd ** 2 * sigma_L_km ** 2) / (((rho_ratio + 0.5) ** 2) * (L_km ** 4)) +
                        (Cd ** 2 * sigma_rho ** 2) / (((rho_ratio + 0.5) ** 4) * (L_km ** 2) * (rho_sw ** 2)) +
                        (16 * rho ** 2 * Cd ** 2 * sigma_rho_sw ** 2) /
                        (((rho_sw + 2 * rho) ** 4) * (L_km ** 2))
                    )
                    return g, sigma_g

                g_km, sg_km = gamma_from_inputs_km(Cd0, L_km, sigma_L_km,
                                                   rho, sigma_rho, rho_sw, sigma_rho_sw)
                p['g_km'] = g_km
                p['sg_km'] = sg_km
                self.lbl_gamma.setText(f"γ: {g_km:.3e} ± {sg_km:.3e} 1/km")

        except Exception as e:
            QMessageBox.critical(self, "SW fit failed", f"{type(e).__name__}: {e}")

    # SOLVE DBM

    def solve_dbm_only(self):
        try:
            if self.data is None:
                raise RuntimeError("Load data first.")
            if not hasattr(self, '_cache_params'):
                raise RuntimeError("Compute parameters first.")

            p = self._cache_params
            dt = p['dt']
            v_kms = p['v_kms']
            sigma_v_kms = p['sv_kms']
            L_km = p['L_km']
            sigma_L_km = p['sL_km']
            rho, sigma_rho = p['rho'], p['srho']
            rho_sw, sigma_rho_sw = p['rho_sw'], p['srho_sw']
            w_kms, sigma_w_kms = p['w_kms'], p['sw_kms']
            Cd0 = p['Cd0']

            max_iter = int(self.spnMaxIter.value())
            user_v0_guess = float(self.spnV0Guess.value())
            user_T_guess = float(self.spnTGuess.value())

            def gamma_from_inputs_km(Cd, L_km, sigma_L_km, rho, sigma_rho, rho_sw, sigma_rho_sw):
                rho_ratio = rho / rho_sw
                g = Cd / (L_km * (rho_ratio + 0.5))  # 1/km
                sigma_g = math.sqrt(
                    (Cd ** 2 * sigma_L_km ** 2) / (((rho_ratio + 0.5) ** 2) * (L_km ** 4)) +
                    (Cd ** 2 * sigma_rho ** 2) / (((rho_ratio + 0.5) ** 4) * (L_km ** 2) * (rho_sw ** 2)) +
                    (16 * rho ** 2 * Cd ** 2 * sigma_rho_sw ** 2) /
                    (((rho_sw + 2 * rho) ** 4) * (L_km ** 2))
                )
                return g, sigma_g

            Rs_km = 6.957e5
            r0_km = 20.0 * Rs_km

            if self.data.R_sun.size > 1:
                mask_mo = (self.data.t_pos >= self.narrow1_t) & (self.data.t_pos <= self.narrow2_t)
                r_series_km = self.data.R_sun[mask_mo]
                if r_series_km.size >= 2:
                    r_km = 0.5 * (r_series_km[0] + r_series_km[-1])
                    sigma_r_km = 0.5 * abs(r_series_km[-1] - r_series_km[0])
                else:
                    r_km = r0_km
                    sigma_r_km = 0.05 * r0_km
            else:
                r_km = r0_km
                sigma_r_km = 0.05 * r0_km

            vt_kms = v_kms + sigma_v_kms
            rt_km = r_km + sigma_r_km
            w = w_kms

            g_eff, sg_eff = gamma_from_inputs_km(Cd0, L_km, sigma_L_km,
                                                 rho, sigma_rho, rho_sw, sigma_rho_sw)

            def systems(g, sigma_g, w, rt, vt):
                import math as m

                def s_pp(p):
                    v0, t = p
                    return (
                        rt + (t * (v0 - w) / (g * (1 + g * (v0 - w) * t)) -
                              m.log(1 + g * (v0 - w) * t) / g ** 2) * sigma_g
                        - (1 / g) * m.log(1 + g * (v0 - w) * t) - w * t - r0_km,
                        vt + (t * (v0 - w) ** 2 / (1 + g * (v0 - w) * t)) * sigma_g
                        - (v0 - w) / (1 + g * (v0 - w) * t) - w
                    )

                def s_pm(p):
                    v0, t = p
                    return (
                        rt + (t * (v0 - w) / (g * (1 + g * (v0 - w) * t)) -
                              m.log(1 + g * (v0 - w) * t) / g ** 2) * sigma_g
                        - (1 / g) * m.log(1 + g * (v0 - w) * t) - w * t - r0_km,
                        vt - (t * (v0 - w) ** 2 / (1 + g * (v0 - w) * t)) * sigma_g
                        - (v0 - w) / (1 + g * (v0 - w) * t) - w
                    )

                def s_mp(p):
                    v0, t = p
                    return (
                        rt - (t * (v0 - w) / (g * (1 + g * (v0 - w) * t)) -
                              m.log(1 + g * (v0 - w) * t) / g ** 2) * sigma_g
                        - (1 / g) * m.log(1 + g * (v0 - w) * t) - w * t - r0_km,
                        vt + (t * (v0 - w) ** 2 / (1 + g * (v0 - w) * t)) * sigma_g
                        - (v0 - w) / (1 + g * (v0 - w) * t) - w
                    )

                def s_mm(p):
                    v0, t = p
                    return (
                        rt - (t * (v0 - w) / (g * (1 + g * (v0 - w) * t)) -
                              m.log(1 + g * (v0 - w) * t) / g ** 2) * sigma_g
                        - (1 / g) * m.log(1 + g * (v0 - w) * t) - w * t - r0_km,
                        vt - (t * (v0 - w) ** 2 / (1 + g * (v0 - w) * t)) * sigma_g
                        - (v0 - w) / (1 + g * (v0 - w) * t) - w
                    )

                def m_pp(p):
                    v0, t = p
                    return (
                        rt + (t * (v0 - w) / (g * (1 - g * (v0 - w) * t)) +
                              m.log(1 - g * (v0 - w) * t) / g ** 2) * sigma_g
                        + (1 / g) * m.log(1 - g * (v0 - w) * t) - w * t - r0_km,
                        vt + (t * (v0 - w) ** 2 / (1 - g * (v0 - w) * t)) * sigma_g
                        - (v0 - w) / (1 - g * (v0 - w) * t) - w
                    )

                def m_pm(p):
                    v0, t = p
                    return (
                        rt - (t * (v0 - w) / (g * (1 - g * (v0 - w) * t)) +
                              m.log(1 - g * (v0 - w) * t) / g ** 2) * sigma_g
                        + (1 / g) * m.log(1 - g * (v0 - w) * t) - w * t - r0_km,
                        vt + (t * (v0 - w) ** 2 / (1 - g * (v0 - w) * t)) * sigma_g
                        - (v0 - w) / (1 - g * (v0 - w) * t) - w
                    )

                def m_mp(p):
                    v0, t = p
                    return (
                        rt + (t * (v0 - w) / (g * (1 - g * (v0 - w) * t)) +
                              m.log(1 - g * (v0 - w) * t) / g ** 2) * sigma_g
                        + (1 / g) * m.log(1 - g * (v0 - w) * t) - w * t - r0_km,
                        vt - (t * (v0 - w) ** 2 / (1 - g * (v0 - w) * t)) * sigma_g
                        - (v0 - w) / (1 - g * (v0 - w) * t) - w
                    )

                def m_mm(p):
                    v0, t = p
                    return (
                        rt - (t * (v0 - w) / (g * (1 - g * (v0 - w) * t)) +
                              m.log(1 - g * (v0 - w) * t) / g ** 2) * sigma_g
                        + (1 / g) * m.log(1 - g * (v0 - w) * t) - w * t - r0_km,
                        vt - (t * (v0 - w) ** 2 / (1 - g * (v0 - w) * t)) * sigma_g
                        - (v0 - w) / (1 - g * (v0 - w) * t) - w
                    )

                return [s_pp, s_pm, s_mp, s_mm, m_pp, m_pm, m_mp, m_mm]

            fset = systems(g_eff, sg_eff, w, rt_km, vt_kms)

            sols_v0, sols_t = [], []

            v0_guesses = [
                max(50.0, v_kms),
                max(50.0, vt_kms),
                max(50.0, v_kms + 100.0),
                max(50.0, w + 50.0),
            ]
            if np.isfinite(user_v0_guess) and user_v0_guess > 0:
                v0_guesses.append(user_v0_guess)

            dist_km = max(1.0, r_km - r0_km)
            base_T = dist_km / max(1.0, abs(v_kms))
            base_Ts = [base_T, 0.5 * base_T, 2 * base_T, 4 * base_T, dt, 2 * dt]
            T_guesses = [max(1000.0, float(Tg)) for Tg in base_Ts if np.isfinite(Tg) and Tg > 0]
            if np.isfinite(user_T_guess) and user_T_guess > 0:
                T_guesses.append(user_T_guess)

            for f in fset:
                for v0i in v0_guesses:
                    for Ti in T_guesses:
                        try:
                            sol = fsolve(f, (v0i, Ti), maxfev=max_iter)
                            v0, Tsol = float(sol[0]), float(sol[1])
                            if (np.isfinite(v0) and np.isfinite(Tsol) and
                                    Tsol > 0 and 0 < v0 < 4000.0):
                                sols_v0.append(v0)
                                sols_t.append(Tsol)
                        except Exception:
                            pass

            if len(sols_v0) == 0:
                raise RuntimeError("No DBM solutions found for given Cd and initial guesses.")

            v0_solutions_kms = np.array(sols_v0)
            T_solutions = np.array(sols_t)

            v0_min, v0_max = float(np.nanmin(v0_solutions_kms)), float(np.nanmax(v0_solutions_kms))
            T_min, T_max = float(np.nanmin(T_solutions)), float(np.nanmax(T_solutions))

            dep_start = self.wide1_t - T_max
            dep_end = self.narrow1_t - T_min
            self.last_dep_window = (dep_start, dep_end)
            self.last_v0_window = (v0_min, v0_max)

            self.lbl_v0win.setText(f"v0 window: [{v0_min:.1f}, {v0_max:.1f}] km/s")
            self.lbl_Tiwin.setText(
                f"T_i window: [{T_min:.1f}, {T_max:.1f}] s "
                f"(≈ [{T_min / 3600.0:.2f}, {T_max / 3600.0:.2f}] h)"
            )
            self.lbl_depwin.setText(
                f"20 R☉ departure: {to_utc(dep_start)}  —  {to_utc(dep_end)}"
            )

        except Exception as e:
            QMessageBox.critical(self, "DBM failed", f"{type(e).__name__}: {e}")

    #  SAVE PLOTS AS IMAGE

    def save_all_plots_image(self):
        try:
            if self.data is None:
                raise RuntimeError("Load data first.")

            out_dir = QFileDialog.getExistingDirectory(self, "Choose output directory")
            if not out_dir:
                return

            mask = self._plot_mask()
            t_plot = self.t_mpl[mask]
            Babs = self.Babs[mask]
            Br = self.Br[mask]
            Bt = self.Bt[mask]
            Bn = self.Bn[mask]
            Tp = self.Tp[mask]
            Vrad = self.Vrad[mask]
            Np = self.Np[mask]
            beta = self.beta[mask]

            fig = Figure(figsize=(12, 10), tight_layout=True)
            axs = [
                fig.add_subplot(511),
                fig.add_subplot(512),
                fig.add_subplot(513),
                fig.add_subplot(514),
                fig.add_subplot(515),
            ]
            formatter = mdates.DateFormatter('%Y-%m-%d\n%H:%M:%S')
            for ax in axs:
                ax.xaxis_date()
                ax.xaxis.set_major_formatter(formatter)
                ax.grid(True, alpha=0.25)

            axs[0].plot(t_plot, Babs, lw=1.2, color='k', label='|B|')
            axs[0].plot(t_plot, Br, lw=0.9, color='tab:red', label='B_R')
            axs[0].plot(t_plot, Bt, lw=0.9, color='tab:green', label='B_T')
            axs[0].plot(t_plot, Bn, lw=0.9, color='tab:blue', label='B_N')
            axs[0].set_ylabel('B [nT]')
            axs[0].legend(loc='upper right', frameon=False)

            axs[1].plot(t_plot, Tp, lw=1.2, color='k')
            axs[1].set_ylabel('T [eV]')

            axs[2].plot(t_plot, Vrad, lw=1.2, color='k')
            axs[2].set_ylabel('v [km/s]')

            axs[3].plot(t_plot, Np, lw=1.2, color='k')
            axs[3].set_ylabel('n [cm$^{-3}$]')

            axs[4].set_yscale('log')
            axs[4].plot(t_plot, beta, lw=1.2, color='k')
            axs[4].set_ylabel('β')

            times_colors = [
                (self.sheath_t, 'red'),
                (self.narrow1_t, 'blue'),
                (self.narrow2_t, 'blue'),
                (self.wide1_t, 'green'),
                (self.wide2_t, 'green')
            ]
            for ax in axs:
                for t, c in times_colors:
                    if np.isfinite(t):
                        ax.axvline(epoch_to_mpl_scalar(t), color=c, lw=1.2, alpha=0.9)

            x0, x1 = self._plot_xlim()
            for ax in axs:
                ax.set_xlim(x0, x1)

            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            path = f"{out_dir}/cme_icme_panels_{stamp}.png"
            fig.savefig(path, dpi=200)
            QMessageBox.information(self, "Saved", f"Saved: {path}")

        except Exception as e:
            QMessageBox.critical(self, "Save failed", f"{type(e).__name__}: {e}")

    # LASCO CATALOG SEARCH

    def search_lasco_catalogue(self):
        try:
            if self.last_dep_window is None or self.last_v0_window is None:
                raise RuntimeError("Please run Solve DBM first.")
            dep_start, dep_end = self.last_dep_window
            v0_min, v0_max = self.last_v0_window

            pre_start = datetime.fromtimestamp(dep_start, tz=timezone.utc) - timedelta(hours=48)
            pre_end = datetime.fromtimestamp(dep_end, tz=timezone.utc) + timedelta(hours=48)
            vmin = max(0.0, v0_min - 150.0)
            vmax = v0_max + 150.0

            def cpa_ok(c):
                try:
                    c = int(c)
                    return (15 <= c <= 165) or (195 <= c <= 345)
                except:
                    return False

            def width_ok(w):
                try:
                    return int(w) > 10
                except:
                    return False

            def speed_in_band(s):
                if s in ('-----', '-------'):
                    return False
                try:
                    val = float(s.replace('*', ''))
                    return vmin <= val <= vmax
                except:
                    return False

            cat_url = "https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/text_ver/univ_all.txt"
            txt = fetch_text(cat_url)
            lines = txt.splitlines()
            data_lines = [ln for ln in lines if re.match(r'^\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}', ln)]

            pre_candidates = []
            for ln in data_lines:
                rec = parse_catalog_line(ln)
                if rec is None:
                    continue
                dt = rec['dt']
                if not (pre_start <= dt <= pre_end):
                    continue
                if not width_ok(rec['width']):
                    continue
                if not cpa_ok(rec['cpa']):
                    continue
                if not (speed_in_band(rec['linear']) or speed_in_band(rec['v20r'])):
                    continue
                pre_candidates.append(rec)

            if not pre_candidates:
                QMessageBox.information(
                    self, "LASCO search",
                    "No catalogue entries passed the pre-filters.\nConsider manual inspection."
                )
                return

            kept = []
            for rec in pre_candidates:
                d = rec['date']
                t = rec['time']
                width = pad_int(rec['width'], 3)
                speed_pad = pad_int(rec['linear'], 4) or pad_int(rec['final'], 4) or pad_int(rec['init'], 4)
                mpa_pad = pad_int(rec['mpa'], 3)
                if None in (width, speed_pad, mpa_pad):
                    continue
                yyyy, mm, dd = d.split('/')
                hh, mi, ss = t.split(':')
                yyyymm = f"{yyyy}_{mm}"
                stamp = f"{yyyy}{mm}{dd}.{hh}{mi}{ss}"
                base = f"https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL_ver2/{yyyymm}/yht/{stamp}.w{width}n.v{speed_pad}.p{mpa_pad}"
                for variant in ("g", "n"):
                    url = base + f"{variant}.yht"
                    yht_text = None
                    try:
                        yht_bytes = fetch_bytes(url, timeout=20)
                        try:
                            yht_text = yht_bytes.decode('utf-8', errors='replace')
                        except:
                            yht_text = yht_bytes.decode('latin-1', errors='replace')
                    except Exception:
                        continue
                    if yht_text is None:
                        continue
                    parsed = parse_yht_for_heights(yht_text)
                    if parsed[0] is None:
                        continue
                    t_sec, h_arr, t0 = parsed
                    t20 = quadratic_time_at_height(t_sec, h_arr, target_h=20.0)
                    if t20 is None:
                        try:
                            idx = np.argsort(np.abs(h_arr - 20.0))[:2]
                            i1, i2 = int(idx[0]), int(idx[1])
                            if h_arr[i1] == h_arr[i2]:
                                continue
                            t_lin = t_sec[i1] + (
                                (20.0 - h_arr[i1]) *
                                (t_sec[i2] - t_sec[i1]) /
                                (h_arr[i2] - h_arr[i1])
                            )
                            if t_lin < 0:
                                continue
                            t20 = t_lin
                        except Exception:
                            continue
                    t20_abs = (t0 + timedelta(seconds=float(t20))).replace(tzinfo=timezone.utc)
                    t20_ts = t20_abs.timestamp()
                    pad12 = 12 * 3600.0
                    if (dep_start - pad12) <= t20_ts <= (dep_end + pad12):
                        kept.append({
                            'rec': rec,
                            't20': t20_abs,
                            'delta_hours': (t20_ts - 0.5 * (dep_start + dep_end)) / 3600.0
                        })
                    break

            if not kept:
                QMessageBox.information(
                    self, "LASCO search",
                    "No CMEs found whose computed 20 R☉ time falls within the window (±12h) after .yht fitting.\n"
                    "Automatic search could not identify a match — manual search is recommended."
                )
                return

            kept.sort(key=lambda k: k['t20'])
            lines = []
            lines.append("LASCO candidates matching filters and 20 R☉ time (±12h):\n")
            lines.append(
                f"Computed 20 R☉ window: {to_utc(self.last_dep_window[0])}  —  {to_utc(self.last_dep_window[1])}"
            )
            lines.append(
                f"v0 window (km/s): [{self.last_v0_window[0]:.1f}, {self.last_v0_window[1]:.1f}]  "
                f"(catalog speed test ±150 km/s)"
            )
            lines.append("")
            for k in kept:
                r = k['rec']
                dtstr = r['date'].replace('/', '-') + ' ' + r['time']
                lines.append(
                    f"- First appearance: {dtstr}  |  CPA={r['cpa']}  Width={r['width']}  "
                    f"Linear={r['linear']} km/s  20R={r['v20r']} km/s  MPA={r['mpa']}\n"
                    f"  → Fitted 20 R☉ time: {k['t20'].strftime('%Y-%m-%d %H:%M:%S')}  "
                    f"(Δ={k['delta_hours']:+.2f} h from window center)"
                )

            dlg = ResultsDialog("LASCO Search Results", "\n".join(lines), self)
            dlg.resize(900, 600)
            dlg.exec_()

        except Exception as e:
            QMessageBox.critical(self, "LASCO search failed", f"{type(e).__name__}: {e}")
            

