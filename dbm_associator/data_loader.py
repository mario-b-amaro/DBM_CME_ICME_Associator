from dataclasses import dataclass

import numpy as np
import pytplot
from pyspedas import projects as sp_projects

from .utils import first_existing, mag_mag_components_from_tplot


@dataclass
class MissionVars:
    t_mag: np.ndarray
    Br: np.ndarray
    Bt: np.ndarray
    Bn: np.ndarray
    Bmag: np.ndarray
    t_v: np.ndarray
    Vx: np.ndarray
    Vy: np.ndarray
    Vz: np.ndarray
    Vrad: np.ndarray
    t_n: np.ndarray
    Np: np.ndarray
    t_T: np.ndarray
    Tp: np.ndarray
    t_pos: np.ndarray
    R_sun: np.ndarray


class MissionLoader:
    def __init__(self, mission: str, trange):
        self.mission = mission
        self.trange = trange

    def load(self) -> MissionVars:
        m = self.mission.lower()
        if m == 'psp':
            return self._load_psp()
        if m == 'solo':
            return self._load_solo()
        if m == 'wind':
            return self._load_wind()
        if m == 'ace':
            return self._load_ace()
        raise ValueError("Unsupported mission.")

    def _load_psp(self) -> MissionVars:
        sp_projects.psp.fields(trange=self.trange, datatype='mag_rtn_1min', level='l2', time_clip=True)
        t_mag, Br, Bt, Bn = mag_mag_components_from_tplot(['psp_fld_l2_mag_RTN_1min', 'b_mult'])
        if t_mag is None:
            raise RuntimeError("PSP MAG data not found.")
        Bmag = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

        sp_projects.psp.spc(trange=self.trange, datatype='l3i', level='l3', time_clip=True)
        v_vec = first_existing('psp_spc_vp_moment_RTN')
        if v_vec is None:
            raise RuntimeError("PSP velocity moments not found.")
        pytplot.split_vec(v_vec)
        vx = pytplot.get_data(v_vec + '_x')
        vy = pytplot.get_data(v_vec + '_y')
        vz = pytplot.get_data(v_vec + '_z')
        t_v = vx.times
        Vx, Vy, Vz = vx.y, vy.y, vz.y

        n_name = first_existing('psp_spc_np_fit', 'psp_spc_np_moment')
        n_dat = pytplot.get_data(n_name)
        t_n, Np = n_dat.times, n_dat.y

        wp_dat = pytplot.get_data('psp_spc_wp_moment') or pytplot.get_data('psp_spc_wp_fit')
        if wp_dat is None:
            raise RuntimeError("PSP: cannot find wp for T.")
        t_T = wp_dat.times
        wp = wp_dat.y
        Tp = (wp ** 2 * 1.6726e-27) / (2.0 * 1.380649e-23)

        sp_projects.psp.spc(trange=self.trange, datatype='l3i', level='l3', time_clip=True, varnames=['sc_pos_HCI'])
        pos_name = first_existing('psp_spc_sc_pos_HCI', 'dsc')
        R_sun = np.array([])
        t_pos = np.array([])
        if pos_name:
            pytplot.split_vec(pos_name)
            px = pytplot.get_data(pos_name + '_x')
            py = pytplot.get_data(pos_name + '_y')
            pz = pytplot.get_data(pos_name + '_z')
            t_pos = px.times
            R_sun = np.sqrt(px.y ** 2 + py.y ** 2 + pz.y ** 2)

        return MissionVars(np.array(t_mag), Br, Bt, Bn, Bmag, np.array(t_v), Vx, Vy, Vz, Vx,
                           np.array(t_n), Np, np.array(t_T), Tp, np.array(t_pos), R_sun)

    def _load_solo(self) -> MissionVars:
        sp_projects.solo.mag(trange=self.trange, datatype='rtn-normal', level='l2', time_clip=True)
        t_mag, Br, Bt, Bn = mag_mag_components_from_tplot(['B_RTN'])
        if t_mag is None:
            t_mag, Br, Bt, Bn = mag_mag_components_from_tplot(pytplot.tplot_names())
        if t_mag is None:
            raise RuntimeError("SOLO MAG vector not found.")
        Bmag = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

        tried = False
        for dt in ['pas-mom', 'pas-grnd-mom', 'pas-eflux', 'pas-raw-mom']:
            try:
                sp_projects.solo.swa(trange=self.trange, datatype=dt, level='l2', time_clip=True)
                tried = True
            except Exception:
                pass
        if not tried:
            raise RuntimeError("SOLO SWA: no PAS/mom data.")

        names = pytplot.tplot_names()
        cand_n = [n for n in names if n.endswith('_density') or n.lower() in ('n', 'np', 'n_p', 'proton_density')]
        n_name = cand_n[0] if cand_n else first_existing('Np', 'N', 'proton_density')
        if n_name is None:
            raise RuntimeError("SOLO: density not found.")
        n_dat = pytplot.get_data(n_name)
        t_n, Np = n_dat.times, n_dat.y

        v_vec = first_existing('V_RTN', 'vp_RTN', 'V_R', 'velocity_RTN', 'velocity')
        if v_vec is None:
            v_vec = next((nm for nm in names if nm.endswith('_RTN')), None)
        if v_vec is None:
            raise RuntimeError("SOLO: velocity not found.")
        pytplot.split_vec(v_vec)
        vx = pytplot.get_data(v_vec + '_x')
        vy = pytplot.get_data(v_vec + '_y')
        vz = pytplot.get_data(v_vec + '_z')
        t_v, Vx, Vy, Vz = vx.times, vx.y, vy.y, vz.y

        T_name = first_existing('T', 'Tp', 'proton_temperature', 'Tpr')
        if T_name is not None:
            T_dat = pytplot.get_data(T_name)
            t_T, Tp = T_dat.times, T_dat.y
        else:
            t_T = t_n
            Tp = np.full_like(Np, np.nan, dtype=float)

        return MissionVars(np.array(t_mag), Br, Bt, Bn, Bmag, np.array(t_v), Vx, Vy, Vz, Vx,
                           np.array(t_n), Np, np.array(t_T), Tp, np.array([]), np.array([]))

    def _load_wind(self) -> MissionVars:
        sp_projects.wind.mfi(trange=self.trange, time_clip=True)
        t_mag, Br, Bt, Bn = mag_mag_components_from_tplot(['BGSE', 'BGSEc'])
        if t_mag is None:
            t_mag, Br, Bt, Bn = mag_mag_components_from_tplot(pytplot.tplot_names())
        if t_mag is None:
            raise RuntimeError("Wind MFI vector not found.")
        Bmag = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

        sp_projects.wind.swe(trange=self.trange, time_clip=True)
        names = pytplot.tplot_names()
        n_name = first_existing('Np', 'proton_density', 'N_p', 'density')
        if n_name is None:
            dens = [n for n in names if 'density' in n.lower()]
            n_name = dens[0] if dens else None
        if n_name is None:
            raise RuntimeError("Wind SWE: density not found.")
        n_dat = pytplot.get_data(n_name)

        v_name = first_existing('Vp', 'flow_speed', 'V_GSE', 'V')
        if v_name is not None:
            v_dat = pytplot.get_data(v_name)
            t_v = v_dat.times
            Vrad = v_dat.y
            Vx, Vy, Vz = Vrad, np.zeros_like(Vrad), np.zeros_like(Vrad)
        else:
            v_vec = first_existing('VGSE', 'V_GSE', 'velocity_gse', 'velocity')
            if v_vec is None:
                v_vec = next((nm[:-2] for nm in names if nm.endswith('_x') and 'v' in nm.lower()
                              and (nm[:-2] + '_y') in names and (nm[:-2] + '_z') in names), None)
            if v_vec is None:
                raise RuntimeError("Wind SWE: velocity not found.")
            pytplot.split_vec(v_vec)
            vx = pytplot.get_data(v_vec + '_x')
            vy = pytplot.get_data(v_vec + '_y')
            vz = pytplot.get_data(v_vec + '_z')
            t_v, Vx, Vy, Vz = vx.times, vx.y, vy.y, vz.y
            Vrad = Vx

        T_name = first_existing('Tpr', 'Tp', 'proton_temperature', 'Temperature')
        if T_name is None:
            t_T, Tp = n_dat.times, np.full_like(n_dat.y, np.nan, dtype=float)
        else:
            T_dat = pytplot.get_data(T_name)
            t_T, Tp = T_dat.times, T_dat.y

        return MissionVars(np.array(t_mag), Br, Bt, Bn, Bmag, np.array(t_v), Vx, Vy, Vz, Vrad,
                           np.array(n_dat.times), n_dat.y, np.array(t_T), Tp, np.array([]), np.array([]))

    def _load_ace(self) -> MissionVars:
        sp_projects.ace.mfi(trange=self.trange, time_clip=True)
        t_mag, Br, Bt, Bn = mag_mag_components_from_tplot(['BGSEc', 'BGSE'])
        if t_mag is None:
            t_mag, Br, Bt, Bn = mag_mag_components_from_tplot(pytplot.tplot_names())
        if t_mag is None:
            raise RuntimeError("ACE MFI vector not found.")
        Bmag = np.sqrt(Br ** 2 + Bt ** 2 + Bn ** 2)

        sp_projects.ace.swe(trange=self.trange, time_clip=True)
        n_name = first_existing('Np', 'proton_density', 'N_p')
        if n_name is None:
            raise RuntimeError("ACE: density not found.")
        n_dat = pytplot.get_data(n_name)

        v_name = first_existing('Vp')
        if v_name is None:
            raise RuntimeError("ACE: Vp not found.")
        v_dat = pytplot.get_data(v_name)
        t_v, Vrad = v_dat.times, v_dat.y
        Vx, Vy, Vz = Vrad, np.zeros_like(Vrad), np.zeros_like(Vrad)

        T_name = first_existing('Tpr', 'Tp', 'proton_temperature')
        if T_name is None:
            t_T, Tp = n_dat.times, np.full_like(n_dat.y, np.nan, dtype=float)
        else:
            T_dat = pytplot.get_data(T_name)
            t_T, Tp = T_dat.times, T_dat.y

        return MissionVars(np.array(t_mag), Br, Bt, Bn, Bmag, np.array(t_v), Vx, Vy, Vz, Vrad,
                           np.array(n_dat.times), n_dat.y, np.array(t_T), Tp, np.array([]), np.array([]))
