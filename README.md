# DBM CME-ICME Association Tool

A simple GUI to associate an in-situ ICME to its likely solar CME source by **back-propagating** the event via the **Drag-Based Model (DBM)** and **propagating uncertainties** to obtain:
- a **departure time window at 20 R☉**
- an **initial speed window at 20 R☉**
- a **LASCO catalogue candidate list** consistent with those windows

This implementation follows the method described in (in preparation).

---

## What the GUI does

1. **Loads in-situ time series** (PSP/WIND/SOLO/ACE):
   - Magnetic field components and |B|
   - Proton density `n`
   - Flow speed `v` (radial/bulk)
   - Proton temperature `T` (when available)
   - Plasma beta `β` (computed)

2. **User-defined event boundaries** (manual, in UTC):
   - Sheath start
   - Magnetic Obstacle (MO) *narrow* interval (start/end)
   - Magnetic Obstacle (MO) *wide* interval (start/end)

3. **Computes DBM inputs** from the MO and pre-sheath solar wind:
   - MO duration `Δt` and uncertainty `σΔt` from wide vs narrow windows
   - MO speed `v` and uncertainty `σv` from averages in both windows
   - MO length `L = v Δt` and uncertainty `σL`
   - MO density `ρ` and uncertainty `σρ`
   - Solar-wind speed `w` and density `ρ_sw` from a pre-sheath interval (default: 24 h)

4. **Solves the DBM back in time** to produce:
   - `v0` window (initial speed at 20 R☉)
   - `T` window (propagation time)
   - departure-time window at 20 R☉

5. **Searches the LASCO CDAW CME catalogue** using the derived windows and (when available) `.yht` height–time fits to estimate when each CME reaches 20 R☉.

---

## Installation

### 1) Create an environment (recommended)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to use the GUI

### Step 1 — Load data

1. Select **Mission** (PSP / SOLO / WIND / ACE).
2. Set **Trange start/end (UTC)** in format: `YYYY-MM-DD HH:MM:SS`
3. Click **Load Data**.

The plots will populate and the boundary input fields will be auto-filled if empty.

### Step 2 — Mark boundaries

Enter the following timestamps (UTC):

* **Sheath start**
* **MO narrow start/end**
* **MO wide start/end**

Then click **Update Boundaries** to redraw vertical markers:

* red = sheath start
* blue = MO narrow boundaries
* green = MO wide boundaries

### Step 3 — Compute parameters

Click **Compute Parameters**.

This fills labels for:

* `dt`, `v`, `L`
* `ρ` (MO density)
* `ρ_sw` and `w` (pre-sheath solar wind)
* `γ` (drag parameter)

### Optional — Fit solar-wind histograms (multi-Gaussian)

If the pre-sheath histograms are clearly multi-modal, click **Fit SW Histograms…**.

* Choose number of Gaussians for density and speed
* Click **Apply & Close**
  This updates `ρ_sw`, `w`, and recomputes `γ`.

### Step 4 — Solve DBM

Click **Solve DBM**.

The GUI reports:

* `v0 window` at 20 R☉
* `T_i window` (propagation time range)
* `20 R☉ departure window` (UTC)

### Step 5 — Search LASCO catalogue

Click **Search LASCO Catalogue** (only after solving DBM).

The tool:

* downloads the CDAW LASCO universal catalogue text file
* filters by time and speed windows
* attempts to compute the **time at 20 R☉** using CDAW `.yht` files
* shows matching candidates in a results dialog

### Export plots

Click **Save All Plots as Image…** to save the 5 stacked panels as a PNG.

---

## Method summary (short)

The CME/ICME propagation is modeled from **20 R☉ outward** using the drag-based model (DBM):

* position and speed evolve as `r(t)` and `v(t)` under an aerodynamic-like drag term controlled by `γ`
* the drag parameter is approximated by

  * `γ = C_D / ( L (ρ/ρ_sw + 1/2) )`
    where `L` is the CME radial size from the in-situ MO duration, and `ρ`/`ρ_sw` are CME and ambient densities

Uncertainties are estimated from:

* the difference between *wide* and *narrow* MO boundary choices
* error propagation into `L` and then into `γ`
* propagation of `σγ` into `σr(t)` and `σv(t)`
* building **8 combinations** (±) for the uncertain DBM system, solving each, then taking min/max to obtain time and speed windows

The output is a **time window at 20 R☉** where one should look for the corresponding CME in coronagraph observations/catalogues.

---

## Notes / limitations

* The method has been primarily tested at distances within and up to 1 AU. Performance at larger heliocentric distances can yield wider windows.
* LASCO catalogue matching is a **candidate filter**, not a final association. There are typically various candidates, and narrowing down further requires validation with coronagraph/EUV context (direction, morphology, GCS, etc.).

---

## Citation

If you find this tool/method useful, please cite the associated paper.

(IN PREPARATION)
