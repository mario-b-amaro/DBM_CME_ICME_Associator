# Changelog

All notable changes to this project are documented in this file.

## [1.1.1] - 2026-04-28
### Fixed
- Added pySPEDAS compatibility shim so mission loaders work with both legacy top-level namespaces (`pyspedas.solo`, etc.) and newer namespace layout (`pyspedas.projects.solo`, etc.).
- Resolved `ImportError: cannot import name projects from pyspedas` on environments with older pySPEDAS versions.

## [1.1.0] - 2026-04-28
### Fixed
- Fixed Solar Orbiter mission loading for current `pyspedas` API by using `pyspedas.projects.solo.*` instead of the removed top-level `pyspedas.solo` namespace.
- Standardized all mission loaders (PSP/SOLO/WIND/ACE) to use `pyspedas.projects.*` for compatibility with `pyspedas>=2.1.0`.

### Changed
- Refactored monolithic GUI script into modular package structure:
  - `dbm_associator/utils.py`
  - `dbm_associator/data_loader.py`
  - `dbm_associator/plotting.py`
  - `dbm_associator/sw_fit.py`
  - `dbm_associator/gui.py`
  - `dbm_associator/main.py`
- Kept `DBM_AssociatorGUI_v1.0.py` as a backward-compatible launcher.
- Added explicit project version metadata in `dbm_associator/version.py`.
- Added GUI window title version tag (`v1.1.0`).

### Documentation
- Added this changelog and versioning baseline from v1.0.
- Updated README with new module layout and version notes.
