# Edendale GXP SignalsPack Generator

Portable module for generating hourly GXP (Grid Exit Point) signals for the Edendale GXP (EDN0331) used by the Process-Heat-RDM framework. Generates a complete SignalsPack containing hourly headroom, tariff, and emissions intensity data across multiple epochs (decision years), with a frozen interface contract intended for downstream model repository consumption. Currently operates in synthetic mode using RETA-calibrated monthly means and diurnal profiles; planned future enhancements include measured hourly data ingestion and PyPSA-generated signals.

## What is SignalsPack?

SignalsPack is a standardised collection of time-series signals for electricity grid infrastructure modelling. For each GXP (Grid Exit Point), it provides hourly data on available capacity, baseline load, headroom (available capacity for new loads), time-of-use tariffs, and grid emissions intensity. These signals enable downstream models (e.g., Process-Heat-RDM) to assess electrification opportunities, capacity constraints, and economic feasibility across multiple decision epochs (years). The interface contract is frozen to ensure stable integration with downstream consumers.

## Outputs (SignalsPack)

For each epoch (year), the generator produces:

- **GXP Hourly Signals:** `gxp_hourly_<epoch>.csv` — hourly capacity, baseline import, headroom, and TOU tariff
- **Grid Emissions Intensity:** `grid_emissions_intensity_<epoch>.csv` — hourly average and marginal CO₂e intensity
- **Provenance Reports:** `gxp_hourly_<epoch>_report.json`, `grid_emissions_intensity_<epoch>_report.json` — schema version, RNG seeds, validation summaries, SHA256 hashes
- **Upgrade Menu:** `gxp_upgrade_menu.csv` — GXP capacity upgrade options with CAPEX, lead times, O&M
- **Manifest:** `signals_manifest.toml` — maps epochs to files and SHA256 hashes
- **Epoch Registry:** `epochs_registry.csv` — epoch metadata (year, semantics, hour counts, timestamp ranges)

**Epochs** represent decision years (e.g., 2020, 2025, 2028, 2035) with epoch-specific capacity schedules, reserve margins, tariff bases, and scaling factors.

See [`INTERFACE_CONTRACT.md`](INTERFACE_CONTRACT.md) for the authoritative frozen schema, column definitions, timestamp policies, and hash integrity requirements.

## Quickstart

### Prerequisites

- Python 3.11+ (uses `tomllib` from stdlib) or Python 3.7+ with `tomli` package
- Dependencies: `numpy`, `pandas`, `pytz` (recommended), `openpyxl` (optional, for Excel EMI template)

### Clone and Setup

```powershell
# Clone repository
git clone https://github.com/Ahmad-Mahmoudi-coder/Edendale_GXP.git
cd Edendale_GXP

# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Generate Signals

```powershell
# From repo root (works from any directory)
# Note: Epoch list must be quoted in PowerShell to prevent comma parsing issues
python .\build_gxp_inputs.py --config .\config.toml --mode synthetic --output-dir .\outputs_latest --clean-output archive --epochs "2020,2025,2028,2035" --seed 42 --write-manifest --stop-on-error
```

### Validate Outputs

```powershell
python .\build_gxp_inputs.py --output-dir .\outputs_latest --validate-only
```

## Configuration

Configuration is managed via `config.toml` (paths are resolved relative to repo root for portability). Key settings:

- **Epochs:** Configured via `epoch_scaling`, `tariff_base`, `reserve_margin_by_epoch`, `tightness_targets`
- **Synthetic mode:** `monthly_mean_import_mw`, `diurnal`, `noise_sigma_mw`, `seed`, `baseline_floor_mw`
- **EMI template:** `emi_template.jan2023_path` (relative to repo root), `enabled`, `gxp_id`, `units`
- **Tariff:** `tariff_clock_tz` (Pacific/Auckland), TOU multipliers, weekday/weekend hours
- **Capacity:** Seasonal capacity ranges (MVA) with power factor conversion

CLI arguments override config values: `--output-dir`, `--measured-dir`, `--seed`, `--epochs`.

## Provenance & Hash Integrity

Each CSV file has a corresponding `*_report.json` containing:
- Schema version and epoch semantics
- Random number generation (base seed, epoch seed, RNG bit generator)
- Time policy (timestamp output TZ, tariff clock TZ, DST policy)
- Validation summary (row counts, min headroom, tightness percentages)
- **Nested `files` object** with SHA256 hashes:
  ```json
  "files": {
    "gxp_hourly_csv": {"path": "...", "sha256": "..."},
    "emissions_csv": {"path": "...", "sha256": "..."}
  }
  ```

The `signals_manifest.toml` aggregates all epoch file paths and hashes.

### Verify Hash Integrity

Use the provided PowerShell script to verify hash integrity:

```powershell
.\scripts\verify_hashes.ps1 -OutputDir "outputs_latest"
```

The script automatically discovers all `*_report.json` files in the output directory, reads CSV paths from the nested `files.*.path` fields, computes SHA256 hashes, and compares them against the `files.*.sha256` values in the reports. See [`scripts/verify_hashes.ps1`](scripts/verify_hashes.ps1) for implementation details.

The `--validate-only` mode checks schema exactness, timestamp format, headroom calculation correctness, and hash integrity (manifest and report JSONs).

## Repository Hygiene

The `.gitignore` excludes:
- `outputs_latest/` — generated outputs directory
- `_archive/` — timestamped archive folders
- `__pycache__/`, `*.pyc`, `*.log` — Python caches and logs

**Archiving:** When `--clean-output archive` is used, all files in the output directory root are moved to `outputs_latest/_archive/YYYYMMDD_HHMMSS/` before generating new outputs. This preserves complete snapshots of previous runs (all interface contract artefacts: CSVs, reports, manifest, registry). Archive folders accumulate; manual cleanup may be required.

## Release Checklist

Before committing and pushing changes:

1. **Generate outputs for all epochs:**
   ```powershell
   python .\build_gxp_inputs.py --config .\config.toml --mode synthetic --output-dir .\outputs_latest --clean-output archive --epochs "2020,2025,2028,2035" --seed 42 --write-manifest --stop-on-error
   ```

2. **Run validation:**
   ```powershell
   python .\build_gxp_inputs.py --output-dir .\outputs_latest --validate-only
   ```

3. **Verify hash integrity:**
   ```powershell
   .\scripts\verify_hashes.ps1 -OutputDir "outputs_latest"
   ```

4. **Confirm outputs are gitignored:**
   - Verify `outputs_latest/`, `outputs_latest/_archive/`, and `test_output/` are excluded
   - Check that no generated CSV, JSON, or TOML files are staged

5. **Commit and push:**
   - Only commit source code, config, documentation, and scripts
   - Do not commit generated outputs or archives

## Roadmap / Next Steps

- **Measured data ingestion:** Add robust CSV/Excel ingestion for measured hourly GXP data (replacing synthetic mode)
- **PyPSA backend:** Integrate PyPSA-generated signals as an alternative generation source
- **Interface stability:** Maintain frozen schema (`INTERFACE_CONTRACT.md`) for downstream consumers; version bump if breaking changes are unavoidable
- **Schema evolution:** Add new signals via new files (not new columns) to maintain backwards compatibility

