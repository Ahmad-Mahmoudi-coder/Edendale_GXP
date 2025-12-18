# SignalsPack Interface Contract

**Version:** 0.1.0  
**Last Updated:** 2024

This document defines the frozen interface contract for the GXP SignalsPack outputs. This interface must remain stable for downstream consumption by the electricity PoC / model repo and later PyPSA.

## Portable Path Configuration

**All config paths are relative to repo root unless absolute.**

The project is designed to be **location-independent** (portable). All file and directory paths in `config.toml` are resolved relative to the repository root directory (detected via `.git` directory or script location). Absolute paths are preserved as-is. CLI arguments can override config values for key paths like `--output-dir` and `--measured-dir`.

## Schema Version

- **Current schema version:** `0.1.0`
- All provenance reports include `schema_version` field
- Manifest file includes `schema_version` in `[signals]` section

## Required Files

For each epoch (year), the following files are generated:

1. **GXP Hourly Signals:** `gxp_hourly_<epoch>.csv`
2. **Grid Emissions Intensity:** `grid_emissions_intensity_<epoch>.csv`
3. **Provenance Reports:** `gxp_hourly_<epoch>_report.json`, `grid_emissions_intensity_<epoch>_report.json`
4. **Upgrade Menu:** `gxp_upgrade_menu.csv` (single file, not per-epoch)
5. **Manifest:** `signals_manifest.toml` (maps epochs to files and SHA256 hashes)
6. **Epoch Registry:** `epochs_registry.csv` (epoch metadata)

## CSV Schemas

### GXP Hourly Signals (`gxp_hourly_<epoch>.csv`)

**Columns (exact order, snake_case):**
1. `gxp_id` (string) - GXP identifier, e.g., "EDN0331"
2. `timestamp_utc` (string) - UTC ISO-8601 with 'Z' suffix, e.g., `2020-01-01T00:00:00Z`
3. `capacity_mw` (float) - GXP capacity available that hour (includes epoch-specific baseline capacity)
4. `baseline_import_mw` (float) - Baseline import at GXP excluding new site electrification increments
5. `reserve_margin_mw` (float) - Reserve margin applied for that epoch (scalar repeated per row)
6. `headroom_mw` (float) - Computed as `max(capacity_mw - baseline_import_mw - reserve_margin_mw, 0)`
7. `tariff_nzd_per_mwh` (float) - Tariff series aligned to UTC timestamps (computed via local tariff clock)
8. `epoch` (int) - The year repeated (helps debugging joins)

**Row Count:**
- Non-leap years: 8760 rows
- Leap years: 8784 rows
- Timestamps must be monotonic, hourly, no gaps/duplicates

**Timestamp Policy:**
- All timestamps use UTC ISO-8601 format with 'Z' suffix
- Tariff calculations use `tariff_clock_tz` (Pacific/Auckland) with IANA DST policy
- Output timestamps are always UTC

### Grid Emissions Intensity (`grid_emissions_intensity_<epoch>.csv`)

**Columns (exact order, snake_case):**
1. `timestamp_utc` (string) - Same time index as `gxp_hourly_<epoch>.csv` (UTC ISO-8601 with 'Z' suffix)
2. `grid_co2e_kg_per_mwh_avg` (float) - Average grid CO2e intensity
3. `grid_co2e_kg_per_mwh_marginal` (float) - Marginal grid CO2e intensity

**Row Count:**
- Must match exactly the row count of corresponding `gxp_hourly_<epoch>.csv`
- Timestamps must align 1:1 with GXP hourly file

### Upgrade Menu (`gxp_upgrade_menu.csv`)

**Columns (exact order):**
1. `gxp_id` (string)
2. `option_id` (string) - e.g., "U0", "U5", "U15"
3. `delta_capacity_mw` (float) - Capacity change in MW
4. `lead_time_years` (float) - Lead time for upgrade
5. `capex_nzd` (float) - Capital expenditure in NZD
6. `fixed_om_nzd_per_year` (float) - Fixed O&M cost per year
7. `lifetime_years` (int) - Upgrade lifetime
8. `notes` (string) - Short description

**Semantics:**
- Upgrades increase `capacity_mw` (or equivalently headroom) at the chosen epoch
- Lead-time is handled in the orchestration layer
- Single file (not per-epoch)

## Data Semantics

### baseline_import_mw vs headroom_mw

- **baseline_import_mw:** Baseline import at GXP excluding new site electrification increments. This represents the "background" load that exists before any new electrification projects are added.

- **headroom_mw:** Available capacity for new loads. Computed as:
  ```
  headroom_mw = max(capacity_mw - baseline_import_mw - reserve_margin_mw, 0)
  ```
  This is the capacity available for new electrification projects. When headroom is near zero, the GXP is "binding" (at capacity).

### Reserve Margin

- Applied per-epoch via `reserve_margin_by_epoch` config
- Represents operational reserve capacity
- Subtracted from available capacity when computing headroom

### Upgrade Menu Semantics

- **delta_capacity_mw:** Change in capacity (positive = increase)
- **lead_time_years:** Years required before upgrade becomes available
- **capex_nzd:** One-time capital cost (real or nominal, specified in report)
- **fixed_om_nzd_per_year:** Annual fixed operations & maintenance cost
- **lifetime_years:** Expected lifetime of upgrade

## Provenance Reports

Each CSV file has a corresponding `*_report.json` file containing:

- **Schema version**
- **Epoch and epoch semantics** (e.g., "baseline_pre_EB", "decision_epoch")
- **Time policy** (timestamp output TZ, tariff clock TZ, DST policy)
- **Random number generation** (base seed, epoch seed, RNG bit generator)
- **Reserve margin applied**
- **Calibration settings** (if enabled)
- **Files object** with nested structure:
  ```json
  "files": {
    "gxp_hourly_csv": {
      "path": "/absolute/path/to/file.csv",
      "sha256": "full_64_hex_hash"
    },
    "emissions_csv": {
      "path": "/absolute/path/to/file.csv",
      "sha256": "full_64_hex_hash"
    }
  }
  ```
- **Backwards compatibility:** Top-level `file_hash_sha256` alias points to `files.gxp_hourly_csv.sha256`
- **Validation summary** (n_rows, min_headroom_mw, pct_headroom_lt_1mw, etc.)
- **Git information** (commit hash, dirty status, repo root)

## Manifest and Registry

### signals_manifest.toml

Top-level `[signals]` section with:
- `schema_version`
- `generated_utc` (UTC ISO-8601 with 'Z' suffix)
- `gxp_id`
- `timestamp_policy` (timestamp_output_tz, tariff_clock_tz, dst_policy)

Per-epoch sections `[epoch.<YEAR>]` with:
- File paths (relative to output directory)
- SHA256 hashes for each file type

### epochs_registry.csv

Columns:
- `epoch` (int) - Year
- `epoch_semantics` (string) - Semantic label
- `year` (int) - Same as epoch
- `n_hours` (int) - Number of hours (8760 or 8784)
- `start_timestamp_utc` (string) - First timestamp
- `end_timestamp_utc` (string) - Last timestamp

## Hash Integrity

All file hashes use SHA256 (64 hex characters). Hash validation:

1. **Manifest hashes** must match computed file hashes
2. **Report JSON embedded hashes** must match computed CSV hashes
3. Validation mode (`--validate-only`) checks all hash integrity

## Interface Stability

**This interface is frozen.** Changes to:
- Column names or order
- Timestamp format
- File naming conventions
- Provenance report structure (except additions)

...will break downstream consumers.

**If new signals are needed:**
- Add new files, not new columns
- Maintain backwards compatibility
- Update schema version if breaking changes are unavoidable

## Quickstart (Portable)

The project runs from any location. All paths are resolved relative to the repository root.

### Generate Signals

From any directory:
```powershell
# From repo root
# Note: Epoch list must be quoted in PowerShell to prevent comma parsing issues
python .\build_gxp_inputs.py --config .\config.toml --mode synthetic --output-dir .\outputs_latest --clean-output archive --epochs "2020,2025,2028,2035" --seed 42 --write-manifest --stop-on-error

# Or from any other directory (paths are resolved relative to repo root)
cd C:\SomeOtherDir
python "C:\Model\Edendale GXP\build_gxp_inputs.py" --config "C:\Model\Edendale GXP\config.toml" --mode synthetic --output-dir "C:\Model\Edendale GXP\outputs_latest" --epochs "2020,2025,2028,2035" --seed 42 --write-manifest
```

### Validate Outputs

```powershell
# From repo root
python .\build_gxp_inputs.py --output-dir .\outputs_latest --validate-only
```

**Key Points:**
- Default `--output-dir` is `outputs_latest/` relative to repo root
- Config file paths (e.g., `emi_template.jan2023_path`) are relative to repo root
- CLI paths can be absolute or relative (relative paths resolved against repo root)
- The script automatically detects the repo root via `.git` directory or script location

## Output Archiving

The `--clean-output archive` option creates timestamped snapshots of all interface contract outputs before generating new ones.

### Archive Behavior

When `--clean-output archive` is used:
- **All files** in the output directory root are moved to a timestamped archive folder
- Archive location: `outputs_latest/_archive/YYYYMMDD_HHMMSS/`
- Only **files** are archived (directories like `_archive` itself are excluded)
- Each run creates a new timestamped archive folder

### Files Archived

The archive includes all interface contract artifacts from the previous run:
- `gxp_hourly_*.csv` and `gxp_hourly_*_report.json` (all epochs)
- `grid_emissions_intensity_*.csv` and `grid_emissions_intensity_*_report.json` (all epochs)
- `gxp_upgrade_menu.csv` and `gxp_upgrade_menu_report.json`
- `signals_manifest.toml`
- `epochs_registry.csv`
- Any other files in the output directory root

### Archive Summary

The script prints a summary message showing:
- Total number of files archived
- Archive folder name
- Breakdown by file type (e.g., "4 GXP hourly CSV, 4 GXP hourly reports, 1 manifest, 1 registry")

### Notes

- **Output directories are gitignored**: The `outputs_latest/` and `_archive/` directories are excluded from version control (see `.gitignore`)
- **Archiving is the mechanism for run-history snapshots**: Use `--clean-output archive` to preserve previous run outputs before generating new ones
- **Archive folders accumulate**: Each run with `--clean-output archive` creates a new timestamped folder; old archives are not automatically cleaned

### Example

```powershell
# First run - generates outputs
python .\build_gxp_inputs.py --config .\config.toml --mode synthetic --output-dir .\outputs_latest --epochs "2020,2025,2028,2035"

# Second run - archives previous outputs, then generates new ones
python .\build_gxp_inputs.py --config .\config.toml --mode synthetic --output-dir .\outputs_latest --clean-output archive --epochs "2020,2025,2028,2035"

# Result: Previous outputs are in outputs_latest/_archive/20240101_120000/ (example timestamp)
#         New outputs are in outputs_latest/
```

## Validation

Run validation mode to check integrity:
```bash
python build_gxp_inputs.py --output-dir outputs_latest --validate-only
```

This validates:
- Schema exactness (column names and order)
- Timestamp format and monotonicity
- Headroom calculation correctness
- Hash integrity (manifest and report JSONs)
- File presence and completeness

## Hash Integrity Verification

To verify that CSV file hashes match the embedded hashes in report JSONs, use the provided PowerShell script:

```powershell
.\scripts\verify_hashes.ps1 -OutputDir "outputs_latest"
```

The script automatically discovers all `*_report.json` files in the output directory, reads CSV paths from the nested `files.*.path` fields (with fallback to top-level `file_hash_sha256` for backwards compatibility), computes SHA256 hashes, and compares them against the `files.*.sha256` values in the reports. It prints PASS/FAIL status grouped by file type (GXP hourly, Emissions, Upgrade menu).

See [`scripts/verify_hashes.ps1`](../scripts/verify_hashes.ps1) for implementation details.

