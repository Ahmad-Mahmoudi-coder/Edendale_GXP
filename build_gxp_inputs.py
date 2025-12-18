#!/usr/bin/env python3
"""
Standalone utility to generate hourly GXP input CSVs for specified epochs.
Supports synthetic (RETA-shaped) and measured data modes.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
import warnings
import hashlib
import json
import subprocess

import numpy as np
import pandas as pd

# Try to import pytz for timezone handling, but make it optional
try:
    import pytz
    HAS_PYTZ = True
except ImportError:
    HAS_PYTZ = False
    warnings.warn("pytz not installed. Timezone handling will be naive. Install with: pip install pytz")

# Try tomllib (Python 3.11+), fallback to tomli
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("Error: tomllib (Python 3.11+) or tomli package required.")
        print("Install with: pip install tomli")
        sys.exit(1)

# Frozen schema constants - DO NOT CHANGE
REQUIRED_GXP_COLUMNS = [
    "gxp_id",
    "timestamp_utc",
    "capacity_mw",
    "baseline_import_mw",
    "reserve_margin_mw",
    "headroom_mw",
    "tariff_nzd_per_mwh",
    "epoch",
]

REQUIRED_EMISS_COLUMNS = [
    "timestamp_utc",
    "grid_co2e_kg_per_mwh_avg",
    "grid_co2e_kg_per_mwh_marginal",
]


def format_timestamp_utc(dt_like) -> pd.Series:
    """
    Canonical formatter for UTC ISO-8601 timestamps with 'Z' suffix.
    
    Accepts:
    - pd.Series of datetime-like objects
    - pd.DatetimeIndex
    - list/array of datetime-like objects
    
    Returns:
    - pd.Series of strings formatted as 'YYYY-MM-DDTHH:MM:SSZ'
    
    Always converts to UTC and appends 'Z' suffix for unambiguous UTC representation.
    """
    # Convert to DatetimeIndex in UTC
    if isinstance(dt_like, pd.DatetimeIndex):
        # If already timezone-aware, convert to UTC; otherwise assume UTC
        if dt_like.tz is None:
            dt_utc = dt_like.tz_localize('UTC')
        else:
            dt_utc = dt_like.tz_convert('UTC')
    elif isinstance(dt_like, pd.Series):
        dt_utc = pd.to_datetime(dt_like, utc=True)
    else:
        # Handle list/array
        dt_utc = pd.to_datetime(dt_like, utc=True)
    
    # Format as UTC ISO-8601 with Z suffix
    if isinstance(dt_utc, pd.DatetimeIndex):
        formatted = dt_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
    else:
        formatted = dt_utc.dt.strftime('%Y-%m-%dT%H:%M:%SZ')
    
    return pd.Series(formatted)


def load_config(config_path: Path) -> dict:
    """Load and parse TOML configuration file."""
    with open(config_path, 'rb') as f:
        return tomllib.load(f)


def get_repo_root(script_path: Optional[Path] = None) -> Path:
    """
    Get the repository root directory by walking up from script location or current directory.
    Looks for .git directory to identify repo root.
    Returns the repo root Path, or current working directory if not found.
    """
    if script_path is None:
        script_path = Path(__file__).resolve()
    
    # Start from script's parent directory
    current = script_path.resolve().parent
    
    # Walk up the directory tree to find .git
    while current != current.parent:
        git_dir = current / '.git'
        if git_dir.exists():
            return current
        current = current.parent
    
    # If no .git found, return script's parent directory as fallback
    return script_path.resolve().parent


def resolve_path(path: str | Path, base: Path) -> Path:
    """
    Resolve a path relative to a base directory.
    
    Args:
        path: Path string or Path object (can be absolute or relative)
        base: Base directory to resolve relative paths against
    
    Returns:
        Path object:
        - If path is absolute, returns it unchanged (as Path)
        - If path is relative, returns base / path (resolved)
    """
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    else:
        return (base / path_obj).resolve()


def sha256_file(path: Path) -> str:
    """
    Compute SHA256 hash of a file using read_bytes().
    Returns lowercase hex digest (64 characters).
    """
    return hashlib.sha256(path.read_bytes()).hexdigest().lower()


# Backwards compatibility alias
def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file (alias for sha256_file)."""
    return sha256_file(file_path)


def get_git_commit_hash(output_dir: Optional[Path] = None, script_path: Optional[Path] = None) -> Tuple[Optional[str], bool]:
    """
    Get current git commit hash if available.
    Tries to detect git repo from output directory or script location.
    Returns (commit_hash, git_detected) tuple.
    """
    # Try multiple paths to find git repo
    search_paths = []
    
    if output_dir:
        search_paths.append(output_dir.resolve())
    if script_path:
        search_paths.append(script_path.resolve().parent)
    search_paths.append(Path.cwd())
    
    for search_path in search_paths:
        # Walk up the directory tree to find .git
        current = search_path
        while current != current.parent:
            git_dir = current / '.git'
            if git_dir.exists():
                # Found git repo, try to get commit hash
                try:
                    result = subprocess.run(
                        ['git', 'rev-parse', 'HEAD'],
                        capture_output=True,
                        text=True,
                        check=True,
                        cwd=current
                    )
                    return result.stdout.strip(), True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Git repo found but can't get commit (maybe detached HEAD or no commits)
                    return None, True
        
            current = current.parent
    
    # No git repo found
    return None, False


def to_jsonable(x):
    """
    Recursively convert numpy/pandas types and other objects to JSON-serializable types.
    Returns (converted_value, warnings_list) where warnings_list contains any conversion issues.
    NumPy 2.0-compatible: uses only np.generic and np.ndarray (no dtype aliases).
    """
    warnings_list = []
    
    # NumPy 2.0-compatible: use only np.generic (for scalars) and np.ndarray (for arrays)
    # Do NOT reference any dtype aliases (np.float_, np.int_, etc.) which were removed in NumPy 2.0
    try:
        if isinstance(x, np.generic):
            # All numpy scalar types (int, float, bool, etc.) - convert via .item()
            return x.item(), warnings_list
        elif isinstance(x, np.ndarray):
            # NumPy arrays - convert to list recursively
            converted_list = []
            for item in x:
                item_val, item_warnings = to_jsonable(item)
                converted_list.append(item_val)
                warnings_list.extend(item_warnings)
            return converted_list, warnings_list
    except (NameError, AttributeError):
        # NumPy not available - skip numpy checks
        pass
    
    # Python native types
    if isinstance(x, bool):
        return bool(x), warnings_list
    elif isinstance(x, pd.Series):
        # Convert array/series to list
        converted_list = []
        for item in x:
            item_val, item_warnings = to_jsonable(item)
            converted_list.append(item_val)
            warnings_list.extend(item_warnings)
        return converted_list, warnings_list
    elif isinstance(x, pd.DataFrame):
        # Convert DataFrame to dict of lists
        converted_dict = {}
        for col in x.columns:
            col_val, col_warnings = to_jsonable(x[col].tolist())
            converted_dict[str(col)] = col_val
            warnings_list.extend(col_warnings)
        return converted_dict, warnings_list
    elif isinstance(x, Path):
        return str(x), warnings_list
    elif isinstance(x, dict):
        converted_dict = {}
        for k, v in x.items():
            key_str = str(k)
            val_json, val_warnings = to_jsonable(v)
            converted_dict[key_str] = val_json
            warnings_list.extend(val_warnings)
        return converted_dict, warnings_list
    elif isinstance(x, (list, tuple)):
        converted_list = []
        for item in x:
            item_json, item_warnings = to_jsonable(item)
            converted_list.append(item_json)
            warnings_list.extend(item_warnings)
        return converted_list, warnings_list
    elif x is None:
        return None, warnings_list
    elif isinstance(x, (str, int, float, bool)):
        return x, warnings_list
    else:
        # Unknown type - convert to string and warn
        warnings_list.append(f"Converted {type(x).__name__} to string: {str(x)[:50]}")
        return str(x), warnings_list


def write_provenance_report(
    output_file: Path,
    year: int,
    config: dict,
    stats: Dict,
    df: pd.DataFrame,
    mode: str,
    script_path: Optional[Path] = None,
    csv_hash: Optional[str] = None
) -> None:
    """Write provenance report JSON file alongside the CSV."""
    # Calculate seasonal means
    df_with_baseline = df.copy()
    df_with_baseline['datetime'] = pd.to_datetime(df_with_baseline['timestamp_utc'], utc=True)
    df_with_baseline['baseline_import_mw'] = stats['baseline_import_mw'].values
    
    jan_mar = df_with_baseline[df_with_baseline['datetime'].dt.month.isin([1, 2, 3])]
    jun_aug = df_with_baseline[df_with_baseline['datetime'].dt.month.isin([6, 7, 8])]
    oct_nov = df_with_baseline[df_with_baseline['datetime'].dt.month.isin([10, 11])]
    
    # Get config values
    synthetic_config = config.get('synthetic', {})
    time_config = config.get('time', {})
    
    # Get git commit hash with improved detection
    git_hash, git_detected = get_git_commit_hash(output_dir=output_file.parent, script_path=script_path)
    
    # Determine epoch semantics
    epoch_semantics = None
    if year == 2020:
        epoch_semantics = "baseline_pre_EB"
    elif year == 2025:
        epoch_semantics = "steady_state_post_EB1"
    elif year == 2028:
        epoch_semantics = "steady_state_post_EB2_EB3"
    elif year == 2035:
        epoch_semantics = "decision_epoch"
    else:
        epoch_semantics = "unknown_epoch"
    
    # Extract time policy settings
    timestamp_output_tz = time_config.get('timestamp_output_tz', 'UTC')
    tariff_clock_tz = time_config.get('tariff_clock_tz', 'Pacific/Auckland')
    dst_policy = time_config.get('dst_policy', 'iana')
    
    # Extract time index info from DataFrame
    # Parse timestamp_utc strings (should be UTC ISO-8601 with Z suffix from CSV)
    df_dt = pd.to_datetime(df['timestamp_utc'], utc=True)
    # Format as ISO-8601 UTC with Z suffix for unambiguous timestamps
    time_index_start = df_dt.min().strftime('%Y-%m-%dT%H:%M:%SZ')
    time_index_end = df_dt.max().strftime('%Y-%m-%dT%H:%M:%SZ')
    n_hours = len(df)
    
    # Get git info (check for dirty status)
    git_hash, git_detected = get_git_commit_hash(output_dir=output_file.parent, script_path=script_path)
    git_dirty = False
    repo_root_path = None
    if git_detected and script_path:
        try:
            repo_root = script_path
            while repo_root != repo_root.parent:
                if (repo_root / '.git').exists():
                    repo_root_path = str(repo_root)
                    result = subprocess.run(
                        ['git', 'status', '--porcelain'],
                        cwd=repo_root,
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    git_dirty = bool(result.stdout.strip())
                    break
                repo_root = repo_root.parent
        except Exception:
            pass
    
    # Get calibration info
    calibration_config = config.get('calibration', {})
    calibration_enabled = calibration_config.get('enabled', False)
    calibration_info = {
        'enabled': calibration_enabled,
        'target_peak_mw': None,
        'peak_match_quantile': None,
        'achieved_peak_mw': None,
        'binding_hours_at_peak': None,
    }
    if calibration_enabled and stats.get('peak_calibration_applied', False):
        calibration_info['target_peak_mw'] = stats.get('peak_calibration_target_MW')
        calibration_info['peak_match_quantile'] = stats.get('peak_calibration_quantile')
        calibration_info['achieved_peak_mw'] = stats.get('peak_calibration_achieved_quantile_MW')
        calibration_info['binding_hours_at_peak'] = int(stats.get('binding_count', 0))
    
    # Build validation summary
    validation_summary = {
        'n_rows': len(df),
        'n_hours': n_hours,
        'min_headroom_mw': float(stats.get('headroom_min', 0)),
        'pct_headroom_lt_1mw': float(stats.get('headroom_lt1_frac', 0)),
        'pct_headroom_lt_5mw': float(stats.get('headroom_lt5_frac', 0)),
        'tariff_min': float(stats.get('tariff_min', 0)),
        'tariff_max': float(stats.get('tariff_max', 0)),
    }
    
    # Compute file hashes
    # Use provided hash if available (computed after file write), otherwise compute now
    if csv_hash is not None:
        gxp_csv_hash = csv_hash.lower()  # Ensure lowercase
    else:
        gxp_csv_hash = sha256_file(output_file).lower()
    
    emissions_file = output_file.parent / f'grid_emissions_intensity_{year}.csv'
    emissions_hash = sha256_file(emissions_file).lower() if emissions_file.exists() else None
    
    report = {
        'schema_version': '0.1.0',
        'generation_timestamp': datetime.now().isoformat(),
        'epoch': year,
        'epoch_semantics': epoch_semantics,
        'mode': mode,
        'time_policy': {
            'timestamp_output_tz': timestamp_output_tz,
            'tariff_clock_tz': tariff_clock_tz,
            'dst_policy': dst_policy,
            'time_index_start': time_index_start,
            'time_index_end': time_index_end,
            'n_hours': n_hours,
        },
        'random_seed_base': stats.get('random_seed_base'),
        'random_seed_epoch': stats.get('random_seed_epoch'),
        'rng_bit_generator': stats.get('rng_bit_generator'),
        'reserve_margin_mw_applied': stats.get('reserve_margin_MW_applied', 0.0),
        'calibration': calibration_info,
        'validation_summary': validation_summary,
        'files': {
            'gxp_hourly_csv': {
                'path': str(output_file.resolve()),
                'sha256': gxp_csv_hash,
            },
            'emissions_csv': {
                'path': str(emissions_file.resolve()) if emissions_file.exists() else None,
                'sha256': emissions_hash if emissions_hash else None,
            },
        },
        # Backwards compatibility alias
        'file_hash_sha256': gxp_csv_hash,
        'git': {
            'commit': git_hash,
            'dirty': git_dirty,
            'repo_root': repo_root_path,
        },
        'config_used': {
            'baseline_floor_MW': synthetic_config.get('baseline_floor_mw', 3.5),
            'monthly_mean_import_mw': synthetic_config.get('monthly_mean_import_mw', []),
            'noise_sigma_mw': synthetic_config.get('noise_sigma_mw', 0.5),
            'rho_ar1': synthetic_config.get('rho_ar1', 0.9),
            'weekly_var_amp_mw': synthetic_config.get('weekly_var_amp_mw', 0.7),
            'weekend_multiplier': synthetic_config.get('weekend_multiplier', 0.97),
            'reserve_margin_MW_requested': stats.get('reserve_margin_MW_requested', 0.0),
            'reserve_margin_MW_applied': stats.get('reserve_margin_MW_applied', 0.0),
            'smooth_clip': synthetic_config.get('smooth_clip', True),
            'smooth_k': synthetic_config.get('smooth_k', 10.0),
        },
        'validation_results': {
            'seasonal_means_MW': {
                'jan_mar': float(jan_mar['baseline_import_mw'].mean()),
                'jun_aug': float(jun_aug['baseline_import_mw'].mean()),
                'oct_nov': float(oct_nov['baseline_import_mw'].mean()),
            },
            'floor_hours': {
                'count': int(stats['floor_hits']),
                'fraction': float(stats['floor_hits'] / len(df)),
                'pct': round(float(stats['floor_hits'] / len(df)) * 100, 2),
            },
            'headroom_statistics': {
                'min_MW': float(stats['headroom_min']),
                'mean_MW': float(stats['headroom_mean']),
                'max_MW': float(stats['headroom_max']),
            },
            'headroom_percentiles_MW': {
                'p1': float(stats.get('headroom_p1', 0)),
                'p5': float(stats.get('headroom_p5', 0)),
                'p10': float(stats.get('headroom_p10', 0)),
                'p50': float(stats.get('headroom_p50', 0)),
                'p90': float(stats.get('headroom_p90', 0)),
            },
            'headroom_threshold_counts': {
                'hours_headroom_lt1_MW': {
                    'count': int(stats.get('headroom_lt1_count', 0)),
                    'fraction': float(stats.get('headroom_lt1_frac', 0)),
                    'pct': round(float(stats.get('headroom_lt1_frac', 0)) * 100, 2),
                },
                'hours_headroom_lt2_MW': {
                    'count': int(stats.get('headroom_lt2_count', 0)),
                    'fraction': float(stats.get('headroom_lt2_frac', 0)),
                    'pct': round(float(stats.get('headroom_lt2_frac', 0)) * 100, 2),
                },
                'hours_headroom_lt5_MW': {
                    'count': int(stats.get('headroom_lt5_count', 0)),
                    'fraction': float(stats.get('headroom_lt5_frac', 0)),
                    'pct': round(float(stats.get('headroom_lt5_frac', 0)) * 100, 2),
                },
                'hours_headroom_lt10_MW': {
                    'count': int(stats.get('headroom_lt10_count', 0)),
                    'fraction': float(stats.get('headroom_lt10_frac', 0)),
                    'pct': round(float(stats.get('headroom_lt10_frac', 0)) * 100, 2),
                },
            },
            'headroom_acceptance_limits': stats.get('headroom_acceptance_limits', {}),
            'incremental_sensitivity': {
                'deltas_mw': stats.get('incremental_sensitivity_deltas', [2, 5, 10, 15]),
                'hours_bind': {},
                'pct_bind': {},
            },
            'emi_template_used': bool(stats.get('emi_template_used', False)),
            'jan_validation': {
                'weekend_weekday_ratio': float(stats.get('jan_weekend_weekday_ratio')) if stats.get('jan_weekend_weekday_ratio') is not None else None,
                'diurnal_range_MW': float(stats.get('jan_diurnal_range')) if stats.get('jan_diurnal_range') is not None else None,
            },
        },
        'summary_statistics': {
            'total_hours': int(len(df)),
            'baseline_import_mw': {
                'min': float(stats['baseline_min']),
                'mean': float(stats['baseline_mean']),
                'max': float(stats['baseline_max']),
            },
            'headroom_MW': {
                'min': float(stats['headroom_min']),
                'mean': float(stats['headroom_mean']),
                'max': float(stats['headroom_max']),
                'p1': float(stats.get('headroom_p1', 0)),
                'p5': float(stats.get('headroom_p5', 0)),
                'p10': float(stats.get('headroom_p10', 0)),
                'p50': float(stats.get('headroom_p50', 0)),
                'p90': float(stats.get('headroom_p90', 0)),
            },
        },
    }
    
    # Populate incremental sensitivity data
    incremental_sens = stats.get('incremental_sensitivity', {})
    deltas = stats.get('incremental_sensitivity_deltas', [2, 5, 10, 15])
    for delta in deltas:
        delta_key = f"hours_bind_if_add_{delta}MW"
        delta_frac_key = f"hours_bind_if_add_{delta}MW_frac"
        report['validation_results']['incremental_sensitivity']['hours_bind'][str(delta)] = int(incremental_sens.get(delta_key, 0))
        report['validation_results']['incremental_sensitivity']['pct_bind'][str(delta)] = float(incremental_sens.get(delta_frac_key, 0.0))
    
    # Convert report to JSON-serializable types
    report_json, json_warnings = to_jsonable(report)
    if json_warnings:
        # Add warnings to report if any occurred
        if '_json_warnings' not in report_json:
            report_json['_json_warnings'] = []
        report_json['_json_warnings'].extend(json_warnings)
    
    # Guard: ensure epoch_semantics is a non-empty string before writing JSON
    if not isinstance(epoch_semantics, str) or epoch_semantics.strip() == "":
        raise RuntimeError(
            f"epoch_semantics must be a non-empty string, got: {repr(epoch_semantics)} "
            f"(type: {type(epoch_semantics).__name__}) for epoch {year}"
        )
    
    # Write JSON report
    report_file = output_file.parent / f"gxp_hourly_{year}_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    
    # Regression guard: verify hash matches after writing report
    # Re-open report and verify the stored hash matches the actual CSV file hash
    try:
        with open(report_file, 'r', encoding='utf-8') as f:
            written_report = json.load(f)
        
        # Get stored hash from nested structure
        stored_hash_nested = None
        if 'files' in written_report and 'gxp_hourly_csv' in written_report['files']:
            stored_hash_nested = written_report['files']['gxp_hourly_csv'].get('sha256', '').lower()
        
        # Get stored hash from alias
        stored_hash_alias = written_report.get('file_hash_sha256', '').lower()
        
        # Compute actual CSV hash
        actual_csv_hash = sha256_file(output_file).lower()
        
        # Verify nested hash matches actual
        if stored_hash_nested and stored_hash_nested != actual_csv_hash:
            raise RuntimeError(
                f"Hash mismatch in report JSON for {output_file.name}: "
                f"stored nested hash={stored_hash_nested}, actual CSV hash={actual_csv_hash}"
            )
        
        # Verify alias hash matches actual
        if stored_hash_alias and stored_hash_alias != actual_csv_hash:
            raise RuntimeError(
                f"Hash mismatch in report JSON alias for {output_file.name}: "
                f"stored alias hash={stored_hash_alias}, actual CSV hash={actual_csv_hash}"
            )
        
        # Verify nested and alias agree (if both exist)
        if stored_hash_nested and stored_hash_alias and stored_hash_nested != stored_hash_alias:
            raise RuntimeError(
                f"Hash inconsistency in report JSON for {output_file.name}: "
                f"nested hash={stored_hash_nested}, alias hash={stored_hash_alias}"
            )
        
    except Exception as e:
        raise RuntimeError(f"Regression guard failed for {output_file.name}: {e}")
    
    return report


def get_capacity_schedule(year: int, config: dict) -> pd.Series:
    """
    Generate hourly capacity_MW series based on date ranges in config.
    Capacity values are in MVA, converted to MW via power_factor.
    """
    # Create hourly datetime index for the year
    start = datetime(year, 1, 1)
    # Handle leap years
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        end = datetime(year, 12, 31, 23)
        hours = 8784
    else:
        end = datetime(year, 12, 31, 23)
        hours = 8760
    
    dt_index = pd.date_range(start=start, end=end, freq='h')
    
    power_factor = config['gxp']['power_factor']
    capacity_ranges = config['capacity']['ranges']
    
    # Initialize capacity series
    capacity_mw = pd.Series(index=dt_index, dtype=float)
    
    # Apply capacity ranges
    for range_config in capacity_ranges:
        start_str = range_config['start']
        end_str = range_config['end']
        capacity_mva = range_config['capacity_mva']
        capacity_mw_value = capacity_mva * power_factor
        
        # Parse date range (MM-DD format)
        start_month, start_day = map(int, start_str.split('-'))
        end_month, end_day = map(int, end_str.split('-'))
        
        # Create mask for this date range
        mask = (
            (dt_index.month > start_month) | 
            ((dt_index.month == start_month) & (dt_index.day >= start_day))
        ) & (
            (dt_index.month < end_month) | 
            ((dt_index.month == end_month) & (dt_index.day <= end_day))
        )
        
        capacity_mw[mask] = capacity_mw_value
    
    return capacity_mw


def generate_ar1_noise(n: int, rho: float, sigma: float, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate AR(1) temporally correlated noise.
    eps[t] = rho*eps[t-1] + sqrt(1-rho^2)*N(0, sigma)
    Uses numpy.random.Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize
    eps = np.zeros(n)
    eps[0] = rng.normal(0, sigma)
    
    # AR(1) process
    for t in range(1, n):
        eps[t] = rho * eps[t-1] + np.sqrt(1 - rho**2) * rng.normal(0, sigma)
    
    return eps


def load_emi_template(config: dict, repo_root: Optional[Path] = None) -> Optional[Dict[str, np.ndarray]]:
    """
    Load EMI template from Jan-2023 data and extract weekday/weekend diurnal shapes.
    Returns dict with 'weekday_shape' and 'weekend_shape' (24-element arrays, normalized to mean 1.0),
    or None if template is disabled or file not found.
    
    Args:
        config: Configuration dictionary
        repo_root: Repository root directory for resolving relative paths (default: auto-detect)
    """
    emi_config = config.get('emi_template', {})
    if not emi_config.get('enabled', False):
        return None
    
    # Get repo root if not provided
    if repo_root is None:
        repo_root = get_repo_root()
    
    # Resolve template path relative to repo root
    template_path_str = emi_config.get('jan2023_path', '')
    if not template_path_str:
        return None
    
    template_path = resolve_path(template_path_str, repo_root)
    if not template_path.exists():
        warnings.warn(f"EMI template file not found: {template_path}. Falling back to default diurnal profile.")
        return None
    
    try:
        # Try to read Excel file first, then CSV
        if template_path.suffix.lower() in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(template_path)
            except ImportError:
                warnings.warn("openpyxl not installed. Cannot read Excel file. Install with: pip install openpyxl")
                return None
        else:
            df = pd.read_csv(template_path)
        
        # Handle EMI-specific format: Trading_Date + TP1-TP48 columns
        if 'Trading_Date' in df.columns and any(col.startswith('TP') for col in df.columns):
            # EMI format: each row is a day, columns TP1-TP48 are half-hourly periods
            gxp_id_filter = emi_config.get('gxp_id', '')
            if gxp_id_filter and 'POC' in df.columns:
                df = df[df['POC'] == gxp_id_filter]
                if len(df) == 0:
                    warnings.warn(f"No data found for GXP ID {gxp_id_filter} in EMI template.")
                    return None
            
            # Parse trading date
            df['Trading_Date'] = pd.to_datetime(df['Trading_Date'])
            
            # Melt TP columns to long format
            tp_cols = [col for col in df.columns if col.startswith('TP')]
            df_long = df.melt(
                id_vars=['Trading_Date'],
                value_vars=tp_cols,
                var_name='TP',
                value_name='kWh'
            )
            
            # Convert TP number to hour and minute
            # TP1 = 00:00-00:30, TP2 = 00:30-01:00, ..., TP48 = 23:30-00:00
            df_long['TP_num'] = df_long['TP'].str.extract(r'TP(\d+)').astype(int)
            df_long['hour'] = (df_long['TP_num'] - 1) // 2
            df_long['minute'] = ((df_long['TP_num'] - 1) % 2) * 30
            
            # Create datetime for start of each trading period
            df_long['datetime'] = df_long['Trading_Date'] + pd.to_timedelta(df_long['hour'], unit='h') + pd.to_timedelta(df_long['minute'], unit='m')
            
            # Convert kWh to MW (half-hourly: MW = 2 * kWh / 1000)
            df_long['MW'] = df_long['kWh'] * 2.0 / 1000.0
            
            df = df_long[['datetime', 'MW']].copy()
        else:
            # Standard format: find datetime and energy columns
            datetime_col = None
            energy_col = None
            
            for col in df.columns:
                col_lower = str(col).lower()
                if any(x in col_lower for x in ['date', 'time', 'datetime', 'timestamp']):
                    datetime_col = col
                elif any(x in col_lower for x in ['energy', 'kwh', 'offtake', 'load', 'mw']):
                    energy_col = col
            
            if datetime_col is None or energy_col is None:
                warnings.warn(f"Could not find datetime or energy column in EMI template. Found columns: {df.columns.tolist()}")
                return None
            
            # Parse datetime
            df['datetime'] = pd.to_datetime(df[datetime_col])
            
            # Convert energy to MW
            time_resolution = emi_config.get('time_resolution_minutes', 30)
            units = emi_config.get('units', 'kWh').lower()
            
            if units == 'kwh':
                # MW = (kWh_per_interval * (60 / interval_minutes)) / 1000
                df['MW'] = df[energy_col] * (60.0 / time_resolution) / 1000.0
            elif units == 'mw':
                df['MW'] = df[energy_col]
            else:
                warnings.warn(f"Unknown units: {units}. Assuming kWh.")
                df['MW'] = df[energy_col] * (60.0 / time_resolution) / 1000.0
        
        # Filter to January 2023
        df = df[df['datetime'].dt.year == 2023]
        df = df[df['datetime'].dt.month == 1]
        
        if len(df) == 0:
            warnings.warn("No January 2023 data found in EMI template.")
            return None
        
        # Aggregate half-hourly to hourly (mean of two half-hours)
        df['hour'] = df['datetime'].dt.hour
        df['date'] = df['datetime'].dt.date
        df['is_weekend'] = df['datetime'].dt.weekday >= 5
        
        # Group by date, hour, and aggregate
        hourly = df.groupby(['date', 'hour', 'is_weekend'])['MW'].mean().reset_index()
        
        # Build weekday and weekend shapes
        weekday_data = hourly[~hourly['is_weekend']]
        weekend_data = hourly[hourly['is_weekend']]
        
        # Average across all days for each hour
        weekday_shape = weekday_data.groupby('hour')['MW'].mean().values
        weekend_shape = weekend_data.groupby('hour')['MW'].mean().values
        
        # Normalize to mean 1.0
        if len(weekday_shape) == 24 and weekday_shape.mean() > 0:
            weekday_shape = weekday_shape / weekday_shape.mean()
        else:
            warnings.warn("Could not extract valid weekday shape from EMI template.")
            return None
        
        if len(weekend_shape) == 24 and weekend_shape.mean() > 0:
            weekend_shape = weekend_shape / weekend_shape.mean()
        else:
            # If no weekend data, use weekday shape
            weekend_shape = weekday_shape.copy()
        
        return {
            'weekday_shape': weekday_shape,
            'weekend_shape': weekend_shape
        }
    
    except Exception as e:
        warnings.warn(f"Error loading EMI template: {e}. Falling back to default diurnal profile.")
        return None


def generate_weekly_variation(n: int, amplitude: float, correlation_days: float = 8.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generate weekly weather-like variation using Gaussian-smoothed random series.
    correlation_days: approximate correlation length in days (default 8 days)
    Uses numpy.random.Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Generate white noise
    white_noise = rng.normal(0, 1, n)
    
    # Apply Gaussian smoothing (approximate with moving average)
    # Convert correlation_days to hours
    window_hours = int(correlation_days * 24)
    if window_hours < 1:
        window_hours = 1
    
    # Use a simple moving average as approximation
    # For better results, could use scipy.ndimage.gaussian_filter1d
    smoothed = pd.Series(white_noise).rolling(window=window_hours, center=True, min_periods=1).mean().values
    
    # Normalize to have std = 1, then scale by amplitude
    if smoothed.std() > 0:
        smoothed = smoothed / smoothed.std() * amplitude
    else:
        smoothed = smoothed * amplitude
    
    return smoothed


def soft_saturate(x: np.ndarray, lower: float, upper: float, k: float = 10.0) -> np.ndarray:
    """
    Apply soft saturation (smooth clipping) that only affects values near bounds.
    Values well within [lower, upper] are unchanged.
    """
    # For values below lower: smoothly push toward lower
    # For values above upper: smoothly push toward upper  
    # For values in [lower, upper]: minimal change (preserve values)
    
    result = x.copy()
    
    # Soft lower bound: only affect values below or near lower
    below_mask = x < lower
    if below_mask.any():
        # Use softplus: lower + log(1 + exp(k*(x - lower))) / k
        # This smoothly transitions from x (when x >> lower) to lower (when x << lower)
        diff = x[below_mask] - lower
        result[below_mask] = lower + np.log1p(np.exp(np.clip(k * diff, -50, 50))) / k
    
    # Soft upper bound: only affect values above or near upper
    above_mask = result > upper
    if above_mask.any():
        # Use inverted softplus: upper - log(1 + exp(k*(upper - x))) / k
        diff = upper - result[above_mask]
        result[above_mask] = upper - np.log1p(np.exp(np.clip(k * diff, -50, 50))) / k
    
    return result


def generate_diurnal_profile(hour: int, day_of_week: int, profile_type: str, emi_shapes: Optional[Dict[str, np.ndarray]] = None) -> float:
    """
    Generate diurnal profile multiplier for a given hour and day.
    Returns a multiplier (typically 0.5 to 1.5 range).
    If emi_shapes is provided, uses EMI template shapes instead of profile_type.
    """
    is_weekend = day_of_week >= 5  # Saturday=5, Sunday=6
    
    # Use EMI template shapes if available
    if emi_shapes is not None:
        if is_weekend:
            return float(emi_shapes['weekend_shape'][hour])
        else:
            return float(emi_shapes['weekday_shape'][hour])
    
    if profile_type == "flat":
        return 1.0
    
    elif profile_type == "two_peak":
        # Two peaks: morning (8-9) and evening (18-20)
        # Adjusted multipliers to better match RETA report (peaks ~22-25 MVA from ~18 MW base)
        if is_weekend:
            # Weekend: flatter profile, lower peaks
            if 8 <= hour < 10:
                return 1.15
            elif 18 <= hour < 21:
                return 1.25
            else:
                return 0.85 + 0.15 * np.sin(np.pi * hour / 12)
        else:
            # Weekday: improved two-peak profile with lunchtime shoulder
            # Reduced peak-to-trough ratio for more realistic mixed-load GXP
            if 8 <= hour < 10:
                return 1.30  # morning peak
            elif 18 <= hour < 21:
                return 1.35  # evening peak (slightly higher)
            elif 12 <= hour < 14:
                return 1.05  # lunchtime shoulder (new)
            elif 0 <= hour < 6:
                return 0.75  # overnight low (increased from 0.70 for less extreme drop)
            else:
                # Smooth transition between periods
                if 6 <= hour < 8:
                    return 0.75 + 0.25 * (hour - 6) / 2  # ramp up to morning peak
                elif 10 <= hour < 12:
                    return 1.30 - 0.15 * (hour - 10) / 2  # ramp down to lunch
                elif 14 <= hour < 18:
                    return 1.05 + 0.20 * (hour - 14) / 4  # ramp up to evening peak
                else:  # 21-23
                    return 1.35 - 0.30 * (hour - 21) / 3  # ramp down to overnight
    
    elif profile_type == "business":
        # Business hours emphasis
        if is_weekend:
            return 0.7
        else:
            if 9 <= hour < 17:
                return 1.3
            elif 7 <= hour < 9 or 17 <= hour < 19:
                return 1.1
            else:
                return 0.7
    
    else:
        warnings.warn(f"Unknown diurnal profile type: {profile_type}, using flat")
        return 1.0


def generate_synthetic_baseline(
    year: int, 
    config: dict, 
    noise_sigma_override: Optional[float] = None,
    monthly_means_override: Optional[List[float]] = None,
    emi_shapes: Optional[Dict[str, np.ndarray]] = None,
    rng: Optional[np.random.Generator] = None
) -> pd.Series:
    """
    Generate synthetic baseline_import_MW using monthly shape, diurnal profile, 
    AR(1) correlated noise, weekly variation, and weekend adjustment.
    If emi_shapes is provided, uses EMI template shapes and multiplicative perturbations.
    Supports calibration via overrides.
    """
    # Create hourly datetime index
    start = datetime(year, 1, 1)
    if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
        end = datetime(year, 12, 31, 23)
    else:
        end = datetime(year, 12, 31, 23)
    
    dt_index = pd.date_range(start=start, end=end, freq='h')
    n_hours = len(dt_index)
    
    # Get config parameters (with overrides for calibration)
    monthly_means = monthly_means_override if monthly_means_override is not None else config['synthetic']['monthly_mean_import_mw']
    diurnal_type = config['synthetic']['diurnal']
    noise_sigma = noise_sigma_override if noise_sigma_override is not None else config['synthetic']['noise_sigma_mw']
    
    # Get realism parameters
    rho_ar1 = config['synthetic'].get('rho_ar1', 0.9)
    weekly_var_amp = config['synthetic'].get('weekly_var_amp_mw', 0.7)
    weekend_mult = config['synthetic'].get('weekend_multiplier', 0.97)
    
    # Use provided RNG or create default (for backward compatibility)
    if rng is None:
        rng = np.random.default_rng()
    
    # Initialize baseline
    baseline = pd.Series(index=dt_index, dtype=float)
    
    # Step 1: Apply monthly mean shape and diurnal profile
    for i, dt in enumerate(dt_index):
        month_idx = dt.month - 1  # 0-indexed
        monthly_base = monthly_means[month_idx]
        
        # Apply diurnal profile (with EMI shapes if available)
        hour = dt.hour
        day_of_week = dt.weekday()  # Monday=0, Sunday=6
        diurnal_mult = generate_diurnal_profile(hour, day_of_week, diurnal_type, emi_shapes)
        
        baseline.iloc[i] = monthly_base * diurnal_mult
    
    # Step 2: Apply weekend multiplier
    weekend_mask = pd.Series([dt.weekday() >= 5 for dt in dt_index], index=dt_index)
    baseline[weekend_mask] = baseline[weekend_mask] * weekend_mult
    
    # Step 3: Renormalize to preserve monthly means BEFORE adding noise (important for targets)
    # The diurnal profile and weekend multiplier change the effective monthly means,
    # so we need to restore them to the target values
    df_temp = pd.DataFrame({'baseline': baseline.values, 'month': [dt.month for dt in dt_index]}, index=dt_index)
    current_monthly_means = df_temp.groupby('month')['baseline'].mean()
    
    # Apply correction factor per month to restore original means
    for month in range(1, 13):
        month_mask = df_temp['month'] == month
        if current_monthly_means[month] > 1e-6:  # Avoid division by zero
            correction = monthly_means[month - 1] / current_monthly_means[month]
            baseline[month_mask] = baseline[month_mask] * correction
    
    # Step 4: Add perturbations (multiplicative if EMI template used, additive otherwise)
    if emi_shapes is not None:
        # Multiplicative perturbations (±2-4%) to preserve EMI diurnal shape
        # Convert noise_sigma to a relative multiplier (e.g., 0.03 = 3% variation)
        noise_relative = noise_sigma / 20.0  # Scale to ~3% for noise_sigma=0.4
        ar1_noise = generate_ar1_noise(n_hours, rho_ar1, noise_relative, rng=rng)
        ar1_mult = 1.0 + ar1_noise  # Multiplicative perturbation
        
        weekly_var_relative = weekly_var_amp / 20.0  # Scale to ~2% for weekly_var_amp=0.4
        weekly_var = generate_weekly_variation(n_hours, weekly_var_relative, correlation_days=8.0, rng=rng)
        weekly_mult = 1.0 + weekly_var  # Multiplicative perturbation
        
        baseline = baseline * ar1_mult * weekly_mult
    else:
        # Additive perturbations (original behavior)
        ar1_noise = generate_ar1_noise(n_hours, rho_ar1, noise_sigma, rng=rng)
        baseline = baseline + ar1_noise
        
        weekly_var = generate_weekly_variation(n_hours, weekly_var_amp, correlation_days=8.0, rng=rng)
        baseline = baseline + weekly_var
    
    # Clip to non-negative (will be further processed later)
    baseline = baseline.clip(lower=0)
    
    return baseline


def calibrate_baseline_to_reduce_binding(
    year: int,
    config: dict,
    capacity_mw: pd.Series,
    baseline_import_mw: pd.Series,
    reserve_margin_mw: float,
    max_binding_hours_frac: float,
    baseline_floor_mw: float
) -> pd.Series:
    """
    Calibrate baseline import to reduce binding hours below target fraction.
    Returns calibrated baseline.
    """
    # Calculate current binding fraction (before applying floor)
    headroom = (capacity_mw - baseline_import_mw).clip(lower=0)
    binding_mask = headroom < 1e-6
    binding_frac = binding_mask.sum() / len(headroom)
    
    if binding_frac <= max_binding_hours_frac:
        return baseline_import_mw
    
    # Need to reduce binding - try reducing noise first, then scale down peak months
    original_noise_sigma = config['synthetic']['noise_sigma_mw']
    original_monthly_means = list(config['synthetic']['monthly_mean_import_mw'])  # Make a copy
    
    # Strategy: reduce noise sigma and scale down top 2 months (Oct, Nov)
    noise_sigma = original_noise_sigma
    monthly_means = original_monthly_means.copy()
    
    max_iterations = 20
    best_baseline = baseline_import_mw
    best_binding_frac = binding_frac
    
    for iteration in range(max_iterations):
        # Regenerate baseline with current parameters (using same seed for determinism)
        calibrated_baseline = generate_synthetic_baseline(
            year, config, 
            noise_sigma_override=noise_sigma,
            monthly_means_override=monthly_means
        )
        
        # Apply epoch scaling
        epoch_scaling = config['epoch_scaling'].get(str(year), 1.0)
        calibrated_baseline = calibrated_baseline * epoch_scaling
        
        # Apply soft cap with reserve margin
        calibrated_baseline = calibrated_baseline.clip(upper=capacity_mw - reserve_margin_mw)
        
        # Apply baseline floor
        calibrated_baseline = calibrated_baseline.clip(lower=baseline_floor_mw, upper=capacity_mw)
        
        # Check binding fraction
        headroom = (capacity_mw - calibrated_baseline).clip(lower=0)
        binding_mask = headroom < 1e-6
        binding_frac = binding_mask.sum() / len(headroom)
        
        if binding_frac < best_binding_frac:
            best_baseline = calibrated_baseline
            best_binding_frac = binding_frac
        
        if binding_frac <= max_binding_hours_frac:
            return calibrated_baseline
        
        # Adjust parameters
        if iteration < 10:
            # Reduce noise
            noise_sigma = max(0.3, noise_sigma * 0.9)
        else:
            # Scale down peak months (Oct=9, Nov=10)
            monthly_means[9] = max(monthly_means[9] * 0.95, original_monthly_means[9] * 0.8)
            monthly_means[10] = max(monthly_means[10] * 0.95, original_monthly_means[10] * 0.8)
    
    # If still not meeting target, return best effort
    warnings.warn(
        f"Calibration did not achieve target binding fraction {max_binding_hours_frac:.2%}. "
        f"Best achieved: {best_binding_frac:.2%}"
    )
    return best_baseline


def calculate_tou_tariff(dt_index: pd.DatetimeIndex, config: dict, year: int) -> pd.Series:
    """
    Calculate time-of-use tariff for each hour.
    Returns Series with hourly tariff values.
    """
    # Get base tariff for this epoch
    base_tariff = config['tariff_base'].get(str(year), 120.0)
    
    # Get TOU config
    tou_config = config.get('tou', {})
    peak_mult = tou_config.get('peak_multiplier', 1.35)
    shoulder_mult = tou_config.get('shoulder_multiplier', 1.10)
    offpeak_mult = tou_config.get('offpeak_multiplier', 0.85)
    weekend_all_offpeak = tou_config.get('weekend_all_offpeak', True)
    
    # Get hour definitions
    peak_hours = set(tou_config.get('weekday_peak_hours', [7,8,9,17,18,19]))
    shoulder_hours = set(tou_config.get('weekday_shoulder_hours', [6,10,11,12,13,14,15,16,20,21]))
    offpeak_hours = set(tou_config.get('weekday_offpeak_hours', [0,1,2,3,4,5,22,23]))
    
    # Initialize tariff series
    tariffs = pd.Series(index=dt_index, dtype=float)
    
    for i, dt in enumerate(dt_index):
        hour = dt.hour
        day_of_week = dt.weekday()  # Monday=0, Sunday=6
        is_weekend = day_of_week >= 5
        
        if weekend_all_offpeak and is_weekend:
            multiplier = offpeak_mult
        elif hour in peak_hours:
            multiplier = peak_mult
        elif hour in shoulder_hours:
            multiplier = shoulder_mult
        elif hour in offpeak_hours:
            multiplier = offpeak_mult
        else:
            # Default to offpeak if hour not in any category
            multiplier = offpeak_mult
        
        tariffs.iloc[i] = base_tariff * multiplier
    
    return tariffs


def load_measured_data(measured_dir: Path, year: int) -> Optional[pd.Series]:
    """
    Load measured half-hourly import data for a given year.
    Expected format: CSV with timestamp,import_MW columns at 30-min resolution.
    Returns hourly aggregated series, or None if file not found.
    """
    if not measured_dir.exists():
        return None
    
    # Try common filename patterns
    possible_names = [
        f"measured_{year}.csv",
        f"import_{year}.csv",
        f"{year}_import.csv",
        f"measured_import_{year}.csv",
    ]
    
    for filename in possible_names:
        filepath = measured_dir / filename
        if filepath.exists():
            try:
                df = pd.read_csv(filepath)
                # Try to identify timestamp and import columns
                if 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'])
                elif 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                else:
                    # Try first column as datetime
                    df['datetime'] = pd.to_datetime(df.iloc[:, 0])
                
                # Find import column
                if 'import_MW' in df.columns:
                    import_col = 'import_MW'
                elif 'import' in df.columns:
                    import_col = 'import'
                else:
                    # Try second column or column with numeric data
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        import_col = numeric_cols[0]
                    else:
                        raise ValueError("Could not identify import column")
                
                # Set datetime as index and resample to hourly (mean)
                df = df.set_index('datetime')
                hourly = df[import_col].resample('h').mean()
                
                # Filter to target year
                hourly = hourly[hourly.index.year == year]
                
                if len(hourly) > 0:
                    return hourly
                else:
                    warnings.warn(f"Measured data for {year} is empty after filtering")
                    return None
                    
            except Exception as e:
                warnings.warn(f"Error loading {filepath}: {e}")
                continue
    
    return None


def generate_gxp_data(
    year: int,
    config: dict,
    mode: str,
    measured_dir: Optional[Path] = None,
    seed_override: Optional[int] = None,
    repo_root: Optional[Path] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Generate complete GXP hourly data for a given year.
    Returns (DataFrame, stats_dict) where DataFrame has columns: 
    gxp_id, datetime, capacity_MW, gxp_headroom_MW, tariff_base_nzd_per_MWh
    
    Args:
        year: Year to generate data for
        config: Configuration dictionary
        mode: Generation mode ('synthetic' or 'measured')
        measured_dir: Directory containing measured data (for measured mode)
        seed_override: Override base random seed from config
        repo_root: Repository root directory for resolving relative paths (default: auto-detect)
    """
    # Get repo root if not provided
    if repo_root is None:
        repo_root = get_repo_root()
    
    # Get capacity schedule
    capacity_mw = get_capacity_schedule(year, config)
    
    # Load EMI template if enabled
    emi_shapes = load_emi_template(config, repo_root=repo_root)
    emi_template_used = emi_shapes is not None
    
    # Get constraint parameters (with correct defaults)
    baseline_floor_mw = config['synthetic'].get('baseline_floor_mw', 3.5)  # MUST be 3.5, not 2.0
    # Check for per-epoch reserve margin first, then fall back to default
    reserve_margin_by_epoch = config.get('reserve_margin_by_epoch', {})
    reserve_margin_mw_requested = config['synthetic'].get('reserve_margin_mw', 0.0)  # Default from config
    if str(year) in reserve_margin_by_epoch:
        reserve_margin_mw = float(reserve_margin_by_epoch[str(year)])
        reserve_margin_mw_requested = reserve_margin_mw  # Per-epoch override
    else:
        reserve_margin_mw = reserve_margin_mw_requested
    max_binding_hours_frac = config['synthetic'].get('max_binding_hours_frac', 0.005)
    
    # Set up reproducible RNG: epoch seed = base seed + year
    # Allow CLI override of base seed (passed via seed_override parameter)
    seed_base = seed_override if seed_override is not None else config['synthetic'].get('seed', 42)
    seed_epoch = seed_base + year
    rng = np.random.default_rng(seed_epoch)
    
    # Print config values used (guard to verify config is loaded)
    monthly_means = config['synthetic'].get('monthly_mean_import_mw', [])
    noise_sigma = config['synthetic'].get('noise_sigma_mw', 0.8)
    epoch_scaling = config['epoch_scaling'].get(str(year), 1.0)
    print(f"\nConfig values used for {year}:")
    print(f"  baseline_floor_MW: {baseline_floor_mw}")
    if str(year) in reserve_margin_by_epoch:
        print(f"  reserve_margin_MW: {reserve_margin_mw} (per-epoch override from reserve_margin_by_epoch)")
    else:
        print(f"  reserve_margin_MW: {reserve_margin_mw} (default from synthetic.reserve_margin_mw)")
    if emi_template_used:
        print(f"  EMI template: enabled")
    print(f"  monthly_mean_import_mw: {monthly_means}")
    print(f"  noise_sigma_mw: {noise_sigma}")
    print(f"  epoch_scaling: {epoch_scaling}")
    print(f"  random_seed_base: {seed_base}, random_seed_epoch: {seed_epoch}")
    
    # Generate or load baseline import
    if mode == "synthetic":
        baseline_import_mw = generate_synthetic_baseline(year, config, emi_shapes=emi_shapes, rng=rng)
    elif mode == "measured":
        if measured_dir is None:
            raise ValueError("measured_dir required for measured mode")
        baseline_import_mw = load_measured_data(measured_dir, year)
        if baseline_import_mw is None:
            warnings.warn(
                f"No measured data found for {year}, falling back to synthetic mode"
            )
            baseline_import_mw = generate_synthetic_baseline(year, config, emi_shapes=emi_shapes, rng=rng)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Ensure baseline aligns with capacity index
    if not baseline_import_mw.index.equals(capacity_mw.index):
        # Reindex baseline to match capacity (forward fill if needed)
        baseline_import_mw = baseline_import_mw.reindex(
            capacity_mw.index, method='ffill'
        ).fillna(baseline_floor_mw)
    
    # Apply epoch scaling (for 2020 this should be 1.0, so no change)
    epoch_scaling = config['epoch_scaling'].get(str(year), 1.0)
    if epoch_scaling != 1.0:
        baseline_import_mw = baseline_import_mw * epoch_scaling
    
    # Apply soft saturation or hard clipping based on config
    smooth_clip = config['synthetic'].get('smooth_clip', True)
    smooth_k = config['synthetic'].get('smooth_k', 10.0)
    
    if smooth_clip:
        # Use soft saturation for smooth transitions
        # Apply element-wise since capacity may vary by hour
        upper_bound = capacity_mw - reserve_margin_mw
        baseline_values = baseline_import_mw.values
        floor_values = np.full(len(baseline_values), baseline_floor_mw)
        upper_values = upper_bound.values
        
        # Apply soft saturation element-wise
        saturated = np.zeros_like(baseline_values)
        for i in range(len(baseline_values)):
            saturated[i] = soft_saturate(
                np.array([baseline_values[i]]),
                floor_values[i],
                upper_values[i],
                k=smooth_k
            )[0]
        
        baseline_import_mw = pd.Series(saturated, index=baseline_import_mw.index)
    else:
        # Use hard clipping (fallback)
        baseline_import_mw = baseline_import_mw.clip(lower=baseline_floor_mw, upper=capacity_mw)
        
        # Apply soft cap with reserve margin (but don't go below floor)
        baseline_import_mw = baseline_import_mw.clip(
            lower=baseline_floor_mw, 
            upper=capacity_mw - reserve_margin_mw
        )
    
    # Calibrate to reduce binding hours if in synthetic mode (only if needed)
    if mode == "synthetic":
        # Check if calibration is needed
        headroom_check = (capacity_mw - baseline_import_mw).clip(lower=0)
        binding_mask_check = headroom_check < 1e-6
        binding_frac_check = binding_mask_check.sum() / len(headroom_check)
        
        if binding_frac_check > max_binding_hours_frac:
            baseline_import_mw = calibrate_baseline_to_reduce_binding(
                year, config, capacity_mw, baseline_import_mw,
                reserve_margin_mw, max_binding_hours_frac, baseline_floor_mw
            )
            # Re-apply floor after calibration
            baseline_import_mw = baseline_import_mw.clip(lower=baseline_floor_mw, upper=capacity_mw)
    
    # Apply peak calibration if enabled
    calibration_config = config.get('calibration', {})
    if calibration_config.get('enabled', False):
        target_peaks = calibration_config.get('target_peak_MW_by_epoch', {})
        peak_match_quantile = calibration_config.get('peak_match_quantile', 0.999)
        
        if str(year) in target_peaks:
            target_peak = float(target_peaks[str(year)])
            current_peak_quantile = baseline_import_mw.quantile(peak_match_quantile)
            
            if current_peak_quantile > 0:
                scale_factor = target_peak / current_peak_quantile
                baseline_import_mw = baseline_import_mw * scale_factor
                # Re-apply floor and upper bound
                baseline_import_mw = baseline_import_mw.clip(lower=baseline_floor_mw, upper=capacity_mw)
    
    # Calculate headroom: max(0, capacity_MW - baseline_import_MW - reserve_margin_MW)
    gxp_headroom_mw = (capacity_mw - baseline_import_mw - reserve_margin_mw).clip(lower=0)
    
    # Calculate TOU tariffs
    tou_tariffs = calculate_tou_tariff(capacity_mw.index, config, year)
    
    # Build output DataFrame with explicit arrays for all columns
    n_rows = len(capacity_mw)
    gxp_id = config['gxp']['gxp_id']
    
    # Convert datetime index to UTC ISO-8601 format with Z suffix for unambiguous timestamps
    # Use canonical formatter to ensure consistent UTC format
    timestamp_utc_series = format_timestamp_utc(capacity_mw.index)
    
    # Build output DataFrame with frozen schema (snake_case, exact column order)
    output = pd.DataFrame({
        'gxp_id': [gxp_id] * n_rows,
        'timestamp_utc': timestamp_utc_series.tolist(),
        'capacity_mw': capacity_mw.values,
        'baseline_import_mw': baseline_import_mw.values,
        'reserve_margin_mw': [reserve_margin_mw] * n_rows,  # Scalar repeated per row
        'headroom_mw': gxp_headroom_mw.values,
        'tariff_nzd_per_mwh': tou_tariffs.values,
        'epoch': [year] * n_rows,
    })
    
    # Regression guard: ensure no legacy column names
    assert "capacity_MW" not in output.columns, "Legacy column name 'capacity_MW' found"
    assert "datetime" not in output.columns, "Legacy column name 'datetime' found"
    assert output.columns.tolist() == REQUIRED_GXP_COLUMNS, f"Column order mismatch. Expected: {REQUIRED_GXP_COLUMNS}, got: {output.columns.tolist()}"
    
    # Ensure numeric columns are float type
    output['capacity_mw'] = output['capacity_mw'].astype(float)
    output['baseline_import_mw'] = output['baseline_import_mw'].astype(float)
    output['reserve_margin_mw'] = output['reserve_margin_mw'].astype(float)
    output['headroom_mw'] = output['headroom_mw'].astype(float)
    output['tariff_nzd_per_mwh'] = output['tariff_nzd_per_mwh'].astype(float)
    output['epoch'] = output['epoch'].astype(int)
    
    # Calculate statistics for reporting
    binding_mask = gxp_headroom_mw < 1e-6
    binding_frac = binding_mask.sum() / len(gxp_headroom_mw)
    floor_hits = (baseline_import_mw <= baseline_floor_mw + 1e-6).sum()
    
    # Peak calibration reporting
    peak_calibration_applied = False
    achieved_peak_quantile = None
    target_peak = None
    peak_match_quantile = None
    if calibration_config.get('enabled', False):
        target_peaks = calibration_config.get('target_peak_MW_by_epoch', {})
        peak_match_quantile = calibration_config.get('peak_match_quantile', 0.999)
        if str(year) in target_peaks:
            peak_calibration_applied = True
            target_peak = float(target_peaks[str(year)])
            achieved_peak_quantile = baseline_import_mw.quantile(peak_match_quantile)
    
    # Headroom distribution statistics - threshold counts
    headroom_lt1_mask = gxp_headroom_mw < 1.0
    headroom_lt1_count = headroom_lt1_mask.sum()
    headroom_lt1_frac = headroom_lt1_count / len(gxp_headroom_mw)
    
    headroom_lt2_mask = gxp_headroom_mw < 2.0
    headroom_lt2_count = headroom_lt2_mask.sum()
    headroom_lt2_frac = headroom_lt2_count / len(gxp_headroom_mw)
    
    headroom_lt5_mask = gxp_headroom_mw < 5.0
    headroom_lt5_count = headroom_lt5_mask.sum()
    headroom_lt5_frac = headroom_lt5_count / len(gxp_headroom_mw)
    
    headroom_lt10_mask = gxp_headroom_mw < 10.0
    headroom_lt10_count = headroom_lt10_mask.sum()
    headroom_lt10_frac = headroom_lt10_count / len(gxp_headroom_mw)
    
    # Headroom percentiles
    headroom_p1 = gxp_headroom_mw.quantile(0.01)
    headroom_p5 = gxp_headroom_mw.quantile(0.05)
    headroom_p10 = gxp_headroom_mw.quantile(0.10)
    headroom_p50 = gxp_headroom_mw.quantile(0.50)
    headroom_p90 = gxp_headroom_mw.quantile(0.90)
    
    # Incremental demand sensitivity diagnostics
    # Count hours where baseline headroom < Δ for each delta in config
    incremental_config = config.get('incremental_sensitivity', {})
    deltas_mw = incremental_config.get('deltas_mw', [2, 5, 10, 15])  # Default if not in config
    
    incremental_sensitivity = {}
    for delta in deltas_mw:
        delta_key = f"hours_bind_if_add_{delta}MW"
        delta_frac_key = f"hours_bind_if_add_{delta}MW_frac"
        hours_bind = (gxp_headroom_mw < float(delta)).sum()
        hours_bind_frac = hours_bind / len(gxp_headroom_mw)
        incremental_sensitivity[delta_key] = hours_bind
        incremental_sensitivity[delta_frac_key] = hours_bind_frac
    
    # TOU tariff statistics
    tou_config = config.get('tou', {})
    peak_hours = set(tou_config.get('weekday_peak_hours', []))
    shoulder_hours = set(tou_config.get('weekday_shoulder_hours', []))
    offpeak_hours = set(tou_config.get('weekday_offpeak_hours', []))
    weekend_all_offpeak = tou_config.get('weekend_all_offpeak', True)
    
    peak_count = 0
    shoulder_count = 0
    offpeak_count = 0
    
    for i, dt in enumerate(capacity_mw.index):
        hour = dt.hour
        day_of_week = dt.weekday()
        is_weekend = day_of_week >= 5
        
        if weekend_all_offpeak and is_weekend:
            offpeak_count += 1
        elif hour in peak_hours:
            peak_count += 1
        elif hour in shoulder_hours:
            shoulder_count += 1
        else:
            offpeak_count += 1
    
    # Calculate January-specific metrics for EMI template validation
    jan_data = baseline_import_mw[baseline_import_mw.index.month == 1]
    jan_weekend_weekday_ratio = None
    jan_diurnal_range = None
    
    if len(jan_data) > 0:
        jan_weekend = jan_data[jan_data.index.weekday >= 5]
        jan_weekday = jan_data[jan_data.index.weekday < 5]
        
        if len(jan_weekend) > 0 and len(jan_weekday) > 0:
            jan_weekend_weekday_ratio = float(jan_weekend.mean() / jan_weekday.mean())
        
        # Calculate average hour-of-day range (max - min across all hours)
        hourly_means = jan_data.groupby(jan_data.index.hour).mean()
        if len(hourly_means) > 0:
            jan_diurnal_range = float(hourly_means.max() - hourly_means.min())
    
    # Compare headroom tightness against epoch-specific acceptance limits (upper bounds)
    # Pass logic: actual <= target + tolerance (treats targets as upper-bound acceptance limits)
    headroom_acceptance_limits = {}
    tightness_targets = config.get('tightness_targets', {})
    tightness_tolerance = config.get('tightness_tolerance', {}).get('abs_tol', 0.005)
    
    if str(year) in tightness_targets:
        targets = tightness_targets[str(year)]
        target_lt1 = targets.get('pct_lt_1MW', None)
        target_lt2 = targets.get('pct_lt_2MW', None)
        target_lt5 = targets.get('pct_lt_5MW', None)
        
        headroom_acceptance_limits = {
            'pct_lt_1MW': {
                'target': float(target_lt1) if target_lt1 is not None else None,
                'actual': float(headroom_lt1_frac),
                'pass': None if target_lt1 is None else bool(headroom_lt1_frac <= target_lt1 + tightness_tolerance)
            },
            'pct_lt_2MW': {
                'target': float(target_lt2) if target_lt2 is not None else None,
                'actual': float(headroom_lt2_frac),
                'pass': None if target_lt2 is None else bool(headroom_lt2_frac <= target_lt2 + tightness_tolerance)
            },
            'pct_lt_5MW': {
                'target': float(target_lt5) if target_lt5 is not None else None,
                'actual': float(headroom_lt5_frac),
                'pass': None if target_lt5 is None else bool(headroom_lt5_frac <= target_lt5 + tightness_tolerance)
            }
        }
    
    stats = {
        'baseline_import_mw': baseline_import_mw,
        'baseline_min': baseline_import_mw.min(),
        'baseline_mean': baseline_import_mw.mean(),
        'baseline_max': baseline_import_mw.max(),
        'headroom_min': gxp_headroom_mw.min(),
        'headroom_mean': gxp_headroom_mw.mean(),
        'headroom_max': gxp_headroom_mw.max(),
        'headroom_p1': headroom_p1,
        'headroom_p5': headroom_p5,
        'headroom_p10': headroom_p10,
        'headroom_p50': headroom_p50,
        'headroom_p90': headroom_p90,
        'binding_frac': binding_frac,
        'binding_count': binding_mask.sum(),
        'headroom_lt1_count': headroom_lt1_count,
        'headroom_lt1_frac': headroom_lt1_frac,
        'headroom_lt2_count': headroom_lt2_count,
        'headroom_lt2_frac': headroom_lt2_frac,
        'headroom_lt5_count': headroom_lt5_count,
        'headroom_lt5_frac': headroom_lt5_frac,
        'headroom_lt10_count': headroom_lt10_count,
        'headroom_lt10_frac': headroom_lt10_frac,
        'incremental_sensitivity': incremental_sensitivity,
        'incremental_sensitivity_deltas': deltas_mw,
        'floor_hits': floor_hits,
        'tariff_min': tou_tariffs.min(),
        'tariff_mean': tou_tariffs.mean(),
        'tariff_max': tou_tariffs.max(),
        'tariff_peak_count': peak_count,
        'tariff_shoulder_count': shoulder_count,
        'tariff_offpeak_count': offpeak_count,
        'emi_template_used': emi_template_used,
        'jan_weekend_weekday_ratio': jan_weekend_weekday_ratio,
        'jan_diurnal_range': jan_diurnal_range,
        'peak_calibration_applied': peak_calibration_applied,
        'peak_calibration_target_MW': target_peak,
        'peak_calibration_achieved_quantile_MW': achieved_peak_quantile,
        'peak_calibration_quantile': calibration_config.get('peak_match_quantile', 0.999) if peak_calibration_applied else None,
        'headroom_acceptance_limits': headroom_acceptance_limits,
        'reserve_margin_MW_requested': reserve_margin_mw_requested,
        'reserve_margin_MW_applied': reserve_margin_mw,
        'random_seed_base': seed_base,
        'random_seed_epoch': seed_epoch,
        'rng_bit_generator': str(type(rng.bit_generator).__name__),
    }
    
    return output, stats


def lookup_map(m: dict, key, default=None):
    """
    Lookup helper that supports both int and str keys.
    Tries key as-is, then as str(key), returns default if not found.
    """
    if key in m:
        return m[key]
    if str(key) in m:
        return m[str(key)]
    return default


def generate_upgrade_menu(config: dict, output_dir: Path, script_path: Optional[Path] = None) -> Optional[Path]:
    """
    Generate GXP upgrade menu CSV file.
    Returns Path to the generated file, or None if disabled.
    """
    menu_config = config.get('gxp_upgrade_menu', {})
    if not menu_config.get('enabled', False):
        return None
    
    gxp_id = config['gxp']['gxp_id']
    deltas_mw = menu_config.get('deltas_mw', [0, 5, 15])
    
    # Get lead times and capex from config (with defaults)
    # Support both int and str keys via lookup_map helper
    lead_time_map = menu_config.get('lead_time_years_by_delta', {})
    capex_map = menu_config.get('capex_nzd_by_delta', {})
    fixed_om = menu_config.get('fixed_om_nzd_per_year', 0)
    lifetime = menu_config.get('lifetime_years', 30)
    
    # Money semantics
    capex_currency = menu_config.get('capex_currency', 'NZD')
    capex_price_basis_year = menu_config.get('capex_price_basis_year', 2025)
    capex_real_or_nominal = menu_config.get('capex_real_or_nominal', 'real')
    includes_gst = menu_config.get('includes_gst', False)
    
    # Default lead times: U0=0, U5=0, U15=1
    default_lead_times = {0: 0, 5: 0, 15: 1}
    # Default CAPEX: placeholder values
    default_capex = {0: 0, 5: 500000, 15: 2000000}
    # Default notes
    notes_map = {0: "no upgrade", 5: "minor connection works", 15: "major augmentation"}
    
    rows = []
    for delta in deltas_mw:
        option_id = f"U{delta}"
        lead_time = lookup_map(lead_time_map, delta, default_lead_times.get(delta, 0))
        capex = lookup_map(capex_map, delta, default_capex.get(delta, 0))
        notes = notes_map.get(delta, f"upgrade {delta}MW")
        
        rows.append({
            'gxp_id': gxp_id,
            'option_id': option_id,
            'delta_capacity_mw': delta,
            'lead_time_years': lead_time,
            'capex_nzd': capex,
            'fixed_om_nzd_per_year': fixed_om,
            'lifetime_years': lifetime,
            'notes': notes,
        })
    
    df = pd.DataFrame(rows)
    
    # Write CSV
    output_file = output_dir / 'gxp_upgrade_menu.csv'
    df.to_csv(output_file, index=False, lineterminator='\n', float_format='%.2f')
    
    # Verify file was written
    if not output_file.exists():
        raise RuntimeError(f"Upgrade menu CSV not written: {output_file}")
    file_size = output_file.stat().st_size
    if file_size == 0:
        raise RuntimeError(f"Upgrade menu CSV is empty: {output_file}")
    
    # Write provenance report
    git_hash, git_detected = get_git_commit_hash(output_dir=output_dir, script_path=script_path)
    file_hash = sha256_file(output_file).lower()
    
    report = {
        'schema_version': 'gxp_upgrade_menu_report_v1',
        'generation_timestamp': datetime.now().isoformat(),
        'output_file': str(output_file.resolve()),
        'file_size_bytes': file_size,
        'file_hash_sha256': file_hash,
        'git_commit_hash': git_hash,
        'git_detected': git_detected,
        'config_used': {
            'gxp_id': gxp_id,
            'deltas_mw': deltas_mw,
            'lead_time_years_by_delta': lead_time_map,
            'capex_nzd_by_delta': capex_map,
            'fixed_om_nzd_per_year': fixed_om,
            'lifetime_years': lifetime,
        },
        'money_semantics': {
            'capex_currency': capex_currency,
            'capex_price_basis_year': capex_price_basis_year,
            'capex_real_or_nominal': capex_real_or_nominal,
            'includes_gst': includes_gst,
        },
        'n_options': len(rows),
    }
    
    # Convert to JSON-safe types
    report_json, _ = to_jsonable(report)
    
    report_file = output_dir / 'gxp_upgrade_menu_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    
    # Print one-line confirmation
    print(f"Upgrade menu: {output_file} ({len(df)} rows, SHA256: {file_hash[:16]}...)")
    
    return output_file


def generate_emissions_intensity(year: int, config: dict, output_dir: Path, script_path: Optional[Path] = None) -> Path:
    """
    Generate grid emissions intensity CSV file for an epoch.
    Returns Path to the generated file.
    """
    emissions_config = config.get('emissions_intensity', {})
    
    # Get values from config (constant per epoch for PoC)
    avg_co2e = emissions_config.get('grid_co2e_kg_per_MWh_avg', 0.1)
    marginal_co2e = emissions_config.get('grid_co2e_kg_per_MWh_marginal', 0.8)
    
    # Create hourly time index for the year (must align with GXP timestamps)
    start_date = pd.Timestamp(f'{year}-01-01', tz='UTC')
    end_date = pd.Timestamp(f'{year}-12-31 23:00:00', tz='UTC')
    dt_index = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Format timestamps using canonical formatter (ensures UTC ISO-8601 with Z suffix)
    timestamp_utc_series = format_timestamp_utc(dt_index)
    
    # Create DataFrame with frozen schema (snake_case)
    df = pd.DataFrame({
        'timestamp_utc': timestamp_utc_series.tolist(),
        'grid_co2e_kg_per_mwh_avg': [avg_co2e] * len(dt_index),
        'grid_co2e_kg_per_mwh_marginal': [marginal_co2e] * len(dt_index),
    })
    
    # Write CSV
    output_file = output_dir / f'grid_emissions_intensity_{year}.csv'
    df.to_csv(output_file, index=False, lineterminator='\n', float_format='%.6f')
    
    # Verify file was written
    if not output_file.exists():
        raise RuntimeError(f"Emissions intensity CSV not written: {output_file}")
    file_size = output_file.stat().st_size
    if file_size == 0:
        raise RuntimeError(f"Emissions intensity CSV is empty: {output_file}")
    
    # Write provenance report
    git_hash, git_detected = get_git_commit_hash(output_dir=output_dir, script_path=script_path)
    file_hash = sha256_file(output_file).lower()
    
    report = {
        'schema_version': 'grid_emissions_intensity_report_v1',
        'generation_timestamp': datetime.now().isoformat(),
        'epoch': year,
        'files': {
            'emissions_csv': {
                'path': str(output_file.resolve()),
                'sha256': file_hash,
            },
        },
        # Backwards compatibility aliases
        'output_file': str(output_file.resolve()),
        'file_size_bytes': file_size,
        'file_hash_sha256': file_hash,
        'git_commit_hash': git_hash,
        'git_detected': git_detected,
        'config_used': {
            'grid_co2e_kg_per_MWh_avg': avg_co2e,
            'grid_co2e_kg_per_MWh_marginal': marginal_co2e,
        },
        'n_hours': len(df),
    }
    
    # Convert to JSON-safe types
    report_json, _ = to_jsonable(report)
    
    report_file = output_dir / f'grid_emissions_intensity_{year}_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)
    
    return output_file


def generate_signals_manifest(
    output_dir: Path,
    config: dict,
    successful_epochs: List[int],
    script_path: Optional[Path] = None
) -> Path:
    """
    Generate signals_manifest.toml file mapping epochs to file paths and SHA256 hashes.
    """
    # Get git info
    git_hash, git_detected = get_git_commit_hash(output_dir=output_dir, script_path=script_path)
    
    # Get time policy from config
    time_config = config.get('time', {})
    timestamp_output_tz = time_config.get('timestamp_output_tz', 'UTC')
    tariff_clock_tz = time_config.get('tariff_clock_tz', 'Pacific/Auckland')
    dst_policy = time_config.get('dst_policy', 'iana')
    
    # Get GXP ID
    gxp_id = config.get('gxp', {}).get('gxp_id', 'EDN0331')
    
    # Build manifest structure
    manifest_data = {
        'signals': {
            'schema_version': '0.1.0',
            'generated_utc': datetime.now(pytz.UTC).strftime('%Y-%m-%dT%H:%M:%SZ') if HAS_PYTZ else datetime.now().isoformat() + 'Z',
            'gxp_id': gxp_id,
            'timestamp_policy': {
                'timestamp_output_tz': timestamp_output_tz,
                'tariff_clock_tz': tariff_clock_tz,
                'dst_policy': dst_policy,
            },
        }
    }
    
    # Add epoch sections
    for year in successful_epochs:
        gxp_csv = output_dir / f'gxp_hourly_{year}.csv'
        gxp_report = output_dir / f'gxp_hourly_{year}_report.json'
        emissions_csv = output_dir / f'grid_emissions_intensity_{year}.csv'
        emissions_report = output_dir / f'grid_emissions_intensity_{year}_report.json'
        
        epoch_data = {}
        sha256_data = {}
        
        if gxp_csv.exists():
            gxp_hash = sha256_file(gxp_csv).lower()
            epoch_data['gxp_hourly_csv'] = f"outputs_latest/gxp_hourly_{year}.csv"
            epoch_data['gxp_hourly_report_json'] = f"outputs_latest/gxp_hourly_{year}_report.json"
            sha256_data['gxp_hourly'] = gxp_hash
        
        if emissions_csv.exists():
            emissions_hash = sha256_file(emissions_csv).lower()
            epoch_data['grid_emissions_csv'] = f"outputs_latest/grid_emissions_intensity_{year}.csv"
            epoch_data['grid_emissions_report_json'] = f"outputs_latest/grid_emissions_intensity_{year}_report.json"
            sha256_data['grid_emissions'] = emissions_hash
        
        if epoch_data:
            epoch_data['sha256'] = sha256_data
            manifest_data[f'epoch.{year}'] = epoch_data
    
    # Write TOML file (manual formatting since we don't have tomli_w)
    manifest_file = output_dir / 'signals_manifest.toml'
    # Write as formatted TOML manually
    with open(manifest_file, 'w', encoding='utf-8') as f:
            f.write('[signals]\n')
            f.write(f'schema_version = "{manifest_data["signals"]["schema_version"]}"\n')
            f.write(f'generated_utc = "{manifest_data["signals"]["generated_utc"]}"\n')
            f.write(f'gxp_id = "{manifest_data["signals"]["gxp_id"]}"\n')
            f.write('\n[signals.timestamp_policy]\n')
            f.write(f'timestamp_output_tz = "{timestamp_output_tz}"\n')
            f.write(f'tariff_clock_tz = "{tariff_clock_tz}"\n')
            f.write(f'dst_policy = "{dst_policy}"\n')
            f.write('\n')
            for year in successful_epochs:
                epoch_key = f'epoch.{year}'
                if epoch_key in manifest_data:
                    f.write(f'[epoch.{year}]\n')
                    epoch_info = manifest_data[epoch_key]
                    for key, value in epoch_info.items():
                        if key == 'sha256':
                            f.write('\n[epoch.{}.sha256]\n'.format(year))
                            for hash_key, hash_value in value.items():
                                f.write(f'{hash_key} = "{hash_value}"\n')
                        elif isinstance(value, str):
                            f.write(f'{key} = "{value}"\n')
                    f.write('\n')
    
    return manifest_file


def generate_epochs_registry(
    output_dir: Path,
    successful_epochs: List[int],
    config: dict
) -> Path:
    """
    Generate epochs_registry.csv file with epoch metadata.
    """
    rows = []
    for year in successful_epochs:
        # Determine epoch semantics
        if year == 2020:
            epoch_semantics = "baseline_pre_EB"
        elif year == 2025:
            epoch_semantics = "steady_state_post_EB1"
        elif year == 2028:
            epoch_semantics = "steady_state_post_EB2_EB3"
        elif year == 2035:
            epoch_semantics = "decision_epoch"
        else:
            epoch_semantics = "unknown_epoch"
        
        # Load GXP file to get timestamp range
        gxp_file = output_dir / f'gxp_hourly_{year}.csv'
        if gxp_file.exists():
            df = pd.read_csv(gxp_file)
            df_dt = pd.to_datetime(df['timestamp_utc'], utc=True)
            n_hours = len(df)
            start_timestamp_utc = df_dt.min().strftime('%Y-%m-%dT%H:%M:%SZ')
            end_timestamp_utc = df_dt.max().strftime('%Y-%m-%dT%H:%M:%SZ')
        else:
            n_hours = 8784 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 8760
            start_timestamp_utc = f"{year}-01-01T00:00:00Z"
            end_timestamp_utc = f"{year}-12-31T23:00:00Z"
        
        rows.append({
            'epoch': year,
            'epoch_semantics': epoch_semantics,
            'year': year,
            'n_hours': n_hours,
            'start_timestamp_utc': start_timestamp_utc,
            'end_timestamp_utc': end_timestamp_utc,
        })
    
    df_registry = pd.DataFrame(rows)
    registry_file = output_dir / 'epochs_registry.csv'
    df_registry.to_csv(registry_file, index=False, lineterminator='\n')
    
    return registry_file


def validate_output(
    df: pd.DataFrame, 
    year: int, 
    config: dict,
    stats: Dict,
    mode: str = 'synthetic'
) -> bool:
    """
    Validate output DataFrame:
    - Complete hourly coverage
    - No duplicates
    - Strictly increasing timestamps
    - No NaNs in numeric columns
    - Valid ranges
    - Binding fraction within target
    """
    errors = []
    warnings = []
    
    # Get validation flags
    validation_config = config.get('validation', {})
    fail_on_tightness_miss = validation_config.get('fail_on_tightness_miss', False)
    fail_on_emi = validation_config.get('fail_on_emi', False)
    
    # Check expected length (8760 or 8784 for leap year)
    expected_hours = 8784 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 8760
    if len(df) != expected_hours:
        errors.append(f"Expected {expected_hours} hours, got {len(df)}")
    
    # Check for duplicates
    if df['timestamp_utc'].duplicated().any():
        errors.append("Duplicate timestamps found")
    
    # Check strictly increasing
    df_dt = pd.to_datetime(df['timestamp_utc'], utc=True)
    if not df_dt.is_monotonic_increasing:
        errors.append("Timestamps are not strictly increasing")
    
    # Check all required columns (frozen schema)
    missing_cols = set(REQUIRED_GXP_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    
    # Check column order matches frozen schema exactly
    actual_order = list(df.columns)
    if actual_order != REQUIRED_GXP_COLUMNS:
        errors.append(f"Column order mismatch. Expected: {REQUIRED_GXP_COLUMNS}, got: {actual_order}")
    
    # Regression guard: ensure no legacy column names
    legacy_names = ['datetime', 'capacity_MW', 'baseline_import_MW', 'gxp_headroom_MW', 'tariff_base_nzd_per_MWh']
    found_legacy = [name for name in legacy_names if name in df.columns]
    if found_legacy:
        errors.append(f"Legacy column names found: {found_legacy}")
    
    # Check for NaNs in numeric columns
    numeric_cols = ['capacity_mw', 'baseline_import_mw', 'reserve_margin_mw', 'headroom_mw', 'tariff_nzd_per_mwh']
    for col in numeric_cols:
        if col in df.columns and df[col].isna().any():
            errors.append(f"Column {col} contains NaN values")
    
    # Check capacity > 0
    if (df['capacity_mw'] <= 0).any():
        errors.append("capacity_mw must be > 0 for all hours")
    
    # Check headroom >= 0
    if (df['headroom_mw'] < 0).any():
        errors.append("headroom_mw must be >= 0 for all hours")
    
    # Check baseline_import_mw >= 0
    if (df['baseline_import_mw'] < 0).any():
        errors.append("baseline_import_mw must be >= 0 for all hours")
    
    # Check reserve_margin_mw >= 0
    if (df['reserve_margin_mw'] < 0).any():
        errors.append("reserve_margin_mw must be >= 0 for all hours")
    
    # Validate headroom calculation: headroom_mw == max(capacity_mw - baseline_import_mw - reserve_margin_mw, 0)
    computed_headroom = (df['capacity_mw'] - df['baseline_import_mw'] - df['reserve_margin_mw']).clip(lower=0)
    headroom_diff = (df['headroom_mw'] - computed_headroom).abs()
    max_diff = headroom_diff.max()
    if max_diff > 1e-6:
        errors.append(f"headroom_mw calculation mismatch. Max difference: {max_diff:.2e}. Expected: headroom_mw == max(capacity_mw - baseline_import_mw - reserve_margin_mw, 0)")
    
    # Check timestamp format (must have Z suffix)
    if not df['timestamp_utc'].str.endswith('Z').all():
        errors.append("All timestamp_utc values must end with 'Z' (UTC ISO-8601 format)")
    
    # Check timestamps are monotonic and hourly
    df_dt = pd.to_datetime(df['timestamp_utc'], utc=True)
    if not df_dt.is_monotonic_increasing:
        errors.append("Timestamps are not strictly increasing")
    
    # Check for duplicates
    if df_dt.duplicated().any():
        errors.append("Duplicate timestamps found")
    
    # Check hourly spacing (should be exactly 1 hour between consecutive timestamps)
    time_diffs = df_dt.diff().dropna()
    expected_diff = pd.Timedelta(hours=1)
    if not (time_diffs == expected_diff).all():
        errors.append("Timestamps are not evenly spaced at 1-hour intervals")
    
    # Check headroom relationship: gxp_headroom_MW should be <= capacity_MW - baseline_import_MW (allowing for reserve margin)
    # This is a soft check since reserve margin may vary
    
    # Check binding fraction
    max_binding_hours_frac = config['synthetic'].get('max_binding_hours_frac', 0.005)
    binding_frac = stats['binding_frac']
    if binding_frac > max_binding_hours_frac:
        errors.append(
            f"Binding fraction {binding_frac:.2%} exceeds target {max_binding_hours_frac:.2%}"
        )
    
    # Check floor hours percentage (hard fail)
    floor_hits_frac = stats['floor_hits'] / len(df)
    if floor_hits_frac > 0.005:  # 0.5%
        errors.append(
            f"Hours at floor ({floor_hits_frac:.2%}) exceeds 0.5% threshold. "
            f"Current: {stats['floor_hits']} hours. Target: <{int(len(df) * 0.005)} hours."
        )
    
    # Check headroom acceptance limits (warning by default, error if fail_on_tightness_miss=true)
    headroom_acceptance_limits = stats.get('headroom_acceptance_limits', {})
    if headroom_acceptance_limits:
        tightness_tolerance = config.get('tightness_tolerance', {}).get('abs_tol', 0.005)
        for threshold, target_data in headroom_acceptance_limits.items():
            if target_data.get('target') is not None:
                target_val = target_data['target']
                actual_val = target_data['actual']
                passed = target_data.get('pass', True)
                if not passed:
                    msg = f"Headroom acceptance limit {threshold}: actual {actual_val:.3f} exceeds upper bound {target_val:.3f} + tolerance {tightness_tolerance:.3f}"
                    if fail_on_tightness_miss:
                        errors.append(msg)
                    else:
                        warnings.append(msg)
    
    # Hard validation for magnitude targets (2020 synthetic mode only)
    if year == 2020 and mode == 'synthetic':
        baseline_import_mw = stats['baseline_import_mw']
        df_with_baseline = df.copy()
        df_with_baseline['datetime'] = pd.to_datetime(df_with_baseline['timestamp_utc'], utc=True)
        df_with_baseline['baseline_import_mw'] = baseline_import_mw.values
        
        jan_mar = df_with_baseline[df_with_baseline['datetime'].dt.month.isin([1, 2, 3])]
        jun_aug = df_with_baseline[df_with_baseline['datetime'].dt.month.isin([6, 7, 8])]
        oct_nov = df_with_baseline[df_with_baseline['datetime'].dt.month.isin([10, 11])]
        
        jan_mar_mean = jan_mar['baseline_import_mw'].mean()
        jun_aug_mean = jun_aug['baseline_import_mw'].mean()
        oct_nov_mean = oct_nov['baseline_import_mw'].mean()
        
        # Hard fail if magnitude targets not met
        if not (19 <= jan_mar_mean <= 22):
            errors.append(
                f"Jan-Mar mean baseline import {jan_mar_mean:.2f} MW is outside target range [19, 22] MW"
            )
        if not (5 <= jun_aug_mean <= 7):
            errors.append(
                f"Jun-Aug mean baseline import {jun_aug_mean:.2f} MW is outside target range [5, 7] MW"
            )
        if not (23 <= oct_nov_mean <= 26):
            errors.append(
                f"Oct-Nov mean baseline import {oct_nov_mean:.2f} MW is outside target range [23, 26] MW"
            )
    
    # EMI template validation (warning by default, error if fail_on_emi=true)
    if stats.get('emi_template_used', False):
        jan_ratio = stats.get('jan_weekend_weekday_ratio')
        jan_range = stats.get('jan_diurnal_range')
        
        if jan_ratio is not None:
            if not (0.95 <= jan_ratio <= 1.00):
                msg = f"Jan weekend/weekday ratio {jan_ratio:.3f} is outside target range [0.95, 1.00]"
                if fail_on_emi:
                    errors.append(msg)
                else:
                    warnings.append(msg)
        
        if jan_range is not None:
            if jan_range > 4.0:
                msg = f"Jan diurnal range {jan_range:.2f} MW exceeds target ≤4.0 MW"
                if fail_on_emi:
                    errors.append(msg)
                else:
                    warnings.append(msg)
    
    # Print warnings (non-fatal)
    if warnings:
        for warning in warnings:
            print(f"Validation warning: {warning}", file=sys.stderr)
    
    # Print errors (fatal)
    if errors:
        for error in errors:
            print(f"Validation error: {error}", file=sys.stderr)
        return False
    
    return True


def validate_all_outputs(output_dir: Path, config: dict) -> bool:
    """
    Validate all generated outputs in output_dir:
    - Load all gxp_hourly_<YEAR>.csv files and check frozen schema
    - Load all grid_emissions_intensity_<YEAR>.csv files and check schema
    - Verify timestamp format (Z suffix), monotonic, hourly spacing
    - Verify headroom calculation: headroom_mw == max(capacity_mw - baseline_import_mw - reserve_margin_mw, 0)
    - Verify manifest exists and hashes match actual files
    """
    errors = []
    warnings = []
    
    # Find all gxp_hourly CSV files
    gxp_files = sorted(output_dir.glob('gxp_hourly_*.csv'))
    if not gxp_files:
        errors.append(f"No gxp_hourly_*.csv files found in {output_dir}")
        return False
    
    # Find all emissions CSV files
    emissions_files = sorted(output_dir.glob('grid_emissions_intensity_*.csv'))
    
    # Validate each GXP file
    for gxp_file in gxp_files:
        year = int(gxp_file.stem.split('_')[-1])
        print(f"Validating {gxp_file.name}...")
        
        try:
            df = pd.read_csv(gxp_file)
            
            # Check column names and order
            if list(df.columns) != REQUIRED_GXP_COLUMNS:
                errors.append(f"{gxp_file.name}: Column mismatch. Expected: {REQUIRED_GXP_COLUMNS}, got: {list(df.columns)}")
            
            # Regression guard: check for legacy column names
            legacy_names = ['datetime', 'capacity_MW', 'baseline_import_MW', 'gxp_headroom_MW', 'tariff_base_nzd_per_MWh']
            found_legacy = [name for name in legacy_names if name in df.columns]
            if found_legacy:
                errors.append(f"{gxp_file.name}: Legacy column names found: {found_legacy}")
            
            # Check timestamp format (must end with Z)
            if not df['timestamp_utc'].str.endswith('Z').all():
                errors.append(f"{gxp_file.name}: Timestamps must end with 'Z' (UTC ISO-8601)")
            
            # Check timestamps are parseable and monotonic
            try:
                df_dt = pd.to_datetime(df['timestamp_utc'], utc=True)
                if not df_dt.is_monotonic_increasing:
                    errors.append(f"{gxp_file.name}: Timestamps are not strictly increasing")
                if df_dt.duplicated().any():
                    errors.append(f"{gxp_file.name}: Duplicate timestamps found")
                
                # Check hourly spacing
                time_diffs = df_dt.diff().dropna()
                expected_diff = pd.Timedelta(hours=1)
                if not (time_diffs == expected_diff).all():
                    errors.append(f"{gxp_file.name}: Timestamps are not evenly spaced at 1-hour intervals")
            except Exception as e:
                errors.append(f"{gxp_file.name}: Failed to parse timestamps: {e}")
            
            # Validate headroom calculation
            computed_headroom = (df['capacity_mw'] - df['baseline_import_mw'] - df['reserve_margin_mw']).clip(lower=0)
            headroom_diff = (df['headroom_mw'] - computed_headroom).abs()
            max_diff = headroom_diff.max()
            if max_diff > 1e-6:
                errors.append(f"{gxp_file.name}: headroom_mw calculation mismatch. Max difference: {max_diff:.2e}")
            
            # Check for NaNs
            numeric_cols = ['capacity_mw', 'baseline_import_mw', 'reserve_margin_mw', 'headroom_mw', 'tariff_nzd_per_mwh']
            for col in numeric_cols:
                if df[col].isna().any():
                    errors.append(f"{gxp_file.name}: Column {col} contains NaN values")
            
            # Check expected row count
            expected_hours = 8784 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 8760
            if len(df) != expected_hours:
                errors.append(f"{gxp_file.name}: Expected {expected_hours} rows, got {len(df)}")
            
            # Validate hash integrity: check report JSON hash matches CSV
            report_file = output_dir / f'gxp_hourly_{year}_report.json'
            if report_file.exists():
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    # Compute actual hash using sha256_file
                    actual_hash = sha256_file(gxp_file).lower()
                    
                    # Get expected hash from nested structure (preferred)
                    expected_hash_nested = None
                    if 'files' in report_data and 'gxp_hourly_csv' in report_data['files']:
                        expected_hash_nested = report_data['files']['gxp_hourly_csv'].get('sha256', '').lower()
                    
                    # Get expected hash from alias
                    expected_hash_alias = report_data.get('file_hash_sha256', '').lower()
                    
                    # Use nested if available, otherwise alias
                    expected_hash = expected_hash_nested if expected_hash_nested else expected_hash_alias
                    
                    # Fail if both nested and alias exist but disagree
                    if expected_hash_nested and expected_hash_alias and expected_hash_nested != expected_hash_alias:
                        errors.append(
                            f"{gxp_file.name}: Hash inconsistency in report JSON. "
                            f"Nested hash: {expected_hash_nested}, Alias hash: {expected_hash_alias}"
                        )
                    
                    # Fail if hash doesn't match
                    if expected_hash and expected_hash != actual_hash:
                        errors.append(
                            f"{gxp_file.name}: Hash mismatch. "
                            f"Report JSON hash: {expected_hash}, "
                            f"Computed CSV hash: {actual_hash}, "
                            f"File: {gxp_file}"
                        )
                except Exception as e:
                    errors.append(f"{gxp_file.name}: Failed to validate hash from report JSON: {e}")
            
        except Exception as e:
            errors.append(f"{gxp_file.name}: Failed to load/validate: {e}")
    
    # Validate each emissions file
    for emissions_file in emissions_files:
        year = int(emissions_file.stem.split('_')[-1])
        print(f"Validating {emissions_file.name}...")
        
        try:
            df = pd.read_csv(emissions_file)
            
            # Check column names and order
            if list(df.columns) != REQUIRED_EMISS_COLUMNS:
                errors.append(f"{emissions_file.name}: Column mismatch. Expected: {REQUIRED_EMISS_COLUMNS}, got: {list(df.columns)}")
            
            # Check timestamp format
            if not df['timestamp_utc'].str.endswith('Z').all():
                errors.append(f"{emissions_file.name}: Timestamps must end with 'Z' (UTC ISO-8601)")
            
            # Check timestamps match corresponding GXP file
            gxp_file = output_dir / f'gxp_hourly_{year}.csv'
            if gxp_file.exists():
                gxp_df = pd.read_csv(gxp_file)
                if not df['timestamp_utc'].equals(gxp_df['timestamp_utc']):
                    errors.append(f"{emissions_file.name}: Timestamps do not match gxp_hourly_{year}.csv")
            
            # Check for NaNs
            numeric_cols = ['grid_co2e_kg_per_mwh_avg', 'grid_co2e_kg_per_mwh_marginal']
            for col in numeric_cols:
                if df[col].isna().any():
                    errors.append(f"{emissions_file.name}: Column {col} contains NaN values")
            
            # Validate hash integrity: check report JSON hash matches CSV
            report_file = output_dir / f'grid_emissions_intensity_{year}_report.json'
            if report_file.exists():
                try:
                    with open(report_file, 'r', encoding='utf-8') as f:
                        report_data = json.load(f)
                    
                    # Compute actual hash using sha256_file
                    actual_hash = sha256_file(emissions_file).lower()
                    
                    # Get expected hash from nested structure (preferred)
                    expected_hash_nested = None
                    if 'files' in report_data and 'emissions_csv' in report_data['files']:
                        expected_hash_nested = report_data['files']['emissions_csv'].get('sha256', '').lower()
                    
                    # Get expected hash from alias
                    expected_hash_alias = report_data.get('file_hash_sha256', '').lower()
                    
                    # Use nested if available, otherwise alias
                    expected_hash = expected_hash_nested if expected_hash_nested else expected_hash_alias
                    
                    # Fail if both nested and alias exist but disagree
                    if expected_hash_nested and expected_hash_alias and expected_hash_nested != expected_hash_alias:
                        errors.append(
                            f"{emissions_file.name}: Hash inconsistency in report JSON. "
                            f"Nested hash: {expected_hash_nested}, Alias hash: {expected_hash_alias}"
                        )
                    
                    # Fail if hash doesn't match
                    if expected_hash and expected_hash != actual_hash:
                        errors.append(
                            f"{emissions_file.name}: Hash mismatch. "
                            f"Report JSON hash: {expected_hash}, "
                            f"Computed CSV hash: {actual_hash}, "
                            f"File: {emissions_file}"
                        )
                except Exception as e:
                    errors.append(f"{emissions_file.name}: Failed to validate hash from report JSON: {e}")
            
        except Exception as e:
            errors.append(f"{emissions_file.name}: Failed to load/validate: {e}")
    
    # Validate manifest if it exists
    manifest_file = output_dir / 'signals_manifest.toml'
    if manifest_file.exists():
        print(f"Validating {manifest_file.name}...")
        try:
            with open(manifest_file, 'rb') as f:
                manifest_data = tomllib.load(f)
            
            # Check manifest structure
            if 'signals' not in manifest_data:
                errors.append("signals_manifest.toml: Missing [signals] section")
            else:
                signals_section = manifest_data['signals']
                if 'schema_version' not in signals_section:
                    errors.append("signals_manifest.toml: Missing schema_version")
            
            # Strict validation: Check all epochs in manifest have all required files
            expected_epochs = []
            missing_artefacts = []
            
            for epoch_key, epoch_data in manifest_data.items():
                if epoch_key.startswith('epoch.') and not epoch_key.endswith('.sha256'):
                    year = int(epoch_key.split('.')[1])
                    expected_epochs.append(year)
                    
                    # Required files per epoch
                    required_files = {
                        'gxp_hourly_csv': output_dir / f'gxp_hourly_{year}.csv',
                        'gxp_hourly_report_json': output_dir / f'gxp_hourly_{year}_report.json',
                        'grid_emissions_csv': output_dir / f'grid_emissions_intensity_{year}.csv',
                        'grid_emissions_report_json': output_dir / f'grid_emissions_intensity_{year}_report.json',
                    }
                    
                    for file_key, file_path in required_files.items():
                        if not file_path.exists():
                            missing_artefacts.append((year, file_path.name, file_key))
            
            # Check for required non-epoch files
            required_global_files = {
                'gxp_upgrade_menu.csv': output_dir / 'gxp_upgrade_menu.csv',
                'gxp_upgrade_menu_report.json': output_dir / 'gxp_upgrade_menu_report.json',
                'epochs_registry.csv': output_dir / 'epochs_registry.csv',
            }
            
            for file_name, file_path in required_global_files.items():
                if not file_path.exists():
                    missing_artefacts.append((None, file_name, 'global'))
            
            # Report missing artefacts grouped by epoch
            if missing_artefacts:
                errors.append("signals_manifest.toml: Missing required artefacts:")
                missing_by_epoch = {}
                global_missing = []
                
                for epoch, file_name, file_key in missing_artefacts:
                    if epoch is None:
                        global_missing.append(file_name)
                    else:
                        if epoch not in missing_by_epoch:
                            missing_by_epoch[epoch] = []
                        missing_by_epoch[epoch].append(file_name)
                
                if global_missing:
                    errors.append("  Global files:")
                    for file_name in global_missing:
                        errors.append(f"    [MISSING] {file_name}")
                
                if missing_by_epoch:
                    for epoch in sorted(missing_by_epoch.keys()):
                        errors.append(f"  Epoch {epoch}:")
                        for file_name in sorted(missing_by_epoch[epoch]):
                            errors.append(f"    [MISSING] {file_name}")
            
            # Validate epoch sections and hashes (only if files exist)
            for epoch_key, epoch_data in manifest_data.items():
                if epoch_key.startswith('epoch.') and not epoch_key.endswith('.sha256'):
                    year = int(epoch_key.split('.')[1])
                    gxp_file = output_dir / f'gxp_hourly_{year}.csv'
                    if gxp_file.exists():
                        actual_hash = sha256_file(gxp_file).lower()
                        if 'sha256' in epoch_data and 'gxp_hourly' in epoch_data['sha256']:
                            expected_hash = epoch_data['sha256']['gxp_hourly']
                            if actual_hash != expected_hash:
                                errors.append(f"signals_manifest.toml: Hash mismatch for gxp_hourly_{year}.csv. Expected: {expected_hash}, got: {actual_hash}")
                    
                    emissions_file = output_dir / f'grid_emissions_intensity_{year}.csv'
                    if emissions_file.exists():
                        actual_hash = sha256_file(emissions_file).lower()
                        if 'sha256' in epoch_data and 'grid_emissions' in epoch_data['sha256']:
                            expected_hash = epoch_data['sha256']['grid_emissions']
                            if actual_hash != expected_hash:
                                errors.append(f"signals_manifest.toml: Hash mismatch for grid_emissions_intensity_{year}.csv. Expected: {expected_hash}, got: {actual_hash}")
        except Exception as e:
            errors.append(f"signals_manifest.toml: Failed to load/validate: {e}")
    else:
        warnings.append("signals_manifest.toml not found (expected if --no-write-manifest was used)")
    
    # Print results
    if warnings:
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
    
    if errors:
        print("\nValidation FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return False
    else:
        print("\nValidation PASSED: All outputs conform to frozen schema.")
        return True


def print_generation_report(year: int, stats: Dict, n_hours: int, df: pd.DataFrame, quiet: bool = False):
    """Print a sanity summary report for the epoch."""
    baseline_import_mw = stats['baseline_import_mw']
    
    # Calculate seasonal means
    df_with_baseline = df.copy()
    df_with_baseline['datetime'] = pd.to_datetime(df_with_baseline['timestamp_utc'], utc=True)
    df_with_baseline['baseline_import_mw'] = baseline_import_mw.values
    
    jan_mar = df_with_baseline[df_with_baseline['datetime'].dt.month.isin([1, 2, 3])]
    jun_aug = df_with_baseline[df_with_baseline['datetime'].dt.month.isin([6, 7, 8])]
    oct_nov = df_with_baseline[df_with_baseline['datetime'].dt.month.isin([10, 11])]
    
    jan_mar_mean = jan_mar['baseline_import_mw'].mean()
    jun_aug_mean = jun_aug['baseline_import_mw'].mean()
    oct_nov_mean = oct_nov['baseline_import_mw'].mean()
    
    floor_hits_frac = stats['floor_hits'] / n_hours
    
    # Compact output for quiet mode
    if quiet:
        incremental_sens = stats.get('incremental_sensitivity', {})
        deltas = stats.get('incremental_sensitivity_deltas', [2, 5, 10, 15])
        
        print(f"\n{year} Summary:")
        print(f"  Seasonal means: Jan-Mar={jan_mar_mean:.2f} | Jun-Aug={jun_aug_mean:.2f} | Oct-Nov={oct_nov_mean:.2f}")
        print(f"  Headroom: min={stats['headroom_min']:.2f} | p5={stats.get('headroom_p5', 0):.2f} | p50={stats.get('headroom_p50', 0):.2f} | mean={stats['headroom_mean']:.2f}")
        print(f"  Headroom thresholds: <1={stats.get('headroom_lt1_frac', 0):.2%} | <2={stats.get('headroom_lt2_frac', 0):.2%} | <5={stats.get('headroom_lt5_frac', 0):.2%} | <10={stats.get('headroom_lt10_frac', 0):.2%}")
        
        delta_strs = []
        for delta in deltas:
            delta_key = f"hours_bind_if_add_{delta}MW"
            delta_frac_key = f"hours_bind_if_add_{delta}MW_frac"
            hours_bind = incremental_sens.get(delta_key, 0)
            hours_bind_frac = incremental_sens.get(delta_frac_key, 0.0)
            delta_strs.append(f"{delta}MW={hours_bind}({hours_bind_frac:.2%})")
        print(f"  Incremental sensitivity: {' | '.join(delta_strs)}")
        
        if stats.get('emi_template_used', False):
            jan_ratio = stats.get('jan_weekend_weekday_ratio')
            jan_range = stats.get('jan_diurnal_range')
            if jan_ratio is not None and jan_range is not None:
                print(f"  EMI Jan: ratio={jan_ratio:.3f} | range={jan_range:.2f}MW")
        
        return
    
    print(f"\n{'='*70}")
    print(f"Sanity Summary Report for {year}")
    print(f"{'='*70}")
    print(f"Total hours: {n_hours}")
    
    print(f"\nBaseline Import (MW) - Internal:")
    print(f"  Min:  {stats['baseline_min']:.2f}")
    print(f"  Mean: {stats['baseline_mean']:.2f}")
    print(f"  Max:  {stats['baseline_max']:.2f}")
    print(f"  Hours at floor: {stats['floor_hits']} ({floor_hits_frac:.2%})")
    
    print(f"\nHeadroom (MW):")
    print(f"  Min:  {stats['headroom_min']:.2f}")
    print(f"  1st percentile:  {stats.get('headroom_p1', 0):.2f}")
    print(f"  5th percentile:  {stats.get('headroom_p5', 0):.2f}")
    print(f"  10th percentile: {stats.get('headroom_p10', 0):.2f}")
    print(f"  50th percentile: {stats.get('headroom_p50', 0):.2f}")
    print(f"  90th percentile: {stats.get('headroom_p90', 0):.2f}")
    print(f"  Mean: {stats['headroom_mean']:.2f}")
    print(f"  Max:  {stats['headroom_max']:.2f}")
    print(f"  Hours with headroom==0: {stats['binding_count']} ({stats['binding_frac']:.2%})")
    
    print(f"\nHeadroom Threshold Counts:")
    print(f"  Hours with headroom < 1 MW:  {stats.get('headroom_lt1_count', 0)} ({stats.get('headroom_lt1_frac', 0):.2%})")
    print(f"  Hours with headroom < 2 MW:  {stats.get('headroom_lt2_count', 0)} ({stats.get('headroom_lt2_frac', 0):.2%})")
    print(f"  Hours with headroom < 5 MW:  {stats.get('headroom_lt5_count', 0)} ({stats.get('headroom_lt5_frac', 0):.2%})")
    print(f"  Hours with headroom < 10 MW: {stats.get('headroom_lt10_count', 0)} ({stats.get('headroom_lt10_frac', 0):.2%})")
    
    print(f"\nIncremental Demand Sensitivity:")
    incremental_sens = stats.get('incremental_sensitivity', {})
    deltas = stats.get('incremental_sensitivity_deltas', [2, 5, 10, 15])
    for delta in deltas:
        delta_key = f"hours_bind_if_add_{delta}MW"
        delta_frac_key = f"hours_bind_if_add_{delta}MW_frac"
        hours_bind = incremental_sens.get(delta_key, 0)
        hours_bind_frac = incremental_sens.get(delta_frac_key, 0.0)
        print(f"  Hours bind if add {delta} MW:  {hours_bind} ({hours_bind_frac:.2%})")
    
    print(f"\nSeasonal Baseline Import Means (MW):")
    print(f"  Jan-Mar: {jan_mar_mean:.2f}")
    print(f"  Jun-Aug: {jun_aug_mean:.2f}")
    print(f"  Oct-Nov: {oct_nov_mean:.2f}")
    
    # Print EMI template validation if used
    if stats.get('emi_template_used', False):
        jan_ratio = stats.get('jan_weekend_weekday_ratio')
        jan_range = stats.get('jan_diurnal_range')
        print(f"\nEMI Template Validation (January):")
        if jan_ratio is not None:
            print(f"  Weekend/Weekday ratio: {jan_ratio:.3f} (target: 0.95-1.00)")
        if jan_range is not None:
            print(f"  Diurnal range: {jan_range:.2f} MW (target: <=4.0 MW)")
    
    print(f"\nTariff (NZD per MWh):")
    print(f"  Min:  {stats['tariff_min']:.2f}")
    print(f"  Mean: {stats['tariff_mean']:.2f}")
    print(f"  Max:  {stats['tariff_max']:.2f}")
    print(f"  Peak hours:    {stats['tariff_peak_count']}")
    print(f"  Shoulder hours: {stats['tariff_shoulder_count']}")
    print(f"  Off-peak hours: {stats['tariff_offpeak_count']}")
    
    # TOU tariff value counts
    df_with_baseline['tariff'] = df['tariff_nzd_per_mwh']
    tariff_counts = df_with_baseline['tariff'].value_counts().sort_index()
    print(f"\nTOU Tariff Value Counts:")
    for tariff_val, count in tariff_counts.items():
        print(f"  {tariff_val:.2f} NZD/MWh: {count} hours")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hourly GXP input CSVs for specified epochs"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.toml'),
        help='Path to config.toml file (default: config.toml)'
    )
    parser.add_argument(
        '--mode',
        choices=['synthetic', 'measured'],
        default='synthetic',
        help='Generation mode: synthetic (RETA-shaped) or measured (from CSV files) (default: synthetic)'
    )
    parser.add_argument(
        '--measured-dir',
        type=Path,
        help='Directory containing measured data files (required for measured mode)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for CSV files (default: outputs_latest/ relative to repo root)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        default=False,
        help='Reduce console output to essential information only (default: False)'
    )
    parser.add_argument(
        '--epochs',
        type=str,
        default='2020,2025,2028,2035',
        help='Comma-separated list of epochs to generate (default: 2020,2025,2028,2035)'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        default=True,
        help='Stop on first epoch failure (default: True). Use --no-stop-on-error to continue.'
    )
    parser.add_argument(
        '--no-stop-on-error',
        dest='stop_on_error',
        action='store_false',
        help='Continue processing remaining epochs after a failure'
    )
    parser.add_argument(
        '--clean-output',
        choices=['none', 'archive', 'delete'],
        default='none',
        help='Clean existing output files before generation: none (default), archive, or delete'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Override base random seed from config (optional)'
    )
    parser.add_argument(
        '--write-manifest',
        action='store_true',
        default=True,
        help='Write signals_manifest.toml and epochs_registry.csv (default: True)'
    )
    parser.add_argument(
        '--no-write-manifest',
        dest='write_manifest',
        action='store_false',
        help='Do not write manifest files'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Validate existing outputs only (do not generate)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Enable debug output including sample rows (default: False)'
    )
    
    args = parser.parse_args()
    
    # Establish repo root early for portable path resolution
    script_path = Path(__file__).resolve()
    repo_root = get_repo_root(script_path)
    
    # Resolve config path relative to repo root
    if not args.config.is_absolute():
        args.config = resolve_path(args.config, repo_root)
    
    # Handle validate-only mode early (before requiring config/mode)
    if args.validate_only:
        # For validation-only, config is optional (only needed for some checks)
        # Output dir is required - default to outputs_latest if not provided
        if args.output_dir is None:
            args.output_dir = repo_root / 'outputs_latest'
        else:
            # Resolve output_dir relative to repo root if relative
            args.output_dir = resolve_path(args.output_dir, repo_root)
        
        if not args.output_dir.exists():
            print(f"Error: Output directory not found: {args.output_dir}", file=sys.stderr)
            sys.exit(1)
        
        # Resolve config path relative to repo root if relative
        if not args.config.is_absolute():
            args.config = resolve_path(args.config, repo_root)
        
        # Load config if it exists, otherwise use empty dict
        if args.config.exists():
            config = load_config(args.config)
        else:
            config = {}
            print("Warning: Config file not found. Validation will proceed with minimal checks.", file=sys.stderr)
        
        success = validate_all_outputs(args.output_dir, config)
        sys.exit(0 if success else 1)
    
    # Load config (required for generation)
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Validate measured_dir for measured mode
    if args.mode == 'measured' and args.measured_dir is None:
        print("Error: --measured-dir required for measured mode", file=sys.stderr)
        sys.exit(1)
    
    # Resolve measured_dir relative to repo root if provided
    if args.measured_dir is not None:
        args.measured_dir = resolve_path(args.measured_dir, repo_root)
        if not args.measured_dir.exists():
            print(f"Warning: Measured directory not found: {args.measured_dir}", file=sys.stderr)
            print("Will fall back to synthetic mode for all epochs", file=sys.stderr)
    
    # Resolve output directory relative to repo root (default to outputs_latest)
    if args.output_dir is None:
        args.output_dir = repo_root / 'outputs_latest'
    else:
        args.output_dir = resolve_path(args.output_dir, repo_root)
    
    # Create output directory if needed
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clean existing output files if requested
    if args.clean_output != 'none':
        import shutil
        from datetime import datetime
        from collections import defaultdict
        
        # Collect all files in output directory root (exclude directories; _archive is a dir so auto-excluded)
        files_to_clean = []
        for item in args.output_dir.iterdir():
            if item.is_file():
                files_to_clean.append(item)
        
        if files_to_clean:
            if args.clean_output == 'archive':
                # Create archive subdirectory with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                archive_dir = args.output_dir / '_archive' / timestamp
                archive_dir.mkdir(parents=True, exist_ok=True)
                
                # Categorize files for summary
                file_categories = defaultdict(int)
                for file_path in files_to_clean:
                    name = file_path.name
                    if name.startswith('gxp_hourly_') and name.endswith('.csv'):
                        file_categories['gxp_hourly_csv'] += 1
                    elif name.startswith('gxp_hourly_') and name.endswith('_report.json'):
                        file_categories['gxp_hourly_reports'] += 1
                    elif name.startswith('grid_emissions_intensity_') and name.endswith('.csv'):
                        file_categories['emissions_csv'] += 1
                    elif name.startswith('grid_emissions_intensity_') and name.endswith('_report.json'):
                        file_categories['emissions_reports'] += 1
                    elif name == 'gxp_upgrade_menu.csv':
                        file_categories['upgrade_menu_csv'] += 1
                    elif name == 'gxp_upgrade_menu_report.json':
                        file_categories['upgrade_menu_report'] += 1
                    elif name == 'signals_manifest.toml':
                        file_categories['manifest'] += 1
                    elif name == 'epochs_registry.csv':
                        file_categories['registry'] += 1
                    else:
                        file_categories['other'] += 1
                
                # Move files to archive
                for file_path in files_to_clean:
                    dest = archive_dir / file_path.name
                    shutil.move(str(file_path), str(dest))
                
                # Build summary message
                summary_parts = []
                if file_categories['gxp_hourly_csv'] > 0:
                    summary_parts.append(f"{file_categories['gxp_hourly_csv']} GXP hourly CSV")
                if file_categories['gxp_hourly_reports'] > 0:
                    summary_parts.append(f"{file_categories['gxp_hourly_reports']} GXP hourly reports")
                if file_categories['emissions_csv'] > 0:
                    summary_parts.append(f"{file_categories['emissions_csv']} emissions CSV")
                if file_categories['emissions_reports'] > 0:
                    summary_parts.append(f"{file_categories['emissions_reports']} emissions reports")
                if file_categories['upgrade_menu_csv'] > 0:
                    summary_parts.append("upgrade menu CSV")
                if file_categories['upgrade_menu_report'] > 0:
                    summary_parts.append("upgrade menu report")
                if file_categories['manifest'] > 0:
                    summary_parts.append("manifest")
                if file_categories['registry'] > 0:
                    summary_parts.append("registry")
                if file_categories['other'] > 0:
                    summary_parts.append(f"{file_categories['other']} other file(s)")
                
                summary_str = ", ".join(summary_parts) if summary_parts else "files"
                print(f"Clean-output: archive. Archived {len(files_to_clean)} files to {archive_dir.name}/ ({summary_str}).")
            elif args.clean_output == 'delete':
                # Delete files
                for file_path in files_to_clean:
                    file_path.unlink()
                
                print(f"Clean-output: delete. Deleted {len(files_to_clean)} files in {args.output_dir}.")
        else:
            print(f"Clean-output: {args.clean_output}. No files found in {args.output_dir}.")
    
    # Parse epochs from CLI argument
    try:
        epochs = [int(y.strip()) for y in args.epochs.split(',')]
    except ValueError:
        print(f"Error: Invalid --epochs format: {args.epochs}. Use comma-separated integers.", file=sys.stderr)
        sys.exit(1)
    
    # Print epochs to generate
    print(f"Epochs to generate: {epochs}")
    
    # Generate upgrade menu before epoch loop (ensures it runs even if epochs fail)
    script_path = Path(__file__).resolve()
    menu_file = generate_upgrade_menu(config, args.output_dir, script_path=script_path)
    if menu_file:
        # Menu already printed confirmation, no additional output needed
        pass
    
    # Pre-flight config completeness check for all epochs
    missing_configs = []
    for year in epochs:
        year_str = str(year)
        
        # Check epoch_scaling
        if year_str not in config.get('epoch_scaling', {}):
            missing_configs.append(f"epoch_scaling[{year_str}]")
        
        # Check tariff_base
        if year_str not in config.get('tariff_base', {}):
            missing_configs.append(f"tariff_base[{year_str}]")
        
        # Check tightness_targets (if section exists)
        tightness_targets = config.get('tightness_targets', {})
        if tightness_targets and year_str not in tightness_targets:
            missing_configs.append(f"tightness_targets[{year_str}]")
        
        # Check reserve_margin_by_epoch (if section exists)
        reserve_margin_by_epoch = config.get('reserve_margin_by_epoch', {})
        if reserve_margin_by_epoch and year_str not in reserve_margin_by_epoch:
            missing_configs.append(f"reserve_margin_by_epoch[{year_str}]")
    
    if missing_configs:
        print(f"Error: Missing config entries for requested epochs:", file=sys.stderr)
        for missing in missing_configs:
            print(f"  - {missing}", file=sys.stderr)
        sys.exit(1)
    
    # Check for deprecated epoch outputs
    deprecated_files = []
    for dep_year in [2024, 2027]:
        dep_file = args.output_dir / f"gxp_hourly_{dep_year}.csv"
        if dep_file.exists():
            deprecated_files.append(dep_year)
    if deprecated_files:
        print(f"Warning: Deprecated epoch outputs found ({', '.join(map(str, deprecated_files))}). Consider archiving/deleting to avoid confusion.", file=sys.stderr)
    
    # Track successful epochs for manifest
    successful_epochs = []
    
    for year in epochs:
        if not args.quiet:
            print(f"Generating data for {year}...")
        
        try:
            df, stats = generate_gxp_data(year, config, args.mode, args.measured_dir, seed_override=args.seed, repo_root=repo_root)
            
            # Validate output (with hard fail on magnitude targets and other checks)
            if not validate_output(df, year, config, stats, args.mode):
                print(f"ERROR: Validation failed for {year}. CSV will NOT be written.", file=sys.stderr)
                sys.exit(1)
            
            # Write CSV
            output_file = args.output_dir / f"gxp_hourly_{year}.csv"
            
            # Ensure all numeric columns have valid values before writing
            numeric_cols = ['capacity_mw', 'baseline_import_mw', 'reserve_margin_mw', 'headroom_mw', 'tariff_nzd_per_mwh']
            for col in numeric_cols:
                if col in df.columns and df[col].isna().any():
                    raise ValueError(f"Column {col} contains NaN values")
                if col in df.columns and (df[col] == '').any():
                    raise ValueError(f"Column {col} contains empty strings")
            
            # Write without BOM, with newline='\n'
            # Use utf-8 (not utf-8-sig) to avoid BOM
            # Explicitly format numeric columns to ensure they're written
            df_formatted = df.copy()
            for col in numeric_cols:
                if col in df_formatted.columns:
                    df_formatted[col] = df_formatted[col].astype(float)
            
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                df_formatted.to_csv(f, index=False, lineterminator='\n', float_format='%.6f')
            
            # Verify CSV was written
            if not output_file.exists():
                raise RuntimeError(f"CSV not written: {output_file}")
            file_size = output_file.stat().st_size
            if file_size == 0:
                raise RuntimeError(f"CSV is empty: {output_file}")
            
            # Print file provenance information
            abs_path = output_file.resolve()
            file_hash = sha256_file(output_file).lower()
            
            print(f"Written: {output_file} ({len(df)} rows)")
            print(f"  Absolute path: {abs_path}")
            print(f"  File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
            print(f"  SHA256 hash: {file_hash[:16]}... (full: {file_hash})")
            
            # Write provenance report JSON
            script_path = Path(__file__).resolve()
            report = write_provenance_report(output_file, year, config, stats, df, args.mode, script_path=script_path)
            report_file = output_file.parent / f"gxp_hourly_{year}_report.json"
            
            # Verify report was written
            if not report_file.exists():
                raise RuntimeError(f"Provenance report not written: {report_file}")
            
            print(f"  Provenance report: {report_file.resolve()}")
            
            # Print lightweight validation summary
            min_headroom = stats['headroom_min']
            pct_lt1 = stats.get('headroom_lt1_frac', 0) * 100
            pct_lt5 = stats.get('headroom_lt5_frac', 0) * 100
            tariff_min = stats['tariff_min']
            tariff_max = stats['tariff_max']
            peak_cal_info = ""
            if stats.get('peak_calibration_applied', False):
                target = stats.get('peak_calibration_target_MW')
                achieved = stats.get('peak_calibration_achieved_quantile_MW')
                binding_hours = int(stats.get('binding_count', 0))
                peak_cal_info = f", peak_cal={achieved:.2f}MW (target={target:.2f}MW, binding={binding_hours}h)"
            print(f"  Validation: {len(df)} rows, min_headroom={min_headroom:.2f}MW, %<1MW={pct_lt1:.2f}%, %<5MW={pct_lt5:.2f}%, tariff={tariff_min:.2f}-{tariff_max:.2f} NZD/MWh{peak_cal_info}")
            
            # Generate emissions intensity file for this epoch
            emissions_file = generate_emissions_intensity(year, config, args.output_dir, script_path=script_path)
            if not args.quiet:
                emissions_hash = sha256_file(emissions_file).lower()
                print(f"  Emissions: {emissions_file.name} ({len(df)} rows, SHA256: {emissions_hash[:16]}...)")
            
            # Track successful epoch
            successful_epochs.append(year)
            
            # Print detailed sanity summary report (if not quiet)
            if not args.quiet:
                print_generation_report(year, stats, len(df), df, quiet=args.quiet)
            
        except Exception as e:
            print(f"FAILED epoch {year}: {type(e).__name__}: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            if args.stop_on_error:
                sys.exit(1)
            else:
                print(f"Continuing with remaining epochs...", file=sys.stderr)
    
    # Write manifest and registry (must happen even if debug fails)
    if args.write_manifest and successful_epochs:
        try:
            manifest_file = generate_signals_manifest(args.output_dir, config, successful_epochs, script_path=script_path)
            registry_file = generate_epochs_registry(args.output_dir, successful_epochs, config)
            if not args.quiet:
                print(f"\nManifest: {manifest_file.name}")
                print(f"Registry: {registry_file.name}")
        except Exception as e:
            print(f"Warning: Failed to write manifest/registry: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    # Show debug samples only if --debug flag is set (use getattr for safety)
    if getattr(args, 'debug', False):
        try:
            print("\n" + "="*60)
            print("First 5 rows of gxp_hourly_2020.csv:")
            print("="*60)
            output_file_2020 = args.output_dir / "gxp_hourly_2020.csv"
            if output_file_2020.exists():
                df_2020 = pd.read_csv(output_file_2020)
                print(df_2020.head(5).to_string(index=False))
                
                # Show 5 rows around mid-July (winter week)
                print("\n" + "="*60)
                print("Sample rows around mid-July 2020 (winter week):")
                print("="*60)
                # Parse timestamp_utc for filtering (don't modify original DataFrame)
                df_2020_dt = pd.to_datetime(df_2020['timestamp_utc'], utc=True)
                july_mid = pd.Timestamp(2020, 7, 15, 12, tz='UTC')  # Mid-July, noon UTC
                july_df = df_2020[(df_2020_dt >= july_mid - pd.Timedelta(hours=2)) & 
                                  (df_2020_dt <= july_mid + pd.Timedelta(hours=2))]
                if len(july_df) > 0:
                    print(july_df.head(5).to_string(index=False))
                else:
                    # Fallback: show any 5 rows from July
                    july_all = df_2020[df_2020_dt.dt.month == 7]
                    if len(july_all) > 0:
                        print(july_all.iloc[len(july_all)//2:len(july_all)//2+5].to_string(index=False))
            print("="*60)
        except Exception as e:
            # Debug printing must never crash the run
            print(f"Warning: Debug sample printing failed: {e}", file=sys.stderr)
    
    # Final success summary
    print("\n" + "="*70)
    print("Generation Complete - Output Summary")
    print("="*70)
    
    all_files_present = True
    missing_files = []
    
    # Check GXP hourly files
    for year in epochs:
        gxp_file = args.output_dir / f'gxp_hourly_{year}.csv'
        if gxp_file.exists():
            gxp_hash = sha256_file(gxp_file).lower()
            print(f"[OK] gxp_hourly_{year}.csv (SHA256: {gxp_hash[:16]}...)")
        else:
            print(f"[MISSING] gxp_hourly_{year}.csv (MISSING)")
            all_files_present = False
            missing_files.append(f'gxp_hourly_{year}.csv')
        
        emissions_file = args.output_dir / f'grid_emissions_intensity_{year}.csv'
        if emissions_file.exists():
            emissions_hash = sha256_file(emissions_file).lower()
            print(f"[OK] grid_emissions_intensity_{year}.csv (SHA256: {emissions_hash[:16]}...)")
        else:
            print(f"[MISSING] grid_emissions_intensity_{year}.csv (MISSING)")
            all_files_present = False
            missing_files.append(f'grid_emissions_intensity_{year}.csv')
    
    # Check upgrade menu
    menu_file = args.output_dir / 'gxp_upgrade_menu.csv'
    if menu_file.exists():
        menu_hash = sha256_file(menu_file).lower()
        print(f"[OK] gxp_upgrade_menu.csv (SHA256: {menu_hash[:16]}...)")
    else:
        print(f"[MISSING] gxp_upgrade_menu.csv (MISSING)")
        all_files_present = False
        missing_files.append('gxp_upgrade_menu.csv')
    
    # Check manifest and registry
    if args.write_manifest:
        manifest_file = args.output_dir / 'signals_manifest.toml'
        if manifest_file.exists():
            manifest_hash = sha256_file(manifest_file).lower()
            print(f"[OK] signals_manifest.toml (SHA256: {manifest_hash[:16]}...)")
        else:
            print(f"[MISSING] signals_manifest.toml (MISSING)")
            all_files_present = False
            missing_files.append('signals_manifest.toml')
        
        registry_file = args.output_dir / 'epochs_registry.csv'
        if registry_file.exists():
            registry_hash = sha256_file(registry_file).lower()
            print(f"[OK] epochs_registry.csv (SHA256: {registry_hash[:16]}...)")
        else:
            print(f"[MISSING] epochs_registry.csv (MISSING)")
            all_files_present = False
            missing_files.append('epochs_registry.csv')
    
    print("="*70)
    if all_files_present:
        print("[OK] All expected outputs generated successfully")
    else:
        print(f"⚠ Some files are missing: {', '.join(missing_files)}")
    
    if not args.quiet:
        print("\nDone!")
    
    # Exit cleanly with code 0
    sys.exit(0)


if __name__ == '__main__':
    main()

