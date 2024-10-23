"""
Intra-dialytic Hypotension (IDH) definition and detection module.

This module implements various IDH definitions and provides functionality to detect
IDH events in dialysis time series data. Supported IDH definitions include:
- SBP threshold based (SBP < 90 or 100 mmHg)
- KDOQI criteria (SBP drop ≥ 20 mmHg or MAP drop ≥ 10 mmHg)
- SBP drop based (≥ 20 or 30 mmHg decrease)
- Combined criteria (SBP < 90 mmHg + SBP drop)
- HanbiLee criteria (baseline-dependent thresholds)

Key Functions:
- get_idh_occ_* : Implementation of each IDH definition
- set_baseline_bp : Calculate baseline blood pressure
- add_idh_col_to_df : Add IDH occurrence columns to dataframe
"""

from typing import Tuple, List
import numpy as np
import pandas as pd


def get_idh_occ_sbp90(vital_sbps: List[float]) -> List[bool]:
    """
    Detect IDH events based on SBP < 90 mmHg criterion.
    
    Args:
        vital_sbps: List of systolic blood pressure measurements
        
    Returns:
        List of boolean values indicating IDH occurrences
    """
    return [bool(sbp < 90) for sbp in vital_sbps]


def get_idh_occ_sbp100(vital_sbps: List[float]) -> List[bool]:
    """
    Detect IDH events based on SBP < 100 mmHg criterion.
    
    Args:
        vital_sbps: List of systolic blood pressure measurements
        
    Returns:
        List of boolean values indicating IDH occurrences
    """
    return [bool(sbp < 100) for sbp in vital_sbps]


def set_baseline_bp_map(df_tdms_ts: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate baseline SBP and MAP from initial measurements.
    
    Uses average of first two measurements if interval < 15 min,
    otherwise uses first measurement only.
    
    Args:
        df_tdms_ts: Dialysis time series dataframe
        
    Returns:
        Tuple of (baseline_sbp, baseline_map)
    """
    # Get valid SBP measurement times
    times_valid_sbps = df_tdms_ts[df_tdms_ts['vital_sbp'].notna()]['time']
    
    try:
        # Check interval between first two measurements
        time_interval = (times_valid_sbps.iloc[1] - times_valid_sbps.iloc[0]).seconds / 60
        
        if time_interval < 15:
            # Use average of first two measurements
            valid_sbps = [sbp for sbp in df_tdms_ts['vital_sbp'] if not np.isnan(sbp)][:2]
            valid_maps = [map_val for map_val in df_tdms_ts['vital_map'] if not np.isnan(map_val)][:2]
            baseline_sbp = np.mean(valid_sbps)
            baseline_map = np.mean(valid_maps)
        else:
            # Use first measurement only
            baseline_sbp = [sbp for sbp in df_tdms_ts['vital_sbp'] if not np.isnan(sbp)][0]
            baseline_map = [map_val for map_val in df_tdms_ts['vital_map'] if not np.isnan(map_val)][0]
            
    except IndexError:
        # Handle case with only one measurement
        baseline_sbp = [sbp for sbp in df_tdms_ts['vital_sbp'] if not np.isnan(sbp)][0]
        baseline_map = [map_val for map_val in df_tdms_ts['vital_map'] if not np.isnan(map_val)][0]
        
    return baseline_sbp, baseline_map


def get_idh_occ_kdoqi(vital_sbps: List[float],
                      vital_maps: List[float],
                      baseline_sbp: float,
                      baseline_map: float) -> List[bool]:
    """
    Detect IDH events based on KDOQI criteria:
    - SBP decrease ≥ 20 mmHg OR
    - MAP decrease ≥ 10 mmHg
    
    Args:
        vital_sbps: List of SBP measurements
        vital_maps: List of MAP measurements
        baseline_sbp: Baseline SBP value
        baseline_map: Baseline MAP value
        
    Returns:
        List of boolean values indicating IDH occurrences
    """
    # Check SBP decrease ≥ 20 mmHg
    idh_occ_sbp = [bool((sbp - baseline_sbp) <= -20) if not np.isnan(sbp) else False
                   for sbp in vital_sbps]
    
    # Check MAP decrease ≥ 10 mmHg
    idh_occ_map = [bool((map_val - baseline_map) <= -10) if not np.isnan(map_val) else False
                   for map_val in vital_maps]
    
    # Combine criteria (either condition satisfied)
    return [sbp_dec or map_dec for sbp_dec, map_dec in zip(idh_occ_sbp, idh_occ_map)]


def get_idh_occ_sbpd20(vital_sbps: List[float],
                       baseline_sbp: float) -> List[bool]:
    """
    Detect IDH events based on SBP decrease ≥ 20 mmHg criterion.
    
    Args:
        vital_sbps: List of SBP measurements
        baseline_sbp: Baseline SBP value
        
    Returns:
        List of boolean values indicating IDH occurrences
    """
    return [bool((sbp - baseline_sbp) <= -20) for sbp in vital_sbps]


def get_idh_occ_sbpd30(vital_sbps: List[float],
                       baseline_sbp: float) -> List[bool]:
    """
    Detect IDH events based on SBP decrease ≥ 30 mmHg criterion.
    
    Args:
        vital_sbps: List of SBP measurements
        baseline_sbp: Baseline SBP value
        
    Returns:
        List of boolean values indicating IDH occurrences
    """
    return [bool((sbp - baseline_sbp) <= -30) for sbp in vital_sbps]


def get_idh_occ_sbp90_sbpd20(vital_sbps: List[float],
                             baseline_sbp: float) -> List[bool]:
    """
    Detect IDH events based on combined criteria:
    - SBP < 90 mmHg AND
    - SBP decrease ≥ 20 mmHg
    
    Args:
        vital_sbps: List of SBP measurements
        baseline_sbp: Baseline SBP value
        
    Returns:
        List of boolean values indicating IDH occurrences
    """
    return [bool(sbp < 90) and bool((sbp - baseline_sbp) <= -20)
            for sbp in vital_sbps]


def get_idh_occ_sbp90_sbpd30(vital_sbps: List[float],
                             baseline_sbp: float) -> List[bool]:
    """
    Detect IDH events based on combined criteria:
    - SBP < 90 mmHg AND
    - SBP decrease ≥ 30 mmHg
    
    Args:
        vital_sbps: List of SBP measurements
        baseline_sbp: Baseline SBP value
        
    Returns:
        List of boolean values indicating IDH occurrences
    """
    return [bool(sbp < 90) and bool((sbp - baseline_sbp) <= -30)
            for sbp in vital_sbps]


def get_idh_occ_hanbilee(vital_sbps: List[float],
                         baseline_sbp: float) -> List[bool]:
    """
    Detect IDH events based on HanbiLee criteria:
    - For baseline SBP ≥ 160: SBP < 100 mmHg
    - For 90 ≤ baseline SBP < 160: SBP < 90 mmHg
    - For baseline SBP < 90: SBP decrease ≥ 20 mmHg
    
    Args:
        vital_sbps: List of SBP measurements
        baseline_sbp: Baseline SBP value
        
    Returns:
        List of boolean values indicating IDH occurrences
    """
    def get_threshold(baseline: float) -> callable:
        if baseline >= 160:
            return lambda x: x < 100
        elif 90 <= baseline < 160:
            return lambda x: x < 90
        else:  # baseline < 90
            return lambda x: (x - baseline) <= -20
            
    threshold_func = get_threshold(baseline_sbp)
    return [bool(threshold_func(sbp)) for sbp in vital_sbps]


def add_idh_col_to_df_tdms_ts(df_tdms_ts: pd.DataFrame) -> pd.DataFrame:
    """
    Add IDH occurrence columns to dialysis time series dataframe for all definitions.
    
    Args:
        df_tdms_ts: Dialysis time series dataframe
        
    Returns:
        DataFrame with added IDH occurrence columns
    """
    # Calculate MAP
    df_tdms_ts['vital_map'] = (df_tdms_ts['vital_sbp'] + 2*df_tdms_ts['vital_dbp'])/3
    
    # Get BP measurements
    vital_sbps = df_tdms_ts['vital_sbp']
    vital_maps = df_tdms_ts['vital_map']
    
    # Calculate baseline values
    baseline_sbp, baseline_map = set_baseline_bp_map(df_tdms_ts)
    
    # Add IDH occurrence columns for each definition
    df_tdms_ts['idh_occ_sbp90'] = get_idh_occ_sbp90(vital_sbps)
    df_tdms_ts['idh_occ_sbp100'] = get_idh_occ_sbp100(vital_sbps)
    df_tdms_ts['idh_occ_kdoqi'] = get_idh_occ_kdoqi(vital_sbps, vital_maps,
                                                    baseline_sbp, baseline_map)
    df_tdms_ts['idh_occ_sbpd20'] = get_idh_occ_sbpd20(vital_sbps, baseline_sbp)
    df_tdms_ts['idh_occ_sbpd30'] = get_idh_occ_sbpd30(vital_sbps, baseline_sbp)
    df_tdms_ts['idh_occ_sbp90_sbpd20'] = get_idh_occ_sbp90_sbpd20(vital_sbps, baseline_sbp)
    df_tdms_ts['idh_occ_sbp90_sbpd30'] = get_idh_occ_sbp90_sbpd30(vital_sbps, baseline_sbp)
    df_tdms_ts['idh_occ_hanbilee'] = get_idh_occ_hanbilee(vital_sbps, baseline_sbp)
    
    return df_tdms_ts
