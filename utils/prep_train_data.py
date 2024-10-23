"""
Training data preparation utilities module.

This module provides utilities for preparing training data, including:
- Data import from various sources (TDMS, CDW, etc.)
- Feature extraction and preprocessing
- Time series data handling
- Missing value imputation

The module supports both intra-session and pre-session prediction tasks.
"""

"""
Training data preparation utilities - Part 2.

This module provides higher-level functions for:
- Feature extraction from multiple data sources
- Time series processing and aggregation
- Patient and session data integration
- Dataset composition for model training
"""
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Set, Optional

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

from train_settings.intrasession import PPS
from define_idh import add_idh_col_to_df_tdms_ts


# Default values for missing measurements
MEASUREMENT_DEFAULTS = {
    'heparin_enable_set': 0.0,
    'dialysate_flow_set': 500,
    'dialysate_flow_real': 500,
    'uf_volume_set': 2200,
    'uf_total_time': 14400,
    'uf_rate_set': 450,
    'bloodflow_current': 280,
    'temperature': 14.8,
    'pm25': 31.0
}

# Last observation carried forward defaults by column
LOCF_DEFAULTS = {
    'heparin_enable_set': 0.0,
    'uf_volume_set': 2500,
    'uf_rate_set': 750,
    'uf_maxrate': 2000,
    'uf_total_time': 14400,
    'dialysate_flow_set': 500,
    'dialysate_temp_set': 36.5,
    'dialysate_sodium_set': 140,
    'dialysate_bicarbonate_set': 33.0,
    'bloodflow_current': 250,
    'bloodpressure_arterial': -170,
    'bloodpressure_venous': 120,
    'bloodpressure_transmem': 30,
    'bodytemp_arterial': 36.5,
    'bodytemp_venous': 35.0,
    'dialysate_flow_real': 500,
    'dialysate_temp_real': 36.5,
    'dialysate_sodium_real': 140,
    'uf_volume_real': 0,
    'uf_rate_real': 500,
    'vital_sbp': 0,
    'vital_dbp': 0
}

# Time windows for feature extraction
TIME_WINDOWS = {
    'prescription_short': 30,  # days
    'prescription_long': 180,  # days
    'diagnosis_short': 180,  # days
}


def get_cdw_variables(preprocessing_output: str) -> List[str]:
    """
    Extract CDW variables from preprocessing settings output.
    
    Args:
        preprocessing_output: Raw text output from preprocessing
        
    Returns:
        List of CDW variable names
    """
    settings = preprocessing_output.split("Average days interval between measurement:")[1].split('\n')
    return [s.strip().replace('- ', '').split(':')[0] 
            for s in settings if s.strip()]


def get_locf_values(series: pd.Series, default_value: float = 0) -> List[float]:
    """
    Apply last observation carried forward to a time series.
    
    Args:
        series: Input time series
        default_value: Value to use for initial missing values
        
    Returns:
        List with LOCF applied
    """
    values = list(series)
    
    # Handle initial missing value
    if np.isnan(values[0]):
        values[0] = default_value
        
    # Apply LOCF
    last_valid = values[0]
    for i in range(1, len(values)):
        if np.isnan(values[i]):
            values[i] = last_valid
        else:
            last_valid = values[i]
            
    return values


class DataPreprocessingSettings:
    """Data preprocessing settings and configuration."""
    
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self._load_data()
        
    def _load_data(self):
        """Load required data files."""
        # Load TDMS data
        folder_matched = "vital_matched_both_nts_ts" if PPS.matched == "matched" else "both_nts_ts"
        
        with open(f"{self.base_dir}/matched_TDMS/{folder_matched}/df_tdms_nts_total.p", 'rb') as f:
            self.tdms_nts_total = pickle.load(f)
            
        # Load session mappings
        with open(f"{self.base_dir}/TDMS_pickle/sessionidtdmstime_2207_2401_0208.p", 'rb') as f:
            self.session_to_time = pickle.load(f)
            
        with open(f"{self.base_dir}/TDMS_pickle/sessionid2ptntid_date_2207_2401_0208.p", 'rb') as f:
            self.session_to_patient = pickle.load(f)
            
        with open(f"{self.base_dir}/TDMS_pickle/ptntid2sessionids_date_order_2207_2401_0208.p", 'rb') as f:
            self.patient_to_sessions = pickle.load(f)
            
        # Load CDW data
        self._load_cdw_data()
        
        # Load environmental data
        with open(f"{self.base_dir}/CDW_pickle/time2temp.p", 'rb') as f:
            self.time_to_temp = pickle.load(f)
            
        with open(f"{self.base_dir}/CDW_pickle/time2pm25.p", 'rb') as f:
            self.time_to_pm25 = pickle.load(f)
            
        # Load dialyzer information
        self.dialyzer_info = pd.read_excel(f"{self.base_dir}/dialyzer_infos_annotated.xlsx")
        self.dialyzer_by_name = self.dialyzer_info.set_index('dialyzer_name').to_dict(orient='index')
        
    def _load_cdw_data(self):
        """Load Clinical Data Warehouse (CDW) files."""
        def load_pickle(filename: str) -> Dict:
            with open(f"{self.base_dir}/CDW_pickle/{filename}", 'rb') as f:
                return pickle.load(f)
                
        self.diagnosis_by_patient = load_pickle('df_diag_by_ptid.p')
        self.diagnosis_before_ckd = load_pickle('diags_before_ckd_by_ptid.p')
        self.prescriptions = load_pickle('df_presc_by_ptid.p')
        self.first_height = load_pickle('first_height_by_ptid.p')
        self.patient_scores = load_pickle('patient_scores_by_ptid_date_pros_240226.p')
        self.dialyzer_type = load_pickle('dialyzer_type_by_ptid_date_pros_240226.p')
        self.hd_records = load_pickle('hd_record_by_ptid_pros_240226.p')
        self.hd_records_by_date = load_pickle('hd_record_by_ptid_date_pros_240226.p')


def import_tdms_timeseries(dir_path: str, 
                          filename: str) -> Tuple[pd.DataFrame, int]:
    """
    Import TDMS time series data from pickle file.
    
    Args:
        dir_path: Directory containing TDMS files
        filename: Name of pickle file to import
        
    Returns:
        Tuple of (DataFrame with time series data, session ID)
    """
    session_id = int(filename.split('_')[-1].replace('.p', ''))
    
    with open(f"{dir_path}/{filename}", 'rb') as f:
        df = pickle.load(f)
        
    return df, session_id


def extract_timeinv_features(df: pd.DataFrame) -> Dict[str, float]:
    """
    Extract time-invariant features from TDMS time series.
    
    Args:
        df: DataFrame containing TDMS time series
        
    Returns:
        Dictionary of time-invariant features
    """
    # Get record after 1000 seconds
    start_time = df['time'].iloc[0]
    try:
        idx = next(i for i, t in enumerate(df['time']) 
                  if t >= start_time + timedelta(seconds=1000))
        record = df.iloc[idx]
    except StopIteration:
        record = df.iloc[0]
        
    # Extract features with default handling
    features = {}
    feature_map = {
        'heparin_enable_set': ('heparin_enable_set', MEASUREMENT_DEFAULTS['heparin_enable_set']),
        'uf_volume_set': ('uf_volume_set', MEASUREMENT_DEFAULTS['uf_volume_set']),
        'uf_rate_set': ('uf_rate_set', MEASUREMENT_DEFAULTS['uf_rate_set']), 
        'uf_total_time': ('uf_total_time', MEASUREMENT_DEFAULTS['uf_total_time']),
        'dialysate_flow_set': ('dialysate_flow_set', MEASUREMENT_DEFAULTS['dialysate_flow_set']),
        'dialysate_flow_real': ('dialysate_flow_real', MEASUREMENT_DEFAULTS['dialysate_flow_real'])
    }
    
    for feature, (col, default) in feature_map.items():
        value = record[col]
        features[feature] = default if pd.isna(value) or value == 0 else value
        
    # Convert heparin enable to binary
    features['heparin_enable_set'] = int(features['heparin_enable_set'] != 0)
    
    return features


def scale_features(x_time: torch.Tensor,
                  x_inv: torch.Tensor,
                  scaler_time: Optional[RobustScaler] = None,
                  scaler_inv: Optional[RobustScaler] = None
                  ) -> Tuple[torch.Tensor, torch.Tensor, RobustScaler, RobustScaler]:
    """
    Scale time series and time-invariant features.
    
    Args:
        x_time: Time series features tensor
        x_inv: Time-invariant features tensor
        scaler_time: Optional pre-fit scaler for time series
        scaler_inv: Optional pre-fit scaler for time-invariant
        
    Returns:
        Tuple of (scaled time series, scaled time-invariant,
                 time series scaler, time-invariant scaler)
    """
    # Scale time series features
    x_time_flat = x_time.reshape(-1, x_time.shape[-1])
    if scaler_time is None:
        scaler_time = RobustScaler().fit(x_time_flat)
    x_time_scaled = torch.from_numpy(
        scaler_time.transform(x_time_flat)
    ).view(x_time.size())
    
    # Scale time-invariant features
    if scaler_inv is None:
        scaler_inv = RobustScaler().fit(x_inv)
    x_inv_scaled = torch.from_numpy(scaler_inv.transform(x_inv))
    
    return x_time_scaled, x_inv_scaled, scaler_time, scaler_inv

class FeatureExtractor:
    """Extracts and processes features from multiple data sources."""
    def __init__(self, settings: DataPreprocessingSettings):
        self.settings = settings
        
    def get_patient_features(self,
                             patient_id: str,
                             session_time: datetime) -> Dict[str, Union[float, bool]]:
        """
        Extract all patient-level features for a given time.
        
        Args:
            patient_id: Patient identifier
            session_time: Time of the dialysis session
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic demographics
        demo_features = self._get_demographic_features(patient_id, session_time)
        features.update(demo_features)
        
        # Diagnosis features
        diagnosis_features = self._get_diagnosis_features(patient_id, session_time)
        features.update(diagnosis_features)
        
        # Prescription features
        prescription_features = self._get_prescription_features(patient_id, session_time)
        features.update(prescription_features)
        
        # Lab features
        lab_features = self._get_lab_features(patient_id, session_time)
        features.update(lab_features)
        
        return features

    def _get_current_weight(self, patient_id: str, session_time: datetime) -> Optional[float]:
        """
        Get patient's current weight from HD records closest to session time.
        
        Args:
            patient_id: Patient identifier
            session_time: Time of the dialysis session
            
        Returns:
            Float value of patient's current weight, or None if not available
            
        The method:
        1. Checks if patient has HD records
        2. Finds the closest record before session time
        3. Returns pre-dialysis weight from that record
        """
        # Check if patient has HD records
        if patient_id not in self.settings.hd_records_by_date:
            return None

        # Get HD records for patient
        patient_records = self.settings.hd_records_by_date[patient_id]
        # Filter records before session time
        valid_records = [
            (date, record) for date, record in patient_records.items()
            if date <= session_time.date()
        ]
        if not valid_records:
            return None   
        # Get most recent record
        _, closest_record = max(valid_records, key=lambda x: x[0])
        # Extract weight value
        try:
            weight = float(closest_record.get('weight', 0))
            if weight > 0:
                return weight
        except (ValueError, TypeError):
            pass
        # If no valid weight found
        return None

    def _get_demographic_features(self,
                                patient_id: str,
                                session_time: datetime
                                ) -> Dict[str, Union[float, int]]:
        """Extract demographic features."""
        demo_dict = self.settings.basic_demo[patient_id]
        
        features = {
            'sex': 1 if demo_dict['sex'] == 'male' else 0,
            'age': (session_time.date() - demo_dict['birth_date'].date()).days // 365
        }
        
        # Calculate BMI if height available
        if patient_id in self.settings.first_height:
            height = self.settings.first_height[patient_id]
            weight = self._get_current_weight(patient_id, session_time)
            if weight:
                features['bmi'] = weight / ((height/100) ** 2)
                
        return features
        
    def _get_diagnosis_features(self,
                              patient_id: str, 
                              session_time: datetime
                              ) -> Dict[str, bool]:
        """Extract diagnosis history features."""
        features = {}
        
        if patient_id not in self.settings.diagnosis_by_patient:
            return {f"{diag_type}_all": False 
                   for diag_type in self.settings.diagnosis_types}
            
        df_diag = self.settings.diagnosis_by_patient[patient_id]
        
        # All time diagnosis
        all_diagnoses = df_diag[df_diag['첫 진단일자'].dt.date <= session_time.date()]
        features.update({
            f"{diag_type}_all": diag_type in all_diagnoses['type'].values
            for diag_type in self.settings.diagnosis_types
        })
        
        # Recent diagnosis
        recent_cutoff = session_time - timedelta(days=TIME_WINDOWS['diagnosis_short'])
        recent_diagnoses = df_diag[
            (df_diag['첫 진단일자'].dt.date <= session_time.date()) &
            (df_diag['첫 진단일자'].dt.date > recent_cutoff.date())
        ]
        features.update({
            f"{diag_type}_recent": diag_type in recent_diagnoses['type'].values
            for diag_type in self.settings.diagnosis_types
        })
        
        # Pre-CKD diagnosis
        if patient_id in self.settings.diagnosis_before_ckd:
            pre_ckd = self.settings.diagnosis_before_ckd[patient_id]
            features.update({
                f"{diag_type}_before_ckd": diag_type in pre_ckd
                for diag_type in self.settings.diagnosis_types
            })
            
        return features
        
    def _get_prescription_features(self,
                                 patient_id: str,
                                 session_time: datetime
                                 ) -> Dict[str, bool]:
        """Extract prescription history features."""
        features = {}
        
        if patient_id not in self.settings.prescriptions:
            return {f"{drug}_period": False 
                   for drug in self.settings.drug_types
                   for period in ['short', 'long']}
            
        df_presc = self.settings.prescriptions[patient_id]
        
        for period, days in [('short', TIME_WINDOWS['prescription_short']),
                           ('long', TIME_WINDOWS['prescription_long'])]:
            period_start = session_time - timedelta(days=days)
            
            active_prescriptions = df_presc[
                (df_presc['presc_date'] <= session_time.date()) &
                (df_presc['last_dosing_date'] >= period_start.date())
            ]
            
            for drug in self.settings.drug_types:
                features[f"{drug}_{period}"] = \
                    active_prescriptions[drug].any() if len(active_prescriptions) else False
                
        return features

    def _get_lab_features(self,
                         patient_id: str,
                         session_time: datetime,
                         max_days_lookback: int = 30
                         ) -> Dict[str, float]:
        """Extract laboratory test features."""
        features = {}
        
        if patient_id not in self.settings.lab_results:
            return {f"lab_{test}": 0 for test in self.settings.lab_tests}
            
        df_lab = self.settings.lab_results[patient_id]
        
        for test in self.settings.lab_tests:
            # Get test results before session time
            test_results = df_lab[
                (df_lab['test'] == test) &
                (df_lab['date'] < session_time) &
                (df_lab['date'] > session_time - timedelta(days=max_days_lookback))
            ]
            
            if len(test_results):
                # Use most recent result
                features[f"lab_{test}"] = test_results.iloc[-1]['value']
            else:
                features[f"lab_{test}"] = 0
                
        return features
    
    def extract_timeinv_vars(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Extract time-invariant variables from TDMS data.
        
        Args:
            df: TDMS time series DataFrame
            
        Returns:
            Dictionary of time-invariant features
        """
        # Get record after initial period
        start_time = df['time'].iloc[0]
        try:
            idx = next(i for i, t in enumerate(df['time']) 
                      if t >= start_time + timedelta(seconds=1000))
            record = df.iloc[idx]
        except StopIteration:
            record = df.iloc[0]
            
        # Extract features
        features = {
            'heparin_enable_set': int(record['heparin_enable_set'] != 0),
            'uf_volume_set': record['uf_volume_set'],
            'uf_rate_set': record['uf_rate_set'],
            'uf_total_time': record['uf_total_time'],
            'dialysate_flow_set': record['dialysate_flow_set'],
            'dialysate_flow_real': record['dialysate_flow_real']
        }
        
        # Handle missing values
        for key, value in features.items():
            if pd.isna(value) or value == 0:
                features[key] = {
                    'heparin_enable_set': 0,
                    'uf_volume_set': 2200,
                    'uf_rate_set': 450,
                    'uf_total_time': 14400,
                    'dialysate_flow_set': 500,
                    'dialysate_flow_real': 500
                }[key]
                
        return features
    
    def get_covariates_from_row(self,
                               df: pd.DataFrame,
                               target_time: datetime,
                               base_covariates: Dict
                               ) -> Tuple[float, float, float]:
        """
        Extract covariates from time series at specific time.
        
        Args:
            df: TDMS time series DataFrame
            target_time: Target prediction time
            base_covariates: Base covariates dictionary
            
        Returns:
            Tuple of (previous_sbp, previous_dbp, previous_pulse)
        """
        # Get valid measurements before target time
        valid_records = df[
            (df['time'] < target_time) & 
            df['vital_sbp'].notna()
        ].copy()
        
        if len(valid_records) == 0:
            return 0, 0, 0
            
        # Get most recent measurement
        last_record = valid_records.iloc[-1]
        return (
            last_record['vital_sbp'],
            last_record['vital_dbp'],
            last_record['vital_pulse']
        )
    
    def set_first_idh_times(self,
                           df: pd.DataFrame,
                           time_start: datetime,
                           idh_types: List[str]
                           ) -> Dict[str, Optional[float]]:
        """
        Calculate time of first IDH event for each type.
        
        Args:
            df: TDMS time series DataFrame
            time_start: Session start time
            idh_types: List of IDH types to check
            
        Returns:
            Dictionary mapping IDH type to first occurrence time (or False)
        """
        first_idh_times = {}
        
        for idh_type in idh_types:
            # Get IDH occurrences
            idh_occs = df[f'idh_occ_{idh_type}'].fillna(False)
            
            if not any(idh_occs):
                first_idh_times[idh_type] = False
                continue
                
            # Find first occurrence
            first_idx = next(i for i, occ in enumerate(idh_occs) if occ)
            first_time = df['time'].iloc[first_idx]
            
            # Convert to seconds from start
            first_idh_times[idh_type] = int((first_time - time_start).total_seconds())
            
        return first_idh_times

class TimeSeriesProcessor:
    """Processes time series data for feature extraction."""
    
    def __init__(self):
        self.scaler_time = None
        self.scaler_inv = None
        
    def scale_features(self,
                      x_time: np.ndarray,
                      x_inv: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale time series and time-invariant features.
        
        Args:
            x_time: Time series features [batch_size, seq_len, n_features]
            x_inv: Time-invariant features [batch_size, n_features]
            
        Returns:
            Tuple of (scaled time series, scaled time-invariant features)
        """
        # Initialize scalers if needed
        if self.scaler_time is None:
            self.scaler_time = RobustScaler()
            self.scaler_inv = RobustScaler()
            
            # Fit scalers
            x_time_flat = x_time.reshape(-1, x_time.shape[-1])
            self.scaler_time.fit(x_time_flat)
            self.scaler_inv.fit(x_inv)
            
        # Transform features
        x_time_flat = x_time.reshape(-1, x_time.shape[-1])
        x_time_scaled = self.scaler_time.transform(x_time_flat)
        x_time_scaled = x_time_scaled.reshape(x_time.shape)
        
        x_inv_scaled = self.scaler_inv.transform(x_inv)
        
        return x_time_scaled, x_inv_scaled

    def get_vital_statistics(self, 
                           df: pd.DataFrame,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None
                           ) -> Dict[str, float]:
        """
        Calculate vital sign statistics for a time period.
        
        Args:
            df: DataFrame with vital signs
            start_time: Optional period start
            end_time: Optional period end
            
        Returns:
            Dictionary of vital statistics
        """
        if start_time and end_time:
            df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
            
        stats = {}
        for vital in ['sbp', 'dbp', 'pulse']:
            values = df[f'vital_{vital}'].dropna()
            if len(values):
                stats.update({
                    f'{vital}_mean': values.mean(),
                    f'{vital}_std': values.std(),
                    f'{vital}_min': values.min(),
                    f'{vital}_max': values.max()
                })
            else:
                stats.update({
                    f'{vital}_mean': 0,
                    f'{vital}_std': 0,
                    f'{vital}_min': 0,
                    f'{vital}_max': 0
                })
        return stats




class DatasetBuilder:
    """Builds training datasets from processed features."""
    
    def __init__(self,
                 feature_extractor: FeatureExtractor,
                 ts_processor: TimeSeriesProcessor):
        self.feature_extractor = feature_extractor
        self.ts_processor = ts_processor
        
    def build_session_dataset(self,
                            session_ids: List[str],
                            obs_window: int,
                            pred_window: int
                            ) -> List[Dict]:
        """
        Build dataset for session-level prediction.
        
        Args:
            session_ids: List of session IDs to process
            obs_window: Observation window in minutes
            pred_window: Prediction window in minutes
            
        Returns:
            List of dictionaries containing processed features
        """
        dataset = []
        
        for session_id in session_ids:
            session_data = self._process_session(
                session_id,
                obs_window,
                pred_window
            )
            if session_data:
                dataset.append(session_data)
                
        return dataset
        
    def _process_session(self,
                        session_id: str,
                        obs_window: int,
                        pred_window: int
                        ) -> Optional[Dict]:
        """Process single session data."""
        # Get session info
        session_info = self.feature_extractor.settings.session_to_patient[session_id]
        patient_id = session_info['ptid']
        session_time = session_info['time']
        
        # Extract features
        try:
            # Patient features
            patient_features = self.feature_extractor.get_patient_features(
                patient_id,
                session_time
            )
            
            # Time series features
            ts_features = self._get_timeseries_features(
                session_id,
                obs_window,
                pred_window
            )
            
            return {
                'session_id': session_id,
                'patient_id': patient_id,
                'session_time': session_time,
                **patient_features,
                **ts_features
            }
            
        except Exception as e:
            print(f"Error processing session {session_id}: {str(e)}")
            return None
            
    def _get_timeseries_features(self,
                               session_id: str,
                               obs_window: int,
                               pred_window: int
                               ) -> Dict:
        """Extract time series features for a session."""
        df = self.feature_extractor.settings.tdms_by_session[session_id]
        
        # Define time windows
        end_time = df['time'].max()
        pred_start = end_time - timedelta(minutes=pred_window)
        obs_start = pred_start - timedelta(minutes=obs_window)
        
        # Get vital statistics
        obs_vitals = self.ts_processor.get_vital_statistics(
            df,
            obs_start,
            pred_start
        )
        
        # Get baseline measures
        baselines = self.ts_processor.get_baseline_measures(
            df,
            ['vital_sbp', 'vital_dbp', 'vital_pulse']
        )
        
        return {
            **obs_vitals,
            **baselines
        }
