"""
Intra-session training data preprocessing module.

This module handles the preprocessing of raw dialysis session data into training datasets,
including:
- Loading and preprocessing TDMS and CDW data
- Feature extraction and engineering
- Handling data before/after first IDH events
- Dataset creation for training
"""
import logging
import pickle
from datetime import timedelta, datetime
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from train_settings.intrasession import PreprocessingSettings as PPS
from define_idh import add_idh_col_to_df_tdms_ts
from utils.prep_train_data import TimeSeriesProcessor, FeatureExtractor

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Time windows
    MAX_TIME_NONBASELINE_BP_RECORDS_MINUTES: int = 45
    
    # File paths
    dir_matched_tdms: str = "dataset/matched_TDMS"
    
    # Prediction settings
    time_pred_window: timedelta = timedelta(minutes=PPS.pred_window)
    time_obs_period: Union[str, timedelta] = (PPS.obs_period if isinstance(PPS.obs_period, str)
                                            else timedelta(minutes=PPS.obs_period))
    leaveout_fromstart: timedelta = timedelta(minutes=PPS.leave_from_end)
    leaveout_fromend: timedelta = timedelta(minutes=PPS.leave_from_end)
    shorter_obs: bool = PPS.shorter_obs


class SessionPreprocessor:
    """Handles preprocessing of individual dialysis sessions."""
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.ts_processor = TimeSeriesProcessor()
        self.feature_extractor = FeatureExtractor()
        
    def prepare_session_data(self, 
                           df_tdms_ts: pd.DataFrame,
                           time_start: pd.Timestamp,
                           time_end: pd.Timestamp
                           ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare time series data for observation and vital signs.
        
        Args:
            df_tdms_ts: Raw TDMS time series data
            time_start: Start time of session
            time_end: End time of session
            
        Returns:
            Tuple of processed DataFrames:
            (df_tdms_ts, df_tdms_observation, df_tdms_vital, df_tdms_vital_all)
        """
        # Add IDH occurrence columns
        df_tdms_ts = add_idh_col_to_df_tdms_ts(df_tdms_ts)
        
        # Get time windows for filtering
        min_start_obs, min_start_vital, max_end = self._get_time_windows(
            time_start, time_end
        )
        
        # Filter DataFrames
        df_tdms_observation = df_tdms_ts[
            (df_tdms_ts['time'] >= min_start_obs) & 
            (df_tdms_ts['time'] <= max_end)
        ].reset_index(drop=True)
        
        df_tdms_vital = df_tdms_ts[
            (df_tdms_ts['time'] >= min_start_vital) & 
            (df_tdms_ts['time'] <= max_end)
        ].reset_index(drop=True)
        
        df_tdms_vital_all = df_tdms_ts.dropna(subset=['vital_sbp']).reset_index(drop=True)
        
        return df_tdms_ts, df_tdms_observation, df_tdms_vital, df_tdms_vital_all
        
    def _get_time_windows(self,
                         time_start: pd.Timestamp,
                         time_end: pd.Timestamp
                         ) -> Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
        """Calculate time windows for data filtering."""
        if self.config.time_obs_period == 'longest':
            min_start_obs = time_start + self.config.leaveout_fromstart
            min_start_vital = time_start + self.config.leaveout_fromstart
        elif self.config.shorter_obs:
            min_start_obs = time_start + self.config.leaveout_fromstart
            min_start_vital = time_start + self.config.time_obs_period + self.config.time_pred_window
        else:
            min_start_obs = time_start + self.config.leaveout_fromstart
            min_start_vital = (time_start + self.config.leaveout_fromstart +
                             self.config.time_obs_period + self.config.time_pred_window)
            
        max_end = time_end - self.config.leaveout_fromend
        return min_start_obs, min_start_vital, max_end

    def get_observation_window(self,
                             df_tdms_observation: pd.DataFrame,
                             df_tdms_vital_all: pd.DataFrame,
                             time_start: pd.Timestamp,
                             target_time: pd.Timestamp
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract observation window for a specific target time.
        
        Args:
            df_tdms_observation: Full observation period data
            df_tdms_vital_all: All vital signs data
            time_start: Session start time
            target_time: Target prediction time
            
        Returns:
            Tuple of (observation window data, vital signs data)
        """
        # Calculate observation start time
        if self.config.time_obs_period == 'longest':
            obs_start = min(df_tdms_observation['time'])
        else:
            target_minutes = (target_time - time_start).seconds / 60
            if (self.config.shorter_obs and 
                target_minutes < self.config.MAX_TIME_NONBASELINE_BP_RECORDS_MINUTES):
                obs_start = (target_time - self.config.time_pred_window -
                           self.config.time_obs_period + self.config.leaveout_fromstart)
            else:
                obs_start = target_time - self.config.time_pred_window - self.config.time_obs_period
                
        obs_end = target_time - self.config.time_pred_window
        
        # Filter observation data
        df_obs_window = df_tdms_observation[
            (df_tdms_observation['time'] >= obs_start) & 
            (df_tdms_observation['time'] <= obs_end)
        ].reset_index(drop=True)
        
        # Remove IDH and vital columns from observation data
        idh_cols = [f'idh_occ_{idh_type}' for idh_type in PPS.idh_types]
        vital_cols = ['vital_sbp', 'vital_dbp', 'vital_map', 'vital_pulse']
        df_obs_window = df_obs_window.drop(idh_cols + vital_cols, axis=1)
        
        # Prepare vital signs data
        df_vital = df_tdms_vital_all[['time', 'vital_sbp', 'vital_dbp', 'vital_pulse'] + idh_cols]
        df_vital = df_vital.dropna(subset=['vital_sbp']).reset_index()
        df_vital['time'] = [(t - time_start).seconds for t in df_vital['time']]
        
        return df_obs_window, df_vital


class DatasetBuilder:
    """Builds training datasets from processed session data."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.session_processor = SessionPreprocessor(config)
        self.feature_extractor = FeatureExtractor()
        
    def build_dataset(self,
                     sessions: List[Dict],
                     data_loader: 'DataLoader'
                     ) -> List[Dict]:
        """
        Build complete training dataset from multiple sessions.
        
        Args:
            sessions: List of session metadata
            data_loader: DataLoader instance for accessing raw data
            
        Returns:
            List of preprocessed training samples
        """
        dataset = []
        stats = {'no_vital_bp': 0, 'missing_data': defaultdict(int)}
        
        for session in tqdm(sessions, desc="Processing sessions"):
            samples = self._process_session(session, data_loader, stats)
            if samples:
                dataset.extend(samples)
                
        self._log_statistics(stats, len(sessions))
        return dataset
        
    def _process_session(self,
                        session: Dict,
                        data_loader: 'DataLoader',
                        stats: Dict) -> Optional[List[Dict]]:
        """
        Process single session into training samples.
        
        Args:
            session: Session metadata dictionary
            data_loader: DataLoader instance for accessing raw data
            stats: Dictionary for tracking processing statistics
            
        Returns:
            Optional list of processed training samples
        """
        try:
            # Load raw TDMS time series data
            df_tdms_ts = data_loader.load_tdms_ts(session['session_id'])
            
            # Check for valid blood pressure measurements
            if not self._has_valid_bp(df_tdms_ts):
                stats['no_vital_bp'] += 1
                return None
                
            # Get required data elements
            data_elements = data_loader.load_session_data(session)
            if not self._validate_data_elements(data_elements, stats):
                return None
                
            # Process time series data
            processed_data = self.session_processor.prepare_session_data(
                df_tdms_ts,
                session['time_start'],
                session['time_end']
            )
            
            # Build samples from processed data
            samples = self._build_session_samples(processed_data, data_elements, session)
            
            if samples:
                return samples
            return None

        except Exception as e:
            print(f"Error processing session {session['session_id']}: {str(e)}")
            return None

    def _build_session_samples(self,
                             processed_data: tuple,
                             data_elements: dict,
                             session: dict) -> List[Dict]:
        """
        Build training samples from processed session data.
        
        Args:
            processed_data: Tuple of (df_tdms_ts, df_tdms_observation, df_tdms_vital, df_tdms_vital_all)
            data_elements: Dictionary of extracted data elements for the session
            session: Session metadata dictionary
            
        Returns:
            List of dictionaries containing processed training samples
        """
        df_tdms_ts, df_tdms_observation, df_tdms_vital, df_tdms_vital_all = processed_data
        samples = []
        
        # Get session times
        time_start = session['time_start'] 
        time_end = session['time_end']
        
        # Calculate observation windows based on settings
        if self.config.time_obs_period == 'longest':
            observation_times = df_tdms_observation['time'].unique()
        else:
            # Create regular time points for prediction
            total_minutes = int((time_end - time_start).total_seconds() / 60)
            step = self.config.min_vital_delta
            observation_times = [
                time_start + timedelta(minutes=m)
                for m in range(0, total_minutes, step)
            ]

        # Process each observation window
        for target_time in observation_times:
            # Get observation window data
            df_obs_window, df_vital = self.session_processor.get_observation_window(
                df_tdms_observation,
                df_tdms_vital_all,
                time_start,
                target_time
            )
            
            if len(df_obs_window) == 0 or len(df_vital) == 0:
                continue

            try:
                # Extract time series features
                time_series = self._extract_time_series(df_obs_window)
                
                # Get target label and time
                label_time = int((target_time - time_start).total_seconds())
                label = any(df_vital[
                    (df_vital['time'] >= target_time) &
                    (df_vital['time'] < target_time + self.config.time_pred_window)
                ][f'idh_occ_{self.config.idh_type}'])

                # Extract time-invariant features
                time_inv_features = self.feature_extractor.extract_timeinv_vars(df_obs_window)

                # Get first IDH times
                first_idh_times = self.feature_extractor.set_first_idh_times(
                    df_vital,
                    time_start,
                    [self.config.idh_type]
                )

                # Build sample dictionary
                sample = {
                    'time_series': time_series,
                    'time_invariant': time_inv_features,
                    'label': label,
                    'label_time': label_time,
                    'session_id': session['session_id'],
                    'first_idh_time': first_idh_times[self.config.idh_type],
                    **data_elements
                }
                
                # Validate sample
                if self._validate_sample(sample):
                    samples.append(sample)

            except Exception as e:
                print(f"Error processing observation window at {target_time}: {str(e)}")
                continue

        return samples

    def _has_valid_bp(self, df: pd.DataFrame) -> bool:
        """Check if dataframe has valid blood pressure measurements."""
        return any(not np.isnan(sbp) for sbp in df['vital_sbp'])
        
    def _validate_data_elements(self,
                              elements: Dict,
                              stats: Dict) -> bool:
        """Validate required data elements are present."""
        if elements is None:
            stats['missing_data']['all'] += 1
            return False
            
        for key, value in elements.items():
            if isinstance(value, bool) and not value:
                stats['missing_data'][key] += 1
                return False
                
            if value is None:
                stats['missing_data'][key] += 1
                return False
                
        return True

    def _extract_time_series(self, df: pd.DataFrame) -> np.ndarray:
        """Extract time series features from observation window."""
        if self.config.time_series_type == 'vital_ts':
            feature_cols = ['vital_sbp', 'vital_dbp', 'vital_pulse']
        elif self.config.time_series_type == 'tdms':
            feature_cols = [
                'bloodflow_current', 'bloodpressure_arterial',
                'bloodpressure_venous', 'bloodpressure_transmem'
            ]
        else:
            raise ValueError(f"Unsupported time series type: {self.config.time_series_type}")

        features = df[feature_cols].values
        
        if self.config.replace_nan_0_ts:
            features = np.nan_to_num(features, 0)
            
        return features

    def _validate_sample(self, sample: Dict) -> bool:
        """Validate processed sample."""
        required_keys = [
            'time_series', 'time_invariant', 'label',
            'label_time', 'session_id', 'first_idh_time'
        ]
        
        if not all(k in sample for k in required_keys):
            return False
            
        if not isinstance(sample['time_series'], np.ndarray):
            return False
            
        if len(sample['time_series'].shape) != 2:
            return False
            
        if not isinstance(sample['time_invariant'], dict):
            return False
            
        if not isinstance(sample['label'], bool):
            return False
            
        if not isinstance(sample['label_time'], (int, float)):
            return False
            
        return True
        
    def _log_statistics(self, stats: Dict, total_sessions: int):
        """Log dataset building statistics."""
        logger.info(f"Sessions without vital BP: {stats['no_vital_bp']}/{total_sessions}")
        for data_type, count in stats['missing_data'].items():
            logger.info(f"Sessions without {data_type}: {count}/{total_sessions}")


class DataLoader:
    """Handles loading and preprocessing of raw dialysis session data."""
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize DataLoader.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config
        
        # Load cached session mappings
        self.session_mappings = self._load_session_mappings()
        
        # Initialize data processors
        self.ts_processor = TimeSeriesProcessor()
        self.feature_extractor = FeatureExtractor()
        
    def _load_session_mappings(self) -> Dict:
        """Load cached session ID mappings and metadata."""
        try:
            with open(f"{self.config.dir_base}/TDMS_pickle/sessionidtdmstime_2207_2401_0208.p", 'rb') as f:
                session_to_time = pickle.load(f)
            with open(f"{self.config.dir_base}/TDMS_pickle/sessionid2ptntid_date_2207_2401_0208.p", 'rb') as f:
                session_to_patient = pickle.load(f)
            with open(f"{self.config.dir_base}/TDMS_pickle/ptntid2sessionids_date_order_2207_2401_0208.p", 'rb') as f:
                patient_to_sessions = pickle.load(f)
                
            return {
                'session_to_time': session_to_time,
                'session_to_patient': session_to_patient,
                'patient_to_sessions': patient_to_sessions
            }
        except FileNotFoundError as e:
            raise RuntimeError(f"Required session mapping file not found: {str(e)}")
            
    def load_sessions(self) -> List[Dict]:
        """
        Load all dialysis sessions with metadata.
        
        Returns:
            List of session dictionaries with metadata
        """
        sessions = []
        
        # Get all session IDs
        for session_id, patient_data in self.session_mappings['session_to_patient'].items():
            time_data = self.session_mappings['session_to_time'].get(session_id)
            if time_data is None:
                continue
                
            sessions.append({
                'session_id': session_id,
                'ptid': patient_data['ptid'],
                'time_start': time_data['time_start'],
                'time_end': time_data['time_end']
            })
            
        return sessions
        
    def load_tdms_ts(self, session_id: str) -> pd.DataFrame:
        """
        Load TDMS time series data for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            DataFrame containing time series data
        """
        # Determine file path based on settings
        folder = "vital_matched_both_nts_ts" if self.config.vital_matched == "matched" else "both_nts_ts"
        file_path = f"{self.config.dir_base}/matched_TDMS/{folder}/session_{session_id}.p"
        
        try:
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
                
            # Add IDH occurrence columns
            df = add_idh_col_to_df_tdms_ts(df)
            
            return df
            
        except FileNotFoundError:
            raise RuntimeError(f"TDMS data file not found for session {session_id}")
            
    def load_session_data(self, session: Dict) -> Dict:
        """
        Load all required data elements for a session.
        
        Args:
            session: Session metadata dictionary
            
        Returns:
            Dictionary containing all data elements
        """
        data = {}
        
        try:
            # Load patient features
            patient_features = self.feature_extractor.get_patient_features(
                session['ptid'],
                session['time_start']
            )
            data.update(patient_features)
            
            # Load environmental data
            env_features = self._load_environmental_data(
                session['time_start']
            )
            data.update(env_features)
            
            # Load dialyzer information
            dialyzer_features = self._load_dialyzer_info(
                session['ptid'],
                session['time_start']
            )
            data.update(dialyzer_features)
            
            return data
            
        except Exception as e:
            print(f"Error loading data for session {session['session_id']}: {str(e)}")
            return None
            
    def _load_environmental_data(self, timestamp: datetime) -> Dict:
        """Load environmental data for timestamp."""
        try:
            with open(f"{self.config.dir_base}/CDW_pickle/time2temp.p", 'rb') as f:
                temp_data = pickle.load(f)
            with open(f"{self.config.dir_base}/CDW_pickle/time2pm25.p", 'rb') as f:
                pm25_data = pickle.load(f)
                
            date_key = timestamp.strftime('%Y-%m-%d')
            return {
                'temperature': temp_data.get(date_key, 14.8),  # Default if missing
                'pm25': pm25_data.get(date_key, 31.0)  # Default if missing
            }
        except FileNotFoundError:
            return {'temperature': 14.8, 'pm25': 31.0}
            
    def _load_dialyzer_info(self, patient_id: str, timestamp: datetime) -> Dict:
        """Load dialyzer information for patient at timestamp."""
        try:
            with open(f"{self.config.dir_base}/CDW_pickle/dialyzer_type_by_ptid_date_pros_240226.p", 'rb') as f:
                dialyzer_data = pickle.load(f)
                
            patient_dialyzer = dialyzer_data.get(patient_id, {})
            date_key = timestamp.strftime('%Y-%m-%d')
            
            if date_key in patient_dialyzer:
                return {
                    'dialyzer_type': patient_dialyzer[date_key],
                    'has_dialyzer_info': True
                }
            return {'has_dialyzer_info': False}
            
        except FileNotFoundError:
            return {'has_dialyzer_info': False}


def _build_session_samples(self,
                         processed_data: tuple,
                         data_elements: dict,
                         session: dict) -> List[Dict]:
    """
    Build training samples from processed session data.
    
    Args:
        processed_data: Tuple of (df_tdms_ts, df_tdms_observation, df_tdms_vital, df_tdms_vital_all)
        data_elements: Dictionary of extracted data elements for the session
        session: Session metadata dictionary
        
    Returns:
        List of dictionaries containing processed training samples
    """
    df_tdms_ts, df_tdms_observation, df_tdms_vital, df_tdms_vital_all = processed_data
    samples = []
    
    # Get session times
    time_start = session['time_start'] 
    time_end = session['time_end']
    
    # Calculate observation windows based on settings
    if self.config.time_obs_period == 'longest':
        observation_times = df_tdms_observation['time'].unique()
    else:
        # Create regular time points for prediction
        total_minutes = int((time_end - time_start).total_seconds() / 60)
        step = self.config.min_vital_delta  # e.g. 25 minutes
        observation_times = [
            time_start + timedelta(minutes=m)
            for m in range(0, total_minutes, step)
        ]

    # Process each observation window
    for target_time in observation_times:
        # Get observation window data
        df_obs_window, df_vital = self.session_processor.get_observation_window(
            df_tdms_observation,
            df_tdms_vital_all,
            time_start,
            target_time
        )
        
        if len(df_obs_window) == 0 or len(df_vital) == 0:
            continue

        try:
            # Extract time series features
            time_series = self._extract_time_series(df_obs_window)
            
            # Get target label and time
            label_time = int((target_time - time_start).total_seconds())
            label = any(df_vital[
                (df_vital['time'] >= target_time) &
                (df_vital['time'] < target_time + self.config.time_pred_window)
            ][f'idh_occ_{self.config.idh_type}'])

            # Extract time-invariant features
            time_inv_features = self.feature_extractor.extract_timeinv_vars(df_obs_window)

            # Get first IDH times
            first_idh_times = self.feature_extractor.set_first_idh_times(
                df_vital,
                time_start,
                [self.config.idh_type]
            )

            # Build sample dictionary
            sample = {
                'time_series': time_series,
                'time_invariant': time_inv_features,
                'label': label,
                'label_time': label_time,
                'session_id': session['session_id'],
                'first_idh_time': first_idh_times[self.config.idh_type],
                **data_elements
            }
            
            # Validate sample
            if self._validate_sample(sample):
                samples.append(sample)

        except Exception as e:
            print(f"Error processing observation window at {target_time}: {str(e)}")
            continue

    return samples

    def _extract_time_series(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract time series features from observation window.
        
        Args:
            df: DataFrame containing observation window data
            
        Returns:
            Array of time series features
        """
        # Select relevant columns based on time series type
        if self.config.time_series_type == 'vital_ts':
            feature_cols = ['vital_sbp', 'vital_dbp', 'vital_pulse']
        elif self.config.time_series_type == 'tdms':
            feature_cols = [
                'bloodflow_current', 'bloodpressure_arterial',
                'bloodpressure_venous', 'bloodpressure_transmem'
            ]
        else:
            raise ValueError(f"Unsupported time series type: {self.config.time_series_type}")

        # Extract and normalize features
        features = df[feature_cols].values
        
        # Handle missing values
        if self.config.replace_nan_0_ts:
            features = np.nan_to_num(features, 0)
            
        return features

    def _validate_sample(self, sample: Dict) -> bool:
        """
        Validate processed sample.
        
        Args:
            sample: Dictionary containing processed features and labels
            
        Returns:
            Boolean indicating if sample is valid
        """
        # Check for required keys
        required_keys = [
            'time_series', 'time_invariant', 'label',
            'label_time', 'session_id', 'first_idh_time'
        ]
        
        if not all(k in sample for k in required_keys):
            return False
            
        # Validate time series features
        if not isinstance(sample['time_series'], np.ndarray):
            return False
            
        if len(sample['time_series'].shape) != 2:
            return False
            
        # Validate time-invariant features
        if not isinstance(sample['time_invariant'], dict):
            return False
            
        # Validate label and times
        if not isinstance(sample['label'], bool):
            return False
            
        if not isinstance(sample['label_time'], (int, float)):
            return False
            
        return True

    def _process_session(self,
                        session: Dict,
                        data_loader: 'DataLoader',
                        stats: Dict
                        ) -> Optional[List[Dict]]:
        """Process single session into training samples."""
        # Load raw data
        df_tdms_ts = data_loader.load_tdms_ts(session['session_id'])
        if not self._has_valid_bp(df_tdms_ts):
            stats['no_vital_bp'] += 1
            return None
            
        # Get required data elements
        data_elements = data_loader.load_session_data(session)
        if not self._validate_data_elements(data_elements, stats):
            return None
            
        # Process time series data
        processed_data = self.session_processor.prepare_session_data(
            df_tdms_ts,
            session['time_start'],
            session['time_end']
        )
        
        # Extract features and build samples
        return self._build_session_samples(processed_data, data_elements, session)
        
    def _has_valid_bp(self, df: pd.DataFrame) -> bool:
        """Check if dataframe has valid blood pressure measurements."""
        return any(not np.isnan(sbp) for sbp in df['vital_sbp'])
        
    def _validate_data_elements(self,
                              elements: Dict,
                              stats: Dict
                              ) -> bool:
        """Validate required data elements are present."""
        for key, value in elements.items():
            if isinstance(value, bool) and not value:
                stats['missing_data'][key] += 1
                return False
        return True
        
    def _log_statistics(self, stats: Dict, total_sessions: int):
        """Log dataset building statistics."""
        logger.info(f"Sessions without vital BP: {stats['no_vital_bp']}/{total_sessions}")
        for data_type, count in stats['missing_data'].items():
            logger.info(f"Sessions without {data_type}: {count}/{total_sessions}")


def preprocess_intrasession_data(config: PreprocessingConfig) -> Dict:
    """
    Main entry point for intra-session data preprocessing.
    
    Args:
        config: Preprocessing configuration
        
    Returns:
        Dictionary containing preprocessed datasets
    """
    data_loader = DataLoader(config)
    dataset_builder = DatasetBuilder(config)

    # Load raw session data
    sessions = data_loader.load_sessions()

    # Build complete dataset
    full_dataset = dataset_builder.build_dataset(sessions, data_loader)

    # Split dataset by IDH occurrence
    datasets = {
        'full': full_dataset,
        'before_first': defaultdict(list),
        'reocc_30min': defaultdict(list),
        'reocc_60min': defaultdict(list)
    }
    for idh_type in PPS.idh_types:
        datasets['before_first'][idh_type] = [
            sample for sample in full_dataset
            if _is_before_first_idh(sample, idh_type)
        ]
        datasets['reocc_30min'][idh_type] = [
            sample for sample in full_dataset
            if _is_reoccurrence(sample, idh_type, minutes=30)
        ]
        datasets['reocc_60min'][idh_type] = [
            sample for sample in full_dataset
            if _is_reoccurrence(sample, idh_type, minutes=60)
        ]

    return datasets


def _is_before_first_idh(sample: Dict, idh_type: str) -> bool:
    """Check if sample is from before first IDH event."""
    first_idh_time = sample[f'time_of_first_idh_{idh_type}']
    if not first_idh_time:
        return True
    pred_time = (sample['time_end'] + timedelta(minutes=PPS.pred_window) -
                sample['time_start']).seconds
    return pred_time <= first_idh_time


def _is_reoccurrence(sample: Dict,
                    idh_type: str,
                    minutes: int
                    ) -> bool:
    """Check if sample is from IDH reoccurrence period."""
    first_idh_time = sample[f'time_of_first_idh_{idh_type}']
    if not first_idh_time:
        return False
    pred_time = (sample['time_end'] + timedelta(minutes=PPS.pred_window) -
                sample['time_start']).seconds
    return pred_time > first_idh_time + (minutes * 60)
