"""
Dataset preparation module for intra-session IDH prediction.

Handles the creation and preprocessing of datasets including feature scaling
using the refactored TimeSeriesProcessor.
"""
from typing import Tuple, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset

from sklearn import preprocessing

from utils.prep_train_data import TimeSeriesProcessor
from train_settings.intrasession import PreprocessingSettings as PPS
from train_settings.common import PPSCommon


def convert_dataset_to_torch_intrasession(time_inv_total: List[List[float]],
                                        time_vars_total: List[List[List[float]]],
                                        labels: List[bool],
                                        labels_time: List[float],
                                        times_of_first_idh: List[float],
                                        idh_occs_at_pred_time_yn: List[int],
                                        enc: preprocessing.LabelEncoder,
                                        ts_processor: Optional[TimeSeriesProcessor] = None,
                                        session_ids: Optional[List[int]] = None) -> Tuple:
    """
    Convert preprocessed data into PyTorch tensors with scaling.
    
    Args:
        time_inv_total: Time-invariant features
        time_vars_total: Time series features
        labels: Target labels
        labels_time: Time of predictions
        times_of_first_idh: Times of first IDH events
        idh_occs_at_pred_time_yn: IDH occurrence flags
        enc: Label encoder
        ts_processor: TimeSeriesProcessor instance for scaling
        session_ids: Optional session identifiers
        
    Returns:
        Tuple of (TensorDataset, input dimensions, ts_processor)
    """
    # Handle empty dataset case
    if len(time_inv_total) == 0 and len(labels) == 0:
        return [], [], [], False, False

    # Convert labels
    y = torch.tensor(enc.transform(labels), dtype=torch.long)

    # Convert time-invariant features
    x_time_inv = torch.tensor(time_inv_total, dtype=torch.float32)

    # Convert and pad time series features
    if PPS.used_longest:
        x_time_vars = [torch.tensor(tv, dtype=torch.float32)
                      for tv in time_vars_total]
    else:
        # Set maximum sequence length based on settings
        if PPS.time_series_type == 'vital_ts':
            max_len = 18
        elif PPS.time_series_type == 'tdms':
            max_len = 30
        elif PPS.time_series_type == 'none':
            max_len = 1
        else:
            raise ValueError(f'Unsupported ts_type: {PPS.time_series_type}')

        # Truncate or downsample sequences
        if PPS.pred_window <= 30:
            x_time_vars = [torch.tensor(tv[-max_len:], dtype=torch.float32)
                          for tv in time_vars_total]
        else:
            x_time_vars = [torch.tensor(tv[-max_len:][-1::2], dtype=torch.float32)
                          for tv in time_vars_total]

    x_time_vars = pad_sequence(x_time_vars, batch_first=True)

    # Scale features if requested
    if PPSCommon.minmax_time_inv:
        if ts_processor is None:
            ts_processor = TimeSeriesProcessor()
        x_time_vars, x_time_inv, _, _ = ts_processor.scale_features(
            x_time_vars,
            x_time_inv
        )

    # Convert remaining tensors
    labels_time = torch.tensor(labels_time, dtype=torch.long)
    times_of_first_idh = torch.tensor(times_of_first_idh, dtype=torch.long)
    idh_occs_at_pred_time_yn = torch.tensor(idh_occs_at_pred_time_yn, dtype=torch.long)

    if session_ids is not None:
        session_ids = torch.tensor(session_ids, dtype=torch.long)

    # Create appropriate TensorDataset based on settings
    if session_ids is not None:
        dataset = TensorDataset(x_time_inv, x_time_vars, y,
                            labels_time, times_of_first_idh,
                            session_ids)
    else:
        dataset = TensorDataset(x_time_inv, x_time_vars, y,
                                labels_time, times_of_first_idh)

    # Get input dimensions
    input_dim_time_inv = len(x_time_inv[0])
    input_dim_time_vars = len(x_time_vars[0][0])

    return (dataset, input_dim_time_inv, input_dim_time_vars,
            ts_processor)
