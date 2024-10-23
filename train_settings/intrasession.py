"""
Intra-session IDH prediction model training settings.

This module defines configurations specific to intra-session prediction tasks, including:
- Model architecture settings
- Training hyperparameters  
- Evaluation metrics settings
- Data preprocessing settings for intra-session prediction
"""

import dataclasses
import numpy as np
from train_settings.common import PPSCommon


@dataclasses.dataclass
class TrainingSettings:
    """Training settings for intra-session prediction models."""
    
    # Model architecture
    embedding_dim = 128
    hidden_dim_time_vars = 256
    hidden_dim_time_inv = 256
    layer_dim = 8
    output_dim = 2
    time_inv_init = False

    # Model selection
    model_selection_criterion = 'auroc'  # Options: 'f1', 'auroc'
    
    # Training parameters
    n_epochs = 100
    lr = 2e-5  
    weight_decay = 5e-4
    batch_size = 256
    max_norm = 1.0
    use_early_stop = True
    patience = 15

    # Output settings
    print_graph = True  # Save performance plots
    print_best_model_result = True  # Print metrics for best model
    print_epoch = 25  # Print interval during training
    
    # Evaluation time periods (minutes)
    time_periods = [
        (0, 35), (35, 65), (65, 95), (95, 125),
        (125, 155), (155, 185), (185, 215), (215, np.inf)
    ]

    # Directories
    dir_model_output = f"{PPSCommon.dir_base}/best_models_intrasession"


@dataclasses.dataclass 
class PreprocessingSettings:
    """Preprocessing settings for intra-session prediction."""

    # Directories
    dir_base = PPSCommon.dir_base
    dir_dataset = f"{dir_base}/train_valid_dataset_intrasession"
    dir_train_data = f"{dir_base}/train_data_intrasession"

    # Task definition
    idh_types = ['sbp90', 'sbp100', 'kdoqi', 'sbpd20', 'sbpd30']
    
    # Prediction window settings
    pred_window = 30  # Minutes
    obs_period = 30  # Minutes, or 'longest'
    leave_from_start = 5  # Minutes
    leave_from_end = 20  # Minutes
    min_vital_delta = 25  # Minimum time delta for previous vitals
    shorter_obs = True  # Shorter observation period before 30-40min
    
    # Feature settings
    time_series_type = 'vital_ts'  # Options: 'vital_ts', 'tdms', 'both', 'none'
    vital_matched = 'nonmatched'  # Options: 'matched', 'nonmatched'
    
    # Split settings  
    test_split_method = PPSCommon.test_split_method
    test_split_cv = PPSCommon.test_split_cv
    train_fold = PPSCommon.train_fold 
    valid_fold = PPSCommon.valid_fold
    test_fold = PPSCommon.test_fold
    calendar_date_to_valid = PPSCommon.calendar_date_to_valid
    calendar_date_to_test = PPSCommon.calendar_date_to_test

    # Data filtering
    only_before_first_idh = False
    replace_nan_0_ts = True

    def __post_init__(self):
        """Validate settings after initialization."""
        assert self.pred_window in [0, 10, 30], \
            f"Prediction window must be one of [0, 10, 30], got {self.pred_window}"
        
        assert self.obs_period in [5, 10, 20, 30, 'longest'], \
            f"Observation period must be one of [5, 10, 20, 30, 'longest'], got {self.obs_period}"
        
        assert self.leave_from_start in [0, 5, 30], \
            f"Leave-out from start must be one of [0, 5, 30], got {self.leave_from_start}"
            
        assert self.leave_from_end in [0, 5, 20], \
            f"Leave-out from end must be one of [0, 5, 20], got {self.leave_from_end}"
            
        assert self.vital_matched in ['matched', 'nonmatched'], \
            f"Vital-matched must be one of ['matched', 'nonmatched'], got {self.vital_matched}"

        # Set max sequence length for non-longest observation periods
        self.used_longest = self.obs_period == 'longest'
        self.max_len = self.obs_period + 1 if not self.used_longest else None
        
        # Update train data directory with lab vars
        self.dir_train_data = f"{self.dir_train_data}/{PPSCommon.lab_vars}"
