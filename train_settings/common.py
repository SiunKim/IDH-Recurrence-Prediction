"""
Common training settings and configurations shared across prediction tasks.

This module defines base configurations used by both intra-session and pre-session
prediction models, including:
- Data preprocessing settings
- Training/validation split configurations  
- Feature selection settings
- Directory paths
"""

import dataclasses
from datetime import datetime


@dataclasses.dataclass
class PPSCommon:
    """Base preprocessing settings shared across prediction tasks."""
    
    # Train-validation-test split settings
    test_split_method = 'by_ptid'  # Options: 'by_ptid', 'by_calendar'
    test_split_cv = True if test_split_method == 'by_ptid' else False
    train_fold = 7  
    valid_fold = 1
    test_fold = 2
    
    # Calendar-based split dates
    calendar_date_to_valid = datetime.strptime('2023-08-01', '%Y-%m-%d')
    calendar_date_to_test = datetime.strptime('2023-09-01', '%Y-%m-%d')

    # Data preprocessing flags
    replace_nan_0_ts = True  # Replace NaN with 0 in time series
    minmax_time_inv = True  # Apply min-max scaling to time-invariant features

    # Feature groups
    id_columns = ['session_id', 'ptnt_id', 'ptid']
    
    basic_demo_keys = ['sex', 'age', 'bmi']
    
    diag_keys_all = [
        'AKI_all', 'CVD_all', 'CANCER_all', 'COMP_all', 'DM_all', 
        'GMNR_all', 'HTN_all', 'LUNGD_all', 'PSYCD_all', 'STROKE_all'
    ]
    
    diag_keys_short = [
        'CVD_short', 'PSYCD_short'
    ]
    
    diag_keys_before_ckd = [
        'DM_before_ckd', 'GMNR_before_ckd', 'CVD_before_ckd',
        'HTN_before_ckd', 'LUNGD_before_ckd', 'PSYCD_before_ckd',
        'STROKE_before_ckd'
    ]
    
    presc_keys = {
        'short': [
            'antihypertensive_short', 'anticoagulants_short', 'diuretics_short',
            'alpha-blocker_short', 'beta-blocker_short', 'arbs_short', 'ccb_short',
            'doac_short', 'vka_short', 'insulin_short', 'sulfonylureas_short',
            'DPP4i_short', 'heparin_short', 'antiosteoporosis_short', 'PPI_short',
            'NSAIDs_short', 'hyperkalemia_short', 'phosphate-binder_short',
            'antiparathyroid_short', 'antianemia_short', 'constipation_short',
            'antigout_short', 'antibiotic_short'
        ],
        'long': [
            'antihypertensive_long', 'anticoagulants_long', 'diuretics_long',
            'alpha-blocker_long', 'beta-blocker_long', 'arbs_long', 'ccb_long',
            'doac_long', 'vka_long', 'insulin_long', 'sulfonylureas_long',
            'DPP4i_long', 'heparin_long', 'antiosteoporosis_long', 'PPI_long',
            'NSAIDs_long', 'hyperkalemia_long', 'phosphate-binder_long',
            'antiparathyroid_long', 'antianemia_long', 'constipation_long',
            'antigout_long', 'antibiotic_long'
        ]
    }
    
    lab_keys = [
        'lab_Hct', 'lab_PLT', 'lab_Hb', 'lab_eGFR', 'lab_Creatinine',
        'lab_BUN (serum, em)', 'lab_Alkaline phosphatase', 'lab_BUN',
        'lab_Phosphorus', 'lab_Bilirubin, total', 'lab_Uric Acid',
        'lab_Cholesterol', 'lab_Calcium', 'lab_CRP (hsCRP)',
        'lab_Potassium (serum)', 'lab_GPT (ALT)', 'lab_GOT (AST)',
        'lab_Protein, total', 'lab_Albumin', 'lab_Chloride (serum)',
        'lab_Sodium (serum)', 'lab_CO2, total (serum)'
    ]

    dialyzer_keys = [
        'dialyerz_esa', 'dialyerz_hdmode', 'dialyerz_ufcoef', 'dialyerz_bfv'
    ]
    
    patient_score_keys = [
        'patient_scores_kidney_function', 'patient_scores_spKt/V',
        'patient_scores_e Kt/V', 'patient_scores_nPCR', 'patient_scores_URR'
    ]
    
    hd_record_keys = {
        'cause': [
            'hd_record_hd_cause_DM', 'hd_record_hd_cause_ELSE',
            'hd_record_hd_cause_HT', 'hd_record_hd_cause_IMMUNE',
            'hd_record_dry_weight'
        ],
        'vein': [
            'hd_record_dialysate_Bdex0.1%',
            'hd_record_vein_type_AVF', 'hd_record_vein_type_AVG'
        ]
    }

    # Time-invariant features
    time_inv_features = {
        'nts': ['weight_pre', 'weight_target'],
        
        'baseline_vital': [
            'baseline_sbp', 'baseline_dbp', 'baseline_pulse'
        ],
        
        'prev_vital': [
            'previous_mean_sbp', 'previous_mean_dbp', 'previous_mean_pulse'
        ],
        
        'interdialytic': [
            'interdialytic_weight_change', 'days_interdialysis'
        ],
        
        'time': [
            'week_day', 'AM_PM_int', 'temp', 'pm25'
        ],
        
        'hd_setting': [
            'heparin_enable_set', 'uf_volume_set', 'uf_rate_set',
            'uf_total_time', 'dialysate_flow_set', 'dialysate_flow_real'
        ]
    }

    # Time series features
    ts_features = {
        'set': ['dialysate_temp_set'],
        
        'real': [
            'bloodflow_current', 'bloodpressure_arterial', 
            'bloodpressure_venous', 'bloodpressure_transmem',
            'bodytemp_arterial', 'bodytemp_venous',
            'dialysate_temp_real', 'uf_rate_real'
        ]
    }

    # Dropped features
    dropped_features = [
        'session_id', 'time',
        'uf_maxrate', 'dialysate_bicarbonate_set', 'uf_total_time',
        'dialysate_sodium_real', 'uf_volume_real',
        'vital_sbp', 'vital_dbp', 'vital_pulse', 'vital_map'
    ]

    # Base directory path
    dir_base = 'E:/IDH_prediction_2023_Nov'
    dir_ptid_split = f'{dir_base}/ptid_train_valid_test'

    # CDW preprocessing settings
    cdw_prepro_setting = "demo_lab_selected_minmaxFalse_1211"
    lab_vars = cdw_prepro_setting.split('_minmax')[0].replace('demo_lab_', '')
