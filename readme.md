# Enhancing Real-time Intradialytic Hypotension Prediction by Incorporating Recurrence Patterns

This repository contains the implementation of our paper on real-time Intradialytic Hypotension (IDH) prediction using deep learning methods that incorporate recurrence patterns.

## ⚠️ Important Notice
Due to privacy regulations and hospital data policies, we cannot share the actual patient data or trained models used in our research. This repository serves as a **reference implementation** only. The code structure and implementation details are provided to help researchers understand our methodology and reproduce similar work with their own data.

## Abstract
Intradialytic hypotension (IDH) is a common yet serious complication during hemodialysis treatment. This study presents a novel approach to real-time IDH prediction by incorporating historical patterns of IDH occurrences. Our method combines LSTM-based temporal modeling with comprehensive patient data to achieve improved prediction accuracy.

## Requirements
- Python 3.8+
- PyTorch 2.0.0
- Additional requirements listed in `requirements.txt`

## Installation
```bash
git clone https://github.com/username/idh-prediction.git
cd idh-prediction
pip install -r requirements.txt
```

## Project Structure
```
idh_prediction/
├── data/
│   ├── matched_TDMS/        # TDMS data directory
│   ├── TDMS_pickle/         # TDMS mapping data
│   │   ├── sessionidtdmstime_2207_2401_0208.p        # Session timestamps
│   │   ├── sessionid2ptntid_date_2207_2401_0208.p    # Session-patient mappings
│   │   └── ptntid2sessionids_date_order_2207_2401_0208.p  # Patient-session mappings
│   ├── CDW_pickle/          # Clinical Data Warehouse data
│   │   ├── df_diag_by_ptid.p        # Diagnosis records
│   │   ├── df_presc_by_ptid.p       # Prescription records
│   │   ├── first_height_by_ptid.p   # Patient height data
│   │   ├── patient_scores_by_ptid_date_pros.p  # Patient scores
│   │   ├── dialyzer_type_by_ptid_date_pros.p   # Dialyzer info
│   │   ├── hd_record_by_ptid_pros.p            # HD records
│   │   ├── time2temp.p      # Temperature data
│   │   └── time2pm25.p      # PM2.5 data
│   ├── train_valid_dataset_intrasession/  # Preprocessed datasets
│   └── best_models_intrasession/          # Trained models
└── src/
    ├── define_idh.py         # IDH definition implementations
    ├── model.py              # LSTM model architecture
    ├── prep_dataset_intrasession.py  # Dataset preparation
    ├── prep_train_data_intrasession.py  # Training data preprocessing
    ├── prep_train_data.py    # Common preprocessing utilities
    ├── train_intrasession.py # Training script
    ├── train.py             # Training utilities
    ├── inference.py         # Model inference
    ├── train_valid_split.py # Dataset splitting utilities
    └── train_settings/      # Configuration settings
```

## Data Organization
The project requires three main types of data:

### 1. TDMS Data (`data/matched_TDMS/`)
- Time series data from dialysis machines
- Includes vital signs and machine parameters
- Contains matched time series recordings

### 2. TDMS Mapping Data (`data/TDMS_pickle/`)
- `sessionidtdmstime`: Maps session IDs to timestamps
- `sessionid2ptntid_date`: Links sessions to patient IDs and dates
- `ptntid2sessionids_date_order`: Maps patients to their chronological sessions

### 3. Clinical Data Warehouse (`data/CDW_pickle/`)
- Patient diagnosis records
- Prescription information
- Patient measurements and scores
- Environmental data (temperature, PM2.5)
- HD treatment records and dialyzer information

### 4. Processed Data
- `data/train_valid_dataset_intrasession/`: Preprocessed datasets
- `data/best_models_intrasession/`: Trained model checkpoints

**Note**: The actual data files are not included in this repository due to privacy regulations. Researchers wanting to implement this system will need to prepare similar data structures using their own institutional data.

## Usage
### Data Preprocessing
```python
from src.prep_dataset_intrasession import preprocess_intrasession_data
from src.train_settings.intrasession import PreprocessingSettings

# Configure preprocessing settings
config = PreprocessingSettings()
# Preprocess data
processed_data = preprocess_intrasession_data(config)
```

### Training
```python
from src.train_intrasession import main as train_main

# Start training with cross validation
train_main(train=True, cross_validation_i=0)

# Train with specific IDH type
PPS.idh_type = TRS.idh_type = 'sbp90'  # or 'sbp100', 'kdoqi', etc.
train_main(train=True, cross_validation_i=0)
```

### Inference
```python
from src.inference import main as inference_main

# Run inference
inference_main(
    model_dir="data/best_models_intrasession/", 
    pred_window=30, 
    obs_period=30
)
```

## Model Architecture
Our model architecture consists of three main components:
1. **Time Series Processing**: LSTM layers for processing vital signs and other temporal features
2. **Static Feature Processing**: Fully connected layers for patient demographics and other time-invariant features
3. **Combined Processing**: Integration of both feature types for final prediction

Key features:
- Bidirectional LSTM layers
- Multiple observation window support
- Comprehensive feature integration

## IDH Definitions
We support multiple IDH definitions including:
- SBP < 90 mmHg
- SBP < 100 mmHg
- KDOQI criteria
- SBP drop ≥ 20 mmHg
- SBP drop ≥ 30 mmHg
- Combined criteria

## Reference
```
@article{kim2024enhancing,
  title={Enhancing Real-time Intradialytic Hypotension Prediction by Incorporating Recurrence Patterns},
  author={Kim, Siun and Ryu, Jiwon and Kim, Sejoong and Kim, Su Hwan and Kim, Myeongju and Yoon, Hyung-Jin},
  journal={},
  year={2024}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
We thank all the medical staff and patients who contributed to this research.