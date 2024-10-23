"""
Intra-session IDH prediction model inference module.

This module handles inference for trained IDH prediction models, including:
- Loading trained models and test data
- Running inference on test sets
- Calculating and saving performance metrics
- Generating visualization plots

Example Usage:
    # Basic inference for all models in a directory
    python inference_intra.py --model_dir models/intra_session --pred_window 30 --obs_period 30
    
    # Inference for specific IDH type with custom settings
    python inference_intra.py --model_dir models/intra_session --idh_type kdoqi \\
        --pred_window 30 --obs_period 30 --batch_size 64 --performance_metric auroc
    
    # Save results to specific directory
    python inference_intra.py --model_dir models/intra_session --pred_window 30 \\
        --obs_period 30 --output_dir results/intra_session
"""

import os
import pickle
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from model import LSTMClassifier
from utils.train import (calculate_performances, compute_all_metrics_with_ci,
                        draw_auroc_curve)


class InferenceConfig:
    """Configuration for inference settings."""
    
    # Performance metrics to track
    metrics_of_interest = [
        'accuracy', 'accuracy_95ci_lower', 'accuracy_95ci_upper', 
        'precision', 'precision_95ci_lower', 'precision_95ci_upper', 
        'specificity', 'specificity_95ci_lower', 'specificity_95ci_upper', 
        'recall', 'recall_95ci_lower', 'recall_95ci_upper', 
        'f1', 'f1_95ci_lower', 'f1_95ci_upper', 
        'auroc', 'auroc_95ci_lower', 'auroc_95ci_upper', 
        'auprc', 'auprc_95ci_lower', 'auprc_95ci_upper', 
        'mcc', 'mcc_95ci_lower', 'mcc_95ci_upper', 
        'npv', 'npv_95ci_lower', 'npv_95ci_upper', 
        'n_sample', 'n_sample_positive'
    ]
    
    # Time periods for evaluation
    time_periods = ['totalsession', 'early', 'late', 
                   '35-65', '65-95', '95-125', '125-155', 
                   '155-185', '185-215', '215-inf']
    
    before_after_periods = ['before_first_idh', 'after_first_idh', 'idh_reocc']
    
    # Minimum F1 score threshold for model selection
    min_f1_threshold = 0.05


def inference_model(model: torch.nn.Module, 
                   test_dl: DataLoader, 
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on test dataset using trained model.
    
    Args:
        model: Trained PyTorch model
        test_dl: Test data loader
        device: Device to run inference on
        
    Returns:
        Tuple of (y_test, probabilities, y_times, first_idh_times)
    """
    model.eval()
    with torch.no_grad():
        y_test = torch.empty(0).to(device)
        probs = torch.empty(0).to(device)
        y_times = torch.empty(0).to(device)
        first_idh_times = torch.empty(0).to(device)

        for data in test_dl:
            x_time_inv, x_time_vars, y, y_time, time_of_first_idh = \
                [d.to(device) for d in data]
            x_time_inv = x_time_inv.float()
            x_time_vars = x_time_vars.float()

            output, _ = model(x_time_inv, x_time_vars)
            batch_probs = F.softmax(output, dim=1)

            y_test = torch.cat((y_test, y))
            probs = torch.cat((probs, batch_probs))
            y_times = torch.cat((y_times, y_time))
            first_idh_times = torch.cat((first_idh_times, time_of_first_idh))

    return (y_test.cpu().numpy(), 
            probs.cpu().numpy()[:, 1],
            y_times.cpu().numpy(), 
            first_idh_times.cpu().numpy())


def majority_vote(predictions: List[np.ndarray]) -> np.ndarray:
    """
    Compute majority vote from multiple model predictions.
    
    Args:
        predictions: List of model predictions
        
    Returns:
        Array of majority vote predictions
    """
    votes = np.array(predictions)
    return (votes.mean(axis=0) >= 0.5).astype(bool)


def calculate_metrics_by_time_period(y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_prob: np.ndarray,
                                   y_times: np.ndarray,
                                   first_idh_times: np.ndarray) -> Dict:
    """
    Calculate performance metrics for different time periods.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        y_times: Prediction times
        first_idh_times: Times of first IDH events
        
    Returns:
        Dictionary of metrics for each time period
    """
    metrics = {}
    
    # Calculate overall metrics
    metrics['totalsession'] = compute_all_metrics_with_ci(
        y_true=y_true, y_pred=y_pred, y_prob=y_prob)
    
    # Calculate metrics for early/late periods
    early_mask = y_times < 3600  # First hour
    metrics['early'] = compute_all_metrics_with_ci(
        y_true=y_true[early_mask], 
        y_pred=y_pred[early_mask],
        y_prob=y_prob[early_mask])
    
    metrics['late'] = compute_all_metrics_with_ci(
        y_true=y_true[~early_mask],
        y_pred=y_pred[~early_mask], 
        y_prob=y_prob[~early_mask])

    return metrics


def load_test_dataset(idh_type: str,
                     pred_window: int,
                     obs_period: int,
                     data_dir: str) -> Tuple[DataLoader, Dict]:
    """
    Load test dataset and preprocessing settings.
    
    Args:
        idh_type: Type of IDH to predict
        pred_window: Prediction window in minutes
        obs_period: Observation period in minutes
        data_dir: Directory containing datasets
        
    Returns:
        Tuple of (test_dataloader, preprocessing_settings)
    """
    dataset_path = os.path.join(
        data_dir,
        f"predwindow{pred_window}_obsperiod{obs_period}",
        f"train_valid_dataset_idh_{idh_type}_5cv.p"
    )
    
    with open(dataset_path, 'rb') as f:
        test_dataset = pickle.load(f)
        
    with open(os.path.join(data_dir, 'PPS_preprocessing_train_valid_dataset.p'), 'rb') as f:
        preprocessing_settings = pickle.load(f)
        
    test_dl = DataLoader(test_dataset[5], batch_size=128, shuffle=False)
    
    return test_dl, preprocessing_settings


def save_results(metrics: Dict,
                output_dir: str,
                idh_type: str,
                pred_window: int) -> None:
    """
    Save inference results to Excel file.
    
    Args:
        metrics: Dictionary of performance metrics
        output_dir: Output directory for results
        idh_type: Type of IDH
        pred_window: Prediction window in minutes
    """
    date_str = datetime.today().strftime('%m%d')
    output_path = os.path.join(
        output_dir,
        f'performance_results_idh_{idh_type}_predw{pred_window}_{date_str}.xlsx'
    )
    
    results_df = pd.DataFrame.from_dict(metrics, orient='index')
    results_df.to_excel(output_path)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run inference for intra-session IDH prediction models.'
    )
    
    parser.add_argument('--model_dir', required=True,
                      help='Directory containing trained models')
    
    parser.add_argument('--pred_window', type=int, required=True,
                      help='Prediction window in minutes')
    
    parser.add_argument('--obs_period', type=int, required=True,
                      help='Observation period in minutes')
    
    parser.add_argument('--idh_type', default=None,
                      choices=['sbp90', 'sbp100', 'kdoqi', 'sbpd20', 'sbpd30'],
                      help='Specific IDH type to evaluate (default: evaluate all)')
    
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size for inference')
    
    parser.add_argument('--output_dir', default='results',
                      help='Directory to save results')
    
    parser.add_argument('--performance_metric', default='auroc',
                      choices=['auroc', 'f1'],
                      help='Primary metric for model selection')
    
    parser.add_argument('--device', default='cuda',
                      help='Device to run inference on')
    
    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of IDH types to evaluate
    idh_types = [args.idh_type] if args.idh_type else \
        ['sbp90', 'sbp100', 'kdoqi', 'sbpd20', 'sbpd30']
        
    for idh_type in idh_types:
        print(f"\nEvaluating IDH type: {idh_type}")
        
        # Load test data
        test_dl, preprocessing_settings = load_test_dataset(
            idh_type, args.pred_window, args.obs_period, args.model_dir
        )
        
        # Load models
        model_files = [f for f in os.listdir(args.model_dir) 
                      if f.endswith('.pt') and idh_type in f]
        
        ensemble_predictions = []
        for model_file in model_files:
            model = torch.load(os.path.join(args.model_dir, model_file))
            model = model.to(device)
            
            # Run inference
            y_test, probs, y_times, first_idh_times = \
                inference_model(model, test_dl, device)
            ensemble_predictions.append(probs)
            
        # Compute ensemble predictions
        final_probs = np.mean(ensemble_predictions, axis=0)
        final_preds = (final_probs >= 0.5).astype(bool)
        
        # Calculate metrics
        metrics = calculate_metrics_by_time_period(
            y_test, final_preds, final_probs, y_times, first_idh_times
        )
        
        # Save results
        save_results(metrics, args.output_dir, idh_type, args.pred_window)
        
        # Plot AUROC curve
        draw_auroc_curve(y_test, final_probs, args.output_dir, 
                        f"auroc_{idh_type}")


if __name__ == "__main__":
    main()
