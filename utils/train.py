"""
Model training utilities module.

This module provides utility functions for model training and evaluation, including:
- Performance metric calculation (accuracy, precision, recall, etc.)
- Confidence interval estimation using bootstrap resampling
- Visualization functions for training curves and ROC curves
- Time period-based evaluation helpers

Key Functions:
- calculate_metrics: Calculate core classification metrics
- compute_all_metrics_with_ci: Calculate metrics with confidence intervals
- get_time_period_indexes: Get time period indexes for sequential data
- draw_epoch_loss_plots: Plot training/validation loss curves
- draw_auroc_curve: Plot ROC curves with AUC scores
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_curve, roc_auc_score,
    precision_recall_curve, auc, matthews_corrcoef,
    confusion_matrix
)
import matplotlib.pyplot as plt


def calculate_metrics(y_true: np.ndarray,
                     y_pred: np.ndarray,
                     y_prob: np.ndarray) -> dict:
    """
    Calculate various performance metrics for binary classification.

    Args:
        y_true: True labels (ground truth)
        y_pred: Predicted labels 
        y_prob: Prediction probabilities

    Returns:
        Dictionary containing calculated metrics:
        - accuracy: Overall prediction accuracy
        - precision: Positive predictive value
        - specificity: True negative rate
        - recall: True positive rate
        - f1: F1 score (harmonic mean of precision and recall)
        - auroc: Area under ROC curve
        - auprc: Area under precision-recall curve
        - mcc: Matthews correlation coefficient
        - npv: Negative predictive value
    """
    try:
        # Calculate confusion matrix metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        npv = tn / (tn + fn) if (tn + fn) != 0 else 0
    except ValueError:
        precision = specificity = npv = 0.0

    # Calculate other core metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision,
        'specificity': specificity, 
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'npv': npv
    }

    # Calculate curve-based metrics
    try:
        metrics['auroc'] = roc_auc_score(y_true, y_prob)
        precision_pr, recall_pr, _ = precision_recall_curve(y_true, y_prob)
        metrics['auprc'] = auc(recall_pr, precision_pr)
    except ValueError:  # Only one class present
        metrics['auroc'] = metrics['auprc'] = 1.0

    # Calculate MCC
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    return metrics


def compute_bootstrap_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray, 
                            y_prob: np.ndarray,
                            n_bootstraps: int = 1000,
                            n_jobs: int = 12) -> dict:
    """
    Compute metrics with confidence intervals using bootstrap resampling.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        n_bootstraps: Number of bootstrap iterations
        n_jobs: Number of parallel jobs

    Returns:
        Dictionary containing metrics with confidence intervals
    """
    def bootstrap_sample(_):
        # Resample with replacement
        indices = np.random.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) > 1:
            resampled_metrics = calculate_metrics(
                y_true[indices],
                y_pred[indices],
                y_prob[indices]
            )
            return resampled_metrics
        return None

    # Compute bootstrap samples in parallel
    bootstrap_results = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_sample)(_) for _ in range(n_bootstraps)
    )
    bootstrap_results = [r for r in bootstrap_results if r is not None]

    # Calculate base metrics
    base_metrics = calculate_metrics(y_true, y_pred, y_prob)

    # Calculate confidence intervals
    results = {}
    for metric in base_metrics:
        values = [r[metric] for r in bootstrap_results]
        results[metric] = base_metrics[metric]
        results[f'{metric}_95ci_lower'] = np.percentile(values, 2.5)
        results[f'{metric}_95ci_upper'] = np.percentile(values, 97.5)

    # Add sample counts
    results['n_sample'] = len(y_true)
    results['n_sample_positive'] = sum(y_true)

    return results


def get_time_period_indexes(times: np.ndarray, 
                           periods: list) -> list:
    """
    Assign time period indices to time series data points.

    Args:
        times: Array of timestamps
        periods: List of (start_time, end_time) tuples defining periods

    Returns:
        List of period indices for each timestamp
    """
    indexes = []
    for time in times:
        if time == 0:
            indexes.append(-1)
            continue
            
        index_found = False
        for idx, (start, end) in enumerate(periods):
            if start * 60 <= time < end * 60:
                indexes.append(idx)
                index_found = True
                break
                
        if not index_found:
            raise ValueError(f"No matching period found for time {time}")

    return indexes


def draw_epoch_loss_plots(train_losses: list,
                         valid_losses: list,
                         save_dir: str,
                         model_name: str,
                         adv_losses: list = None):
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch
        valid_losses: List of validation losses per epoch 
        save_dir: Directory to save plot
        model_name: Name for plot file
        adv_losses: Optional list of adversarial losses
    """
    plt.figure(figsize=(10, 6))
    plt.title('Training and Validation Loss')
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    if adv_losses is not None:
        plt.plot(epochs, adv_losses, label='Adversarial Loss')
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{save_dir}/{model_name}_loss.png', dpi=300)
    plt.close()


def draw_auroc_curve(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    save_dir: str,
                    model_name: str):
    """
    Plot ROC curve with AUC score.

    Args:
        y_true: True labels
        y_prob: Prediction probabilities
        save_dir: Directory to save plot
        model_name: Name for plot file
    """
    plt.figure(figsize=(8, 8))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)

    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], '--', linewidth=2)
    plt.title(f'ROC Curve (AUC = {auroc:.3f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(f'{save_dir}/{model_name}_roc.png', dpi=300)
    plt.close()
