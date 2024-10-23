"""
Train, validation, and test dataset splitting module.

This module handles the splitting of dialysis session data into train, validation,
and test sets while maintaining balanced IDH distributions across splits.
Supports patient-based and calendar-based splitting strategies.

Key Functions:
- set_ptid_trn_vld: Main entry point for train-validation-test splitting
- get_balanced_split: Find splits with balanced IDH distributions
- validate_split_balance: Check IDH distribution balance across splits
"""

import os
import pickle
from typing import List, Tuple, Dict, Union, Set
from collections import Counter
import random

from train_settings.intrasession import PPS


class SplitConfig:
    """Configuration for dataset splitting."""
    
    # IDH type-specific balance thresholds
    idh_thresholds = {
        'intra': {
            'sbp90': 1.0,
            'sbp100': 1.0,
            'kdoqi': 3.0,
            'sbpd20': 3.0,
            'sbpd30': 3.0,
            'sbp90_sbpd20': 0.5,
            'sbp90_sbpd30': 0.5,
            'hanbilee': 0.5
        },
        'pre': {
            'sbp90': 2.0,
            'sbp100': 2.0,
            'kdoqi': 4.0,
            'sbpd20': 4.0,
            'sbpd30': 4.0,
            'sbp90_sbpd20': 1.0,
            'sbp90_sbpd30': 1.0,
            'hanbilee': 1.0
        }
    }
    
    # IDH types to check for balance
    idh_types = [
        'kdoqi', 'sbpd20', 'sbp100', 'sbp90'
    ]
    
    # Number of random seeds to try
    max_seed_attempts = 1000


def get_ptid_split(train_data: List[dict], 
                   random_seed: int) -> Tuple[List[str], List[str], List[str]]:
    """
    Split patient IDs into train, validation and test sets.
    
    Args:
        train_data: List of training data dictionaries
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ptids, valid_ptids, test_ptids)
    """
    random.seed(random_seed)
    
    # Get unique patient IDs
    ptids = list(set(td['ptid'] for td in train_data))
    random.shuffle(ptids)
    
    # Calculate split indices
    total_fold = PPS.train_fold + PPS.valid_fold + PPS.test_fold
    train_idx = int(len(ptids) * PPS.train_fold / total_fold)
    valid_idx = int(len(ptids) * (PPS.train_fold + PPS.valid_fold) / total_fold)
    
    # Split patient IDs
    train_ptids = ptids[:train_idx]
    valid_ptids = ptids[train_idx:valid_idx]
    test_ptids = ptids[valid_idx:]
    
    return train_ptids, valid_ptids, test_ptids


def get_split_data(train_data: List[dict], 
                   ptids: List[str],
                   idh_type: str) -> Tuple[List[dict], int, float]:
    """
    Get split dataset and calculate IDH statistics.
    
    Args:
        train_data: List of training data dictionaries
        ptids: List of patient IDs for this split
        idh_type: Type of IDH to analyze
        
    Returns:
        Tuple of (split_data, num_idh, idh_proportion)
    """
    split_data = [td for td in train_data if td['ptid'] in ptids]
    num_idh = sum(td[f"idh_{idh_type}"] for td in split_data)
    idh_prop = (num_idh / len(split_data)) * 100 if split_data else 0
    
    return split_data, num_idh, idh_prop


def validate_split_balance(train_data: List[dict],
                          train_ptids: List[str],
                          valid_ptids: List[str], 
                          test_ptids: List[str],
                          idh_type: str,
                          threshold: float) -> bool:
    """
    Validate that IDH distributions are balanced across splits.
    
    Args:
        train_data: List of training data dictionaries
        train_ptids: List of patient IDs for training set
        valid_ptids: List of patient IDs for validation set
        test_ptids: List of patient IDs for test set
        idh_type: Type of IDH to analyze
        threshold: Maximum allowed difference in IDH proportions
        
    Returns:
        bool indicating if split is balanced
    """
    # Get IDH proportions for each split
    _, _, train_prop = get_split_data(train_data, train_ptids, idh_type)
    _, _, valid_prop = get_split_data(train_data, valid_ptids, idh_type)
    _, _, test_prop = get_split_data(train_data, test_ptids, idh_type)
    
    # Check if differences are within threshold
    return (abs(train_prop - valid_prop) <= threshold and
            abs(valid_prop - test_prop) <= threshold and
            abs(test_prop - train_prop) <= threshold)


def print_split_stats(train_data: List[dict],
                      train_ptids: List[str],
                      valid_ptids: List[str],
                      test_ptids: List[str],
                      idh_type: str,
                      random_seed: int) -> None:
    """
    Print statistics about the data split.
    
    Args:
        train_data: List of training data dictionaries
        train_ptids: List of patient IDs for training set
        valid_ptids: List of patient IDs for validation set
        test_ptids: List of patient IDs for test set
        idh_type: Type of IDH being analyzed
        random_seed: Random seed used for split
    """
    train_data_split, train_idh, train_prop = get_split_data(train_data, train_ptids, idh_type)
    valid_data_split, valid_idh, valid_prop = get_split_data(train_data, valid_ptids, idh_type)
    test_data_split, test_idh, test_prop = get_split_data(train_data, test_ptids, idh_type)
    
    print(f'\nRandom seed: {random_seed}')
    print(f'Number of total samples in train: {len(train_data_split)}')
    print(f'Number of total samples in valid: {len(valid_data_split)}')
    print(f'Number of total samples in test: {len(test_data_split)}')
    print(f'Number of IDH-True samples in train: {train_idh} ({train_prop:.1f}%)')
    print(f'Number of IDH-True samples in valid: {valid_idh} ({valid_prop:.1f}%)')
    print(f'Number of IDH-True samples in test: {test_idh} ({test_prop:.1f}%)')


def get_balanced_split(train_data: List[dict],
                       idh_type: str,
                       start_seed: int = 42,
                       task_type: str = 'intra') -> Tuple[List[str], List[str], List[str]]:
    """
    Find a balanced split of the data for the given IDH type.
    
    Args:
        train_data: List of training data dictionaries
        idh_type: Type of IDH to analyze
        start_seed: Initial random seed to try
        task_type: Type of task ('intra' or 'pre')
        
    Returns:
        Tuple of (train_ptids, valid_ptids, test_ptids) for balanced split
    """
    threshold = SplitConfig.idh_thresholds[task_type][idh_type]
    
    print(f"\nFinding balanced split for IDH type: {idh_type}")
    
    for seed in range(start_seed, start_seed + SplitConfig.max_seed_attempts):
        train_ptids, valid_ptids, test_ptids = get_ptid_split(train_data, seed)
        
        if validate_split_balance(train_data, train_ptids, valid_ptids, test_ptids,
                                idh_type, threshold):
            print_split_stats(train_data, train_ptids, valid_ptids, test_ptids,
                            idh_type, seed)
            return train_ptids, valid_ptids, test_ptids
            
    raise ValueError(f"Could not find balanced split after {SplitConfig.max_seed_attempts} attempts")


def save_split_results(train_ptids: List[str],
                       valid_ptids: List[str],
                       test_ptids: List[str],
                       save_dir: str) -> None:
    """
    Save the split results to pickle files.
    
    Args:
        train_ptids: List of patient IDs for training set
        valid_ptids: List of patient IDs for validation set
        test_ptids: List of patient IDs for test set
        save_dir: Directory to save the split files
    """
    with open(f'{save_dir}/ptid_trn.p', 'wb') as f:
        pickle.dump(train_ptids, f)
    with open(f'{save_dir}/ptid_vld.p', 'wb') as f:
        pickle.dump(valid_ptids, f)
    with open(f'{save_dir}/ptid_tst.p', 'wb') as f:
        pickle.dump(test_ptids, f)


def set_ptid_trn_vld(train_data: List[dict], 
                      save_dir: str,
                      task_type: str = 'intra') -> None:
    """
    Main entry point for splitting data into train, validation and test sets.
    
    Finds splits that maintain balanced IDH distributions across all IDH types
    and saves the results.
    
    Args:
        train_data: List of training data dictionaries
        save_dir: Directory to save split results
        task_type: Type of task ('intra' or 'pre')
    """
    train_ptids, valid_ptids, test_ptids = None, None, None
    
    # Try finding balanced splits for each IDH type
    for idh_type in SplitConfig.idh_types:
        if train_ptids is None:
            # First IDH type - get initial split
            train_ptids, valid_ptids, test_ptids = get_balanced_split(
                train_data, idh_type, 42, task_type
            )
        else:
            # Verify split is balanced for this IDH type
            threshold = SplitConfig.idh_thresholds[task_type][idh_type]
            if not validate_split_balance(train_data, train_ptids, valid_ptids, test_ptids,
                                        idh_type, threshold):
                raise ValueError(f"Split not balanced for IDH type: {idh_type}")
            print_split_stats(train_data, train_ptids, valid_ptids, test_ptids,
                            idh_type, 42)

    # Save final split results
    save_split_results(train_ptids, valid_ptids, test_ptids, save_dir)
