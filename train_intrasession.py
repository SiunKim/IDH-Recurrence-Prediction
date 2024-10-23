# source: https://www.kaggle.com/code/scratchpad/notebook20da2ab86c
import os
import datetime

from collections import defaultdict

import torch
from torch import optim
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset

import numpy as np

from utils.train import (
    calculate_performances, 
    get_time_period_indexes_for_y_time_total,
    draw_epoch_loss_plots,
    draw_auroc_curve
    )

from model import LSTMClassifier

from train_settings.intrasession import TRS, PPS

from prep_train_data_intrasession import preprocess_train_data_total_intrasession
from prep_dataset_intrasession import (
    import_or_preprocess_train_valid_test_dataset_intrasession
    )



#GLOVAL VARIABLES IN TRAIN
MINIMUN_F1_FOR_BEST_MODEL = 0.05 #minimum f1 score for selecting the best model
RANDOM_SEED = 42
BATCH_SIZE_DEFAULT = 128
MILESTONES_MULTISTEPLR = [1/2, 2/3]
MILESTONES_MULTISTEPLR = [int(m*TRS.n_epochs) for m in MILESTONES_MULTISTEPLR]
GAMMA_MULTISTEPLR = 0.5



#CLASS
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

#FUNCTIONS
def set_all_random_seeds(random_seed=RANDOM_SEED):
    '''Set all reandom seeds'''
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)


def set_folder_and_fname_for_model_saving(break_if_exists=False,
                                          cross_validation_i=False,
                                          del_cov=False,
                                          ts_type='none',
                                          lambda_adv=False):
    '''Set folder name and fname for model saving'''
    #set folder_name and fname_model
    folder_name = \
        f"predwindow{PPS.predwindow}_obsperiod{PPS.obsperiod}_onlybeforefirstidh{PPS.only_before_first_idh}_mtd{PPS.min_time_del_for_prev_vital}"
    dir_model_saving = TRS.dir_model_saving
    if del_cov:
        dir_model_saving = dir_model_saving + f"/del_{del_cov}_tstype_{ts_type}"
    if TRS.adversarial:
        dir_model_saving += '_adversarial'
    fname_model = f"best_model_IDH_{PPS.idh_type}"
    #cross_validation_i
    if isinstance(cross_validation_i, int):
        fname_model += f"_cv{cross_validation_i}"
    if lambda_adv:
        fname_model += f"_labmdaadv{str(lambda_adv).replace('.', '')}"

    #check whether folder for model saving exits (if not, made it)
    try:
        os.listdir(dir_model_saving)
    except FileNotFoundError:
        os.mkdir(f"{dir_model_saving}")
    if folder_name in os.listdir(dir_model_saving):
        pass
    else:
        os.mkdir(f"{dir_model_saving}/{folder_name}")
    #update TRS.dir_model_saving
    dir_model_saving = f"{dir_model_saving}/{folder_name}"

    #check whether the best model trained on same settings already in the directory and give new ID
    for i in range(1, 100):
        if f'{fname_model}_{i}.pt' not in os.listdir(dir_model_saving):
            fname_model = f'{fname_model}_{i}'
            break

    #check break_if_exists
    if break_if_exists and int(fname_model.split('_')[-1])>1:
        return 'break_because_exists', dir_model_saving

    return fname_model, dir_model_saving


def set_folder_name_and_fname_for_train_valid_dataset(del_cov=False,
                                                      ts_type='none',
                                                      subgroup=False):
    '''Set folder name and fname for importing/saving train_valid_dataset'''
    folder_name = f"predwindow{PPS.predwindow}_obsperiod{PPS.obsperiod}_onlybeforefirstidh{PPS.only_before_first_idh}_mtd{PPS.min_time_del_for_prev_vital}"
    if subgroup:
        folder_name += f'_{subgroup}'
    dir_train_valid_dataset = PPS.dir_train_valid_dataset
    if del_cov:
        dir_train_valid_dataset = dir_train_valid_dataset +  f'/del_{del_cov}_tstype_{ts_type}'
    else:
        dir_train_valid_dataset = dir_train_valid_dataset +  '/complete_input'
    if PPS.adversarial:
        dir_train_valid_dataset += '_adversarial'

    if PPS.test_split_cv:
        fname_dataset = f"train_valid_dataset_idh_{PPS.idh_type}_5cv.p"
    else:
        fname_dataset = f"train_valid_dataset_idh_{PPS.idh_type}_5cv.p"

    #check whether folder for saving dataloaders exists (if not, made it)
    try:
        os.listdir(dir_train_valid_dataset)
    except FileNotFoundError:
        os.mkdir(dir_train_valid_dataset)
    if folder_name in os.listdir(dir_train_valid_dataset):
        pass
    else:
        os.mkdir(f"{dir_train_valid_dataset}/{folder_name}")
    #set dir_dataset as absolute path
    dir_dataset = f"{dir_train_valid_dataset}/{folder_name}"

    return fname_dataset, dir_dataset


def create_loaders(train_ds, valid_ds, bs=BATCH_SIZE_DEFAULT, jobs=0):
    '''Create dataset loaders'''
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


def add_training_setting_to_metrics_at_best_model(metrics_at_best_model):
    '''Add training setting information to metrics_at_best_model'''
    metrics_at_best_model['lr'] = TRS.lr
    metrics_at_best_model['batch_size'] = TRS.batch_size
    metrics_at_best_model['max_norm'] = TRS.max_norm
    metrics_at_best_model['weight_decay'] = TRS.weight_decay
    metrics_at_best_model['hidden_dim_time_inv'] = TRS.hidden_dim_time_inv
    metrics_at_best_model['hidden_dim_time_vars'] = TRS.hidden_dim_time_vars
    metrics_at_best_model['input_dim_time_inv'] = TRS.input_dim_time_inv
    metrics_at_best_model['input_dim_time_vars'] = TRS.input_dim_time_vars
    metrics_at_best_model['time_inv_inti'] = TRS.time_inv_inti
    metrics_at_best_model['layer_dim'] = TRS.layer_dim
    metrics_at_best_model['used_longest'] = PPS.used_longest
    metrics_at_best_model['test_split_method'] = PPS.test_split_method

    return metrics_at_best_model


def filter_by_time_condition(y_vld_total,
                             y_time_vld_total,
                             preds_total,
                             probs_total,
                             time_of_first_idh_vld_total,
                             condition):
    """
    Filter validation data based on a given time condition.

    Args:
        y_vld_total (list): Actual target values for the validation set.
        y_time_vld_total (list): Timestamps for the validation set.
        preds_total (list): Predicted target values for the validation set.
        probs_total (list): Predicted probabilities for the validation set.
        time_of_first_idh_vld_total (list): Timestamps of the first IDH event.
        condition (callable): Function to filter data based on time.

    Returns:
        tuple: Filtered lists of y_vld, y_time_vld, preds_vld, and probs_vld based on the specified condition.
    """
    y_vld_filtered = [y for y, y_time, time_of_first_idh
                        in zip(y_vld_total, y_time_vld_total, time_of_first_idh_vld_total)
                        if condition(y_time, time_of_first_idh)]
    y_time_vld_filtered = [y_time for _, y_time, time_of_first_idh
                            in zip(preds_total, y_time_vld_total, time_of_first_idh_vld_total)
                            if condition(y_time, time_of_first_idh)]
    preds_vld_filtered = [p for p, y_time, time_of_first_idh
                            in zip(preds_total, y_time_vld_total, time_of_first_idh_vld_total)
                            if condition(y_time, time_of_first_idh)]
    probs_vld_filtered = [p for p, y_time, time_of_first_idh
                            in zip(probs_total, y_time_vld_total, time_of_first_idh_vld_total) if condition(y_time, time_of_first_idh)]

    return y_vld_filtered, y_time_vld_filtered, preds_vld_filtered, probs_vld_filtered

def get_y_probs_tp_for_before_after_first_idh(y_vld_total,
                                              preds_total,
                                              probs_total,
                                              y_time_vld_total,
                                              time_of_first_idh_vld_total,
                                              before_after_first):
    """
    Filter validation data based on the timing of the first IDH event.

    Args:
        y_vld_total (list): Actual target values for the validation set.
        preds_total (list): Predicted target values for the validation set.
        probs_total (list): Predicted probabilities for the validation set.
        y_time_vld_total (list): Timestamps for the validation set.
        time_of_first_idh_vld_total (list): Timestamps of the first IDH event.
        before_after_first (str): Period to filter by ('before_first_idh',
        'after_first_idh', 'idh_reocc').

    Returns:
        tuple: Filtered lists of y_vld, y_time_vld, preds_vld,
        and probs_vld based on the specified period.
    """
    def before_first_idh_condition(y_time, time_of_first_idh):
        return True if time_of_first_idh in [0, False] \
            else y_time <= time_of_first_idh
    def after_first_idh_condition(y_time, time_of_first_idh):
        return False if time_of_first_idh in [0, False] \
            else y_time > time_of_first_idh
    def idh_reocc_condition(y_time, time_of_first_idh):
        return False if time_of_first_idh in [0, False] \
            else y_time > time_of_first_idh + 25 * 60
    if before_after_first == 'before_first_idh':
        condition = before_first_idh_condition
    elif before_after_first == 'after_first_idh':
        condition = after_first_idh_condition
    else:  # idh_reocc
        condition = idh_reocc_condition

    y_vld_filtered, y_time_vld_filtered, preds_vld_filtered, probs_vld_filtered = \
        filter_by_time_condition(y_vld_total,
                                 y_time_vld_total,
                                 preds_total,
                                 probs_total,
                                 time_of_first_idh_vld_total,
                                 condition)

    return y_vld_filtered, y_time_vld_filtered, preds_vld_filtered, probs_vld_filtered


def calculate_performances_by_time_periods(y_vld_total,
                                           preds_total,
                                           probs_total,
                                           y_time_vld_total,
                                           time_of_first_idh_vld_total,
                                           metrics_at_best_model,
                                           calculate_function=False):
    """
    Calculate performance metrics for different time periods based on validation data.

    Args:
        y_vld_total (list): Actual target values for the validation set.
        preds_total (list): Predicted target values for the validation set.
        probs_total (list): Predicted probabilities for the validation set.
        y_time_vld_total (list): Timestamps for the validation set.
        time_of_first_idh_vld_total (list): Timestamps of the first IDH event.
        metrics_at_best_model (dict): Dictionary to store performance metrics.
        calculate_function (callable, optional): Custom function for calculating metrics. Defaults to False.

    Returns:
        dict: Updated metrics_at_best_model with performance metrics for each time period.

    This function processes data to compute metrics for early/late periods, session-based periods (before/after first IDH, IDH reoccurrence), and custom time periods.
    """
    def update_metrics_from_y_probs_tp(y_vld_tp, preds_total, probs_tp, metrics_key):
        """
        Filter performance metrics for a specific time period.

        Args:
            y_vld_tp (list): Actual target values for the time period.
            preds_total (list): Predicted target values for the entire validation set.
            probs_tp (list): Predicted probabilities for the time period.
            metrics_key (str): Key to identify the metrics for the time period.

        Returns:
            dict: Updated metrics_at_best_model with the new performance metrics.
        """
        if y_vld_tp:
            if calculate_function is False:
                perf_metric_tp = calculate_performances(y_vld_tp, probs_tp)
            else:
                perf_metric_tp = calculate_function(y_vld_tp, preds_total, probs_tp)
            perf_metric_tp['sample_n'] = len(y_vld_tp)
            perf_metric_tp['sample_positive_n'] = sum(y_vld_tp)
            metrics_at_best_model['by_time_periods'][metrics_key] = perf_metric_tp
        else:
            metrics_at_best_model['by_time_periods'][metrics_key] = defaultdict(float)
            metrics_at_best_model['by_time_periods'][metrics_key]['sample_n'] = 0
            metrics_at_best_model['by_time_periods'][metrics_key]['sample_positive_n'] = 0
        return metrics_at_best_model

    def process_time_period(tp_label, condition):
        """
        Process and update metrics for a specified time period based on a condition.

        Args:
            tp_label (str): Label for the time period.
            condition (callable): Function to filter data based on time.

        Returns:
            dict: Updated metrics_at_best_model with metrics for the specified time period.
        """
        y_vld_tp = [y for y, y_time in zip(y_vld_total, y_time_vld_total) if condition(y_time)]
        preds_tp = [p for p, y_time in zip(preds_total, y_time_vld_total) if condition(y_time)]
        probs_tp = [p for p, y_time in zip(probs_total, y_time_vld_total) if condition(y_time)]
        metrics_at_best_model = update_metrics_from_y_probs_tp(y_vld_tp,
                                                               preds_tp,
                                                               probs_tp,
                                                               tp_label)
        if PPS.only_before_first_idh in [False, 'Reocc']:
            for period, (y_vld_period, y_time_vld_period, preds_period, probs_period) \
                in zip(periods,
                       zip([y_vld_before, y_vld_after, y_vld_reocc],
                           [y_time_vld_before, y_time_vld_after, y_time_vld_reocc],
                           [preds_vld_before, preds_vld_after, preds_vld_reocc], 
                           [probs_vld_before, probs_vld_after, probs_vld_reocc])):
                y_vld_tp_period = [y for y, y_time in zip(y_vld_period, y_time_vld_period)
                                        if condition(y_time)]
                preds_tp_period = [p for p, y_time in zip(preds_period, y_time_vld_period)
                                        if condition(y_time)]
                probs_tp_period = [p for p, y_time in zip(probs_period, y_time_vld_period)
                                        if condition(y_time)]
                metrics_key = f'{tp_label}_{period}'
                metrics_at_best_model = update_metrics_from_y_probs_tp(y_vld_tp_period,
                                                                       preds_tp_period,
                                                                       probs_tp_period,
                                                                       metrics_key)
        return metrics_at_best_model

    metrics_at_best_model['by_time_periods'] = {}
    periods = ['before_first_idh', 'after_first_idh', 'idh_reocc']
    #total_session
    if PPS.only_before_first_idh in [False, 'Reocc']:
        for period in periods:
            before_after_first = period
            y_vld, y_time_vld, preds_vld, probs_vld = \
                get_y_probs_tp_for_before_after_first_idh(y_vld_total,
                                                          preds_total,
                                                          probs_total,
                                                          y_time_vld_total,
                                                          time_of_first_idh_vld_total,
                                                          before_after_first)
            metrics_key = f'totalsession_{before_after_first}'
            metrics_at_best_model = update_metrics_from_y_probs_tp(y_vld,
                                                                   preds_vld,
                                                                   probs_vld,
                                                                   metrics_key)
            #save y-time-pred-probs / before-after-reocc
            if before_after_first=='before_first_idh':
                y_vld_before, y_time_vld_before, preds_vld_before, probs_vld_before =\
                    y_vld, y_time_vld, preds_vld, probs_vld
            if before_after_first=='after_first_idh':
                y_vld_after, y_time_vld_after, preds_vld_after, probs_vld_after =\
                    y_vld, y_time_vld, preds_vld, probs_vld
            else: #idh_reocc
                y_vld_reocc, y_time_vld_reocc, preds_vld_reocc, probs_vld_reocc =\
                    y_vld, y_time_vld, preds_vld, probs_vld

    #early-late
    metrics_at_best_model = process_time_period('early', lambda y_time: y_time < 60 * 60)
    metrics_at_best_model = process_time_period('late', lambda y_time: y_time >= 60 * 60)

    #for time-period
    for tp_idx, (start_time, end_time) in enumerate(TRS.time_periods):
        if start_time == 0:
            continue
        tp_str = f"{start_time}-{end_time}"
        time_period_indexes = get_time_period_indexes_for_y_time_total(y_time_vld_total,
                                                                       TRS.time_periods)
        y_vld_tp = [y for y, y_tp_idx in zip(y_vld_total, time_period_indexes)
                        if y_tp_idx == tp_idx]
        preds_tp = [p for p, y_tp_idx in zip(preds_total, time_period_indexes)
                        if y_tp_idx == tp_idx]
        probs_tp = [p for p, y_tp_idx in zip(probs_total, time_period_indexes)
                        if y_tp_idx == tp_idx]
        metrics_at_best_model = update_metrics_from_y_probs_tp(y_vld_tp, preds_tp, probs_tp,
                                                               tp_str)
        if PPS.only_before_first_idh in [False, 'Reocc']:
            for period, (y_vld_period, y_time_vld_period, preds_period, probs_period) in zip(
                    periods,
                    zip([y_vld_before, y_vld_after, y_vld_reocc],
                        [y_time_vld_before, y_time_vld_after, y_time_vld_reocc],
                        [preds_vld_before, preds_vld_after, preds_vld_reocc],
                        [probs_vld_before, probs_vld_after, probs_vld_reocc])):
                time_period_indexes_period = \
                    get_time_period_indexes_for_y_time_total(y_time_vld_period,
                                                             TRS.time_periods)
                y_vld_tp_period = [y for y, y_tp_idx in zip(y_vld_period,
                                                            time_period_indexes_period)
                                    if y_tp_idx == tp_idx]
                preds_tp_period = [p for p, y_tp_idx in zip(preds_period,
                                                            time_period_indexes_period)
                                    if y_tp_idx == tp_idx]
                probs_tp_period = [p for p, y_tp_idx in zip(probs_period,
                                                            time_period_indexes_period)
                                    if y_tp_idx == tp_idx]
                metrics_at_best_model = update_metrics_from_y_probs_tp(y_vld_tp_period,
                                                                       preds_tp_period,
                                                                       probs_tp_period,
                                                                       f'{tp_str}_{period}')

    return metrics_at_best_model


def eval_model(model, fname_model, valid_dl, criterion,
               epoch, best_score, metrics_at_best_model, dir_model_saving):
    '''Evaluate the model'''
    #evluating model
    model.eval()
    #set torch objects for saving y, probs, y_time
    y_vld_total = torch.empty(0).cuda()
    probs_total = torch.empty(0).cuda()
    y_time_vld_total = torch.empty(0).cuda()
    time_of_first_idh_vld_total = torch.empty(0).cuda()
    #set val_loss_total
    val_loss_total = 0.0
    #measure validation performance in valid_dl
    for x_time_inv_vld, x_time_vars_vld, y_vld, y_time_vld, time_of_first_idh_vld in valid_dl:
        #torch to cuda
        x_time_inv_vld = x_time_inv_vld.cuda()
        x_time_vars_vld = x_time_vars_vld.cuda()
        y_vld = y_vld.cuda()
        y_time_vld = y_time_vld.cuda()
        time_of_first_idh_vld = time_of_first_idh_vld.cuda()
        #dtype to float32
        x_time_inv_vld = x_time_inv_vld.to(torch.float32)
        x_time_vars_vld = x_time_vars_vld.to(torch.float32)

        #get model output and validation loss (add to val_loss_total)
        out_vld, _ = model(x_time_inv_vld, x_time_vars_vld)
        val_loss = criterion(out_vld, y_vld)
        val_loss_total += val_loss
        probs = F.softmax(out_vld, dim=1)
        #append to y_vld/probs/y_time_vld_total
        y_vld_total = torch.cat((y_vld_total, y_vld), dim=0)
        probs_total = torch.cat((probs_total, probs), dim=0)
        y_time_vld_total = torch.cat((y_time_vld_total, y_time_vld), dim=0)
        time_of_first_idh_vld_total = torch.cat((time_of_first_idh_vld_total,
                                                 time_of_first_idh_vld), dim=0)

    #calculate evaluation metrics
    #tensors to cpu
    y_vld_total = y_vld_total.cpu().numpy()
    probs_total = probs_total.detach().cpu().numpy()
    probs_total = probs_total[:, 1]
    y_time_vld_total = y_time_vld_total.detach().cpu().numpy()
    time_of_first_idh_vld_total = time_of_first_idh_vld_total.detach().cpu().numpy()
    #calculate evaluation metrics - zerio_division warning off
    performance_metrics = calculate_performances(y_vld_total, probs_total)

    #select best model by TRS.model_selection_criterion
    score = performance_metrics[TRS.model_selection_criterion]
    f1 = performance_metrics['f1']
    if best_score<score and MINIMUN_F1_FOR_BEST_MODEL<f1:
        #update best_score
        best_score = score
        #print-out and save performance metrics in metrics_at_best_model
        print(f'Best model ever at epoch {epoch}!')
        for p_name, p_value in performance_metrics.items():
            if TRS.print_best_model_result:
                print(f"{p_name[0].upper() + p_name[1:]}: {p_value:.4f}")
            metrics_at_best_model[p_name] = p_value
        #save epoch of best model in metrics_at_best_model
        metrics_at_best_model['epoch'] = epoch

        #calcualte performances by time_periods and update them in metrics_at_best_model
        metrics_at_best_model = \
            calculate_performances_by_time_periods(y_vld_total, probs_total, probs_total,
                                                   y_time_vld_total,
                                                   time_of_first_idh_vld_total,
                                                   metrics_at_best_model)
        #darw auroc curve
        if TRS.print_graph:
            draw_auroc_curve(y_vld_total, probs_total, dir_model_saving, fname_model)

    return val_loss_total, best_score, metrics_at_best_model


def training_loop(model, optimizer, criterion, scheduler,
                  useearlystop, patience,
                  train_dl, valid_dl,
                  dir_model_saving,
                  fname_model,
                  save_torch=True):
    '''Training loop for training'''         
    print(f'Start model training - {fname_model}')
    #set variables for recording training loop
    best_score = 0.0
    metrics_at_best_model = {}
    #set metrics_at_best_model
    metrics_at_best_model = add_training_setting_to_metrics_at_best_model(metrics_at_best_model)
    #set lists for train/valid loss
    train_loss_total_by_epoch = []
    valid_loss_total_by_epoch = []
    #set early stopper
    if useearlystop:
        early_stopper = EarlyStopper(patience=patience, min_delta=0)
    #training loop
    for epoch in range(1, TRS.n_epochs + 1):
        train_loss_total = 0.0
        for _, (x_time_inv_batch, x_time_vars_batch, y_batch, _, _) in enumerate(train_dl):
            #training model
            model.train()
            #torch to cuda
            x_time_inv_batch = x_time_inv_batch.cuda()
            x_time_vars_batch = x_time_vars_batch.cuda()
            y_batch = y_batch.cuda()
            #dtype to float32
            x_time_inv_batch = x_time_inv_batch.to(torch.float32)
            x_time_vars_batch = x_time_vars_batch.to(torch.float32)

            #get output and train
            optimizer.zero_grad()
            out, _ = model(x_time_inv_batch, x_time_vars_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            if TRS.max_norm!='not_use':
                utils.clip_grad_norm_(model.parameters(), TRS.max_norm)
            optimizer.step()
            #add up train_loss_total
            train_loss_total += loss
        #step by epoch
        scheduler.step()

        #evalutat model
        val_loss_total, best_score, metrics_at_best_model = \
            eval_model(model, fname_model, valid_dl, criterion,
                       epoch, best_score, metrics_at_best_model, dir_model_saving)

        #save train/val_loss
        train_loss_total_by_epoch.append(float(train_loss_total)/len(valid_dl))
        valid_loss_total_by_epoch.append(float(val_loss_total)/len(valid_dl))
        #print losses every N epochs
        if epoch%TRS.print_epoch==0:
            print(f'Epoch: {epoch}')
            print(f'Train loss: {float(train_loss_total)/len(train_dl)}')
            print(f'Validation loss: {float(val_loss_total)/len(train_dl)}')

        #early stopping criterion
        if early_stopper:
            if early_stopper.early_stop(val_loss_total):
                break

    #save model and training settings
    if save_torch:
        torch.save(model, f"{dir_model_saving}/{fname_model}.pt")
        print('Restoring the best model weights as pt file!')
    with open(f"{dir_model_saving}/{fname_model}_settings.txt", "w", encoding='utf-8-sig') as f:
        f.write(str(metrics_at_best_model))
    print('The whole training process is finished!')

    #print-out epoch-loss plots
    if TRS.print_graph:
        draw_epoch_loss_plots(train_loss_total_by_epoch, valid_loss_total_by_epoch,
                            dir_model_saving, fname_model)


def main(train=True,
         break_if_exists=False,
         cross_validation_i=False,
         del_cov=False,
         ts_type='vital_ts',
         subgroup=False,
         fast_preprocessing=False,
         return_train_data=False):
    '''Main function'''
    #set random seeds for reproducibility
    set_all_random_seeds(random_seed=RANDOM_SEED)

    #set foler_name and fname_model for model saving
    fname_model, dir_model_saving = \
        set_folder_and_fname_for_model_saving(break_if_exists=break_if_exists,
                                              cross_validation_i=cross_validation_i,
                                              del_cov=del_cov,
                                              ts_type=ts_type)
    #set foler_name and fname_model for saving datasetstrain_valid_dataset_intrasession
    fname_dataset, dir_dataset = \
        set_folder_name_and_fname_for_train_valid_dataset(del_cov=del_cov,
                                                          ts_type=ts_type,
                                                          subgroup=subgroup)

    #break_if_exists
    if fname_model!='break_because_exists':
        #import train_ds, valid_da from pickle file (if no pickle file exist, perform preprocessing)
        if PPS.test_split_cv:
            assert isinstance(cross_validation_i, int), \
                "Under a cross validation setting, cross_validation_i must be specified!"
            print(fname_dataset)
            print(dir_dataset)
            if fast_preprocessing:
                (train_ds1, train_ds2, train_ds3, train_ds4, train_ds5,
                _, input_dim_time_inv, input_dim_time_vars, return_train_data)  = \
                    import_or_preprocess_train_valid_test_dataset_intrasession(
                        fname_dataset,
                        dir_dataset,
                        subgroup=subgroup,
                        fast_preprocessing=True,
                        return_train_data=return_train_data
                        )
            else:
                (train_ds1, train_ds2, train_ds3, train_ds4, train_ds5,
                _, input_dim_time_inv, input_dim_time_vars) = \
                    import_or_preprocess_train_valid_test_dataset_intrasession(
                        fname_dataset,
                        dir_dataset,
                        subgroup=subgroup)
            #set train_ds and valid_ds by cross_validation_i
            train_ds_list = [train_ds1, train_ds2, train_ds3, train_ds4, train_ds5]
            train_ds = \
                ConcatDataset(train_ds_list[:cross_validation_i] +
                              train_ds_list[cross_validation_i+1:])
            valid_ds = train_ds_list[cross_validation_i]
        else:
            train_ds, valid_ds, _, input_dim_time_inv, input_dim_time_vars = \
                import_or_preprocess_train_valid_test_dataset_intrasession(
                    fname_dataset,
                    dir_dataset,
                    subgroup=subgroup
                    )

        if train:
            #set train/valid dataloader
            print(f'Creating data loaders with batch size: {TRS.batch_size}')
            train_dl, valid_dl = create_loaders(train_ds, valid_ds, TRS.batch_size)

            #Model training
            #re-assign the input dimensions for time-variant and time-invariants section
            TRS.input_dim_time_inv = input_dim_time_inv
            TRS.input_dim_time_vars = input_dim_time_vars
            #set model and optimizer
            model = LSTMClassifier(TRS.input_dim_time_inv,
                                   TRS.input_dim_time_vars,
                                   TRS.hidden_dim_time_inv,
                                   TRS.hidden_dim_time_vars,
                                   TRS.embedding_dim,
                                   TRS.layer_dim,
                                   TRS.output_dim)
            model.cuda()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=TRS.lr)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=MILESTONES_MULTISTEPLR,
                                                    gamma=GAMMA_MULTISTEPLR)

            #perform training
            training_loop(model, optimizer, criterion, scheduler,
                          TRS.useearlystop, TRS.patience,
                          train_dl, valid_dl, dir_model_saving, fname_model)

    if fast_preprocessing:
        return return_train_data

# main(train=False,
#      break_if_exists=False,
#      cross_validation_i=0,
#      del_cov=False)

# PPS.predwindow = TRS.predwindow = 30
# PPS.obsperiod = TRS.obsperiod = 30
# PPS.only_before_first_idh = TRS.only_before_first_idh = False

# for idh_type in PPS.idh_types:
#     PPS.idh_type = TRS.idh_type = idh_type
#     main(train=False,
#          break_if_exists=False,
#          cross_validation_i=0)



# for idh_type in ['sbp90', 'sbp100']:
#     TRS.idh_type = idh_type
#     main()
