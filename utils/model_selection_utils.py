#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, Union, List
import numpy as np
import torch as t
from tqdm import tqdm, trange
from copy import deepcopy
import pandas as pd
from sklearn.model_selection import ParameterGrid

from loaders.loader import Loader
from datasets.dataset import Dataset, Entity
from algorithm.base_model import PyMADModel
from model_selection.inject_anomalies import InjectAnomalies
from utils.utils import de_unfold
from model_trainer.hyperparameter_grids import *
from model_selection.anomaly_parameters import ANOMALY_PARAM_GRID


######################################################
# Functions to predict Y_hat given a model
######################################################


def predict(batch: dict, model_name: str,
            model: PyMADModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given a model and an input batch (Y), use the model to predict Y_hat.

    Parameters
    ----------
    batch: dict
        Input batch 
    model_name: str
        Name of the model to use for prediction
    """
    _VALID_MODEL_NAMES = ['RNN', 'DGHL', 'LSTMVAE', 'MD', 'RM',
                          'NN']  # TODO: Should be stored somewhere centrally

    model_type = model_name.split('_')[0]

    if model_type == 'RNN':
        return _predict_rnn(batch, model)
    elif model_type == 'DGHL':
        return _predict_dghl(batch, model)
    elif model_type == 'NN':
        return _predict_nn(batch, model)
    elif model_type == 'MD':
        return _predict_md(batch, model)
    elif model_type == 'RM':
        return _predict_rm(batch, model)
    elif model_type == 'LSTMVAE':
        return _predict_lstmvae(batch, model)
    elif model_type == 'LOF':
        return _predict_lof(batch, model)
    elif model_type == 'KDE':
        return _predict_kde(batch, model)
    elif model_type == 'ABOD':
        return _predict_abod(batch, model)
    elif model_type == 'CBLOF':
        return _predict_cblof(batch, model)
    elif model_type == 'COF':
        return _predict_cof(batch, model)
    elif model_type == 'SOS':
        return _predict_sos(batch, model)
    else:
        raise AttributeError(
            f'Model type must be one of {_VALID_MODEL_NAMES}, but {model_type} was passed!'
        )


######################################################
# Functions to compute observations necessary to compute
# ranking metrics
######################################################


def evaluate_model(data: Union[Dataset, Entity],
                   model: PyMADModel,
                   model_name: str,
                   padding_type: str = 'right',
                   eval_batch_size: int = 128) -> dict:
    """Compute observations necessary to evaluate a model on a given dataset.

    Description
    -----------
    This function computes predicted anomaly scores of an entity (entity scores), 
    Y_hat (the predicted values of the entity) and Y_sigma. Y_sigma is NaN in most
    cases except in the case of LSTM-VAE. These observations are useful for two
    classes of metrics, names forecasting error and cetrality. 

    Parameters
    ----------
    data:Union[Dataset, Entity]
        Dataset to evaluate the model on.   
    
    model:PyMADModel
        Model
    
    model_name:str
        Name of the model. 

    padding_type:str='right', 
        Padding type. By default, 'right'.
    
    eval_batch_size:int=32
        Evaluation batch size. By default, 32.

    Returns
    ---------
    PREDICTIONS: dict
        The prediction dictionary comprises of entity_scores, Y, Y_hat, Y_sigma, mask and anomaly_labels.
    """

    anomaly_labels = data.entities[0].labels

    dataloader = Loader(dataset=data,
                        batch_size=eval_batch_size,
                        window_size=model.window_size,
                        window_step=model.window_step,
                        shuffle=False,
                        padding_type=padding_type,
                        sample_with_replace=False,
                        verbose=False,
                        mask_position='None',
                        n_masked_timesteps=0)

    if model.window_size == -1:
        window_size = data.entities[0].Y.shape[1]
    else:
        window_size = model.window_size

    entity_scores = t.zeros((len(dataloader), data.n_features, window_size))

    n_features = data.n_features
    if 'DGHL' in model_name and (
            data.entities[0].X is not None):  # DGHL also considers covariates
        n_features = n_features + data.entities[0].X.shape[0]

    Y = np.zeros((len(dataloader), n_features, window_size))
    Y_hat = np.zeros((len(dataloader), n_features, window_size))
    Y_sigma = np.zeros((len(dataloader), n_features, window_size))
    mask = np.zeros((len(dataloader), n_features, window_size))

    step = 0
    for batch in dataloader:
        batch_size, n_features, window_size = batch['Y'].shape
        # Entity anomaly scores to compute PR-AUC and Centrality
        batch_anomaly_score = model.window_anomaly_score(input=batch,
                                                         return_detail=True)
        entity_scores[step:(step + batch_size), :, :] = batch_anomaly_score

        # Forecasting Error
        Y_b, Y_hat_b, Y_sigma_b, mask_b = predict(batch, model_name, model)
        Y[step:(step + batch_size), :, :] = Y_b
        Y_hat[step:(step + batch_size), :, :] = Y_hat_b
        Y_sigma[step:(step + batch_size), :, :] = Y_sigma_b
        mask[step:(step + batch_size), :, :] = mask_b

        step += batch_size

    # Final Anomaly Scores and forecasts
    entity_scores = model.final_anomaly_score(
        input=entity_scores, return_detail=False
    )  # return_detail = False averages the anomaly scores across features.
    entity_scores = entity_scores.detach().cpu().numpy()

    Y_hat = de_unfold(windows=Y_hat, window_step=model.window_step)
    Y = de_unfold(windows=Y, window_step=model.window_step)
    Y_sigma = de_unfold(windows=Y_sigma, window_step=model.window_step)
    mask = de_unfold(windows=mask, window_step=model.window_step)

    # Remove extra padding from Anomaly Scores and forecasts
    entity_scores = _adjust_scores_with_padding(
        scores=entity_scores,
        padding_size=dataloader.padding_size,
        padding_type=padding_type)

    Y_hat = _adjust_scores_with_padding(scores=Y_hat,
                                        padding_size=dataloader.padding_size,
                                        padding_type=padding_type)
    Y = _adjust_scores_with_padding(scores=Y,
                                    padding_size=dataloader.padding_size,
                                    padding_type=padding_type)
    Y_sigma = _adjust_scores_with_padding(scores=Y_sigma,
                                          padding_size=dataloader.padding_size,
                                          padding_type=padding_type)
    mask = _adjust_scores_with_padding(scores=mask,
                                       padding_size=dataloader.padding_size,
                                       padding_type=padding_type)

    return {
        'entity_scores': entity_scores,
        'Y': Y,
        'Y_hat': Y_hat,
        'Y_sigma': Y_sigma,
        'mask': mask,
        'anomaly_labels': anomaly_labels
    }


def evaluate_model_synthetic_anomalies(data: Union[Dataset, Entity],
                                       model: PyMADModel,
                                       model_name: str,
                                       padding_type: str = 'right',
                                       eval_batch_size: int = 128,
                                       n_repeats: int = 3,
                                       random_states: List[int] = [0, 1, 2],
                                       max_window_size: int = 128,
                                       min_window_size: int = 8,
                                       inject_abn_list: list = []) -> dict:
    """Compute observations necessary to evaluate a model on a given dataset with synthetic anomalies injected.

    Description
    -----------
    This function injects synthetic anomalies to the data and computes 
    predicted anomaly scores (entity scores). These observations
    are useful to evaluate performance of a model on synthetic anomalies. 

    Parameters
    ----------
    data:Union[Dataset, Entity]
        Dataset to evaluate the model on. If a dataset object is given, 
        we assume that it has a single entity. 
    
    model:PyMADModel
        Model
    
    model_name:str
        Name of the model. 

    padding_type:str='right'
        Padding type. By default, 'right'.
    
    eval_batch_size:int=32
        Evaluation batch size. By default, 32.

    n_repeats:int=3
        Number of indepent anomaly injection trials of each anomaly type. By default n_repeats=3. 
    
    random_states:List[int]=[0, 1, 2]
        Random seed for each trial. Constrols the anomaly injection. 
    
    max_window_size: int
        Maximum window size of injected anomaly
    
    min_window_size: int
        Miniumum window size of injected anomaly
    
    Returns
    ---------
    PREDICTIONS: dict
        The prediction dictionary comprises of entity_scores, Y (anomalous Y) and 
        anomalous scores returned by the anomaly injection algorithm. 
    """
    # ANOMALY_TYPES = list(ANOMALY_PARAM_GRID.keys())
    ANOMALY_TYPES = inject_abn_list
    original_data = deepcopy(data)
    PREDICTIONS = {}

    for anomaly_type in ANOMALY_TYPES:
        # We'll create a single long time-series for each anomaly type (including multiple repetitions and hyper-parameters)

        # Start with a clean slate (the original data)
        data = deepcopy(original_data)
        T = data.entities[0].Y
        data_std = max(np.std(T), 0.01)

        T_a_concatenated = []
        anomaly_sizes_concatenated = []
        anomaly_labels_concatenated = []

        for i in trange(n_repeats):
            # Create an anomaly object for each random seed for each unique repetition
            anomaly_obj = InjectAnomalies(random_state=random_states[i],
                                          verbose=False,
                                          max_window_size=max_window_size,
                                          min_window_size=min_window_size)
            for anomaly_params in list(
                    ParameterGrid(ANOMALY_PARAM_GRID[anomaly_type])):
                anomaly_params['T'] = T
                anomaly_params['scale'] = anomaly_params['scale'] * data_std
                anomaly_type = anomaly_params['anomaly_type']

                # Inject synthetic anomalies to the data
                T_a, anomaly_sizes, anomaly_labels = anomaly_obj.inject_anomalies(
                    **anomaly_params)
                anomaly_sizes = anomaly_sizes / data_std

                T_a_concatenated.append(T_a)
                anomaly_sizes_concatenated.append(anomaly_sizes)
                anomaly_labels_concatenated.append(anomaly_labels)

        T_a_concatenated = np.concatenate(T_a_concatenated, axis=1)
        anomaly_sizes_concatenated = np.concatenate(anomaly_sizes_concatenated,
                                                    axis=0)
        anomaly_labels_concatenated = np.concatenate(
            anomaly_labels_concatenated, axis=0)

        data.entities[0].Y = T_a_concatenated
        data.entities[0].n_time = T_a_concatenated.shape[1]
        data.entities[0].mask = np.ones((T_a_concatenated.shape))
        data.total_time = T_a_concatenated.shape[1]

        # Evaluate each model on the long time series
        if model.window_size == -1:
            window_size = data.entities[0].Y.shape[1]
        else:
            window_size = model.window_size

        dataloader = Loader(dataset=data,
                            batch_size=eval_batch_size,
                            window_size=model.window_size,
                            window_step=model.window_step,
                            shuffle=False,
                            padding_type=padding_type,
                            sample_with_replace=False,
                            verbose=True,
                            mask_position='None',
                            n_masked_timesteps=0)

        entity_scores = t.zeros(
            (len(dataloader), data.n_features, window_size))

        step = 0
        for batch in dataloader:
            batch_size, _, _ = batch['Y'].shape
            batch_anomaly_score = model.window_anomaly_score(
                input=batch, return_detail=True)
            entity_scores[step:(step + batch_size), :, :] = batch_anomaly_score
            step += batch_size

        # Final Anomaly Scores
        entity_scores = model.final_anomaly_score(
            input=entity_scores, return_detail=False
        )  # return_detail = False averages the anomaly scores across features.
        entity_scores = entity_scores.detach().cpu().numpy()

        # Remove extra padding from Anomaly Scores
        entity_scores = _adjust_scores_with_padding(
            scores=entity_scores,
            padding_size=dataloader.padding_size,
            padding_type=padding_type)

        PREDICTIONS[
            f'anomalysizes_type_{anomaly_type}'] = anomaly_sizes_concatenated
        PREDICTIONS[
            f'anomalylabels_type_{anomaly_type}'] = anomaly_labels_concatenated
        PREDICTIONS[f'entityscores_type_{anomaly_type}'] = entity_scores
        PREDICTIONS[f'Ta_type_{anomaly_type}'] = T_a_concatenated

    return PREDICTIONS


# for i in trange(n_repeats):
#     for j, anomaly_params in enumerate(
#             list(ParameterGrid(list(ANOMALY_PARAM_GRID.values())))):
#         anomaly_obj = InjectAnomalies(random_state=random_states[i],
#                                       verbose=False,
#                                       max_window_size=max_window_size,
#                                       min_window_size=min_window_size)
#         data = deepcopy(original_data)
#         T = data.entities[0].Y

#         data_std = max(np.std(T), 0.01)

#         anomaly_params['T'] = T
#         anomaly_params['scale'] = anomaly_params['scale'] * data_std
#         anomaly_type = anomaly_params['anomaly_type']

#         # Inject synthetic anomalies to the data
#         T_a, anomaly_sizes, anomaly_labels =\
#             anomaly_obj.inject_anomalies(**anomaly_params)

#         anomaly_sizes = anomaly_sizes / data_std
#         data.entities[0].Y = T_a
#         data.entities[0].n_time = T_a.shape[1]
#         data.entities[0].mask = np.ones((T_a.shape))
#         data.total_time = T_a.shape[1]

# return PREDICTIONS

######################################################
# Helper functions
######################################################


def rank_models(
        models_performance_matrix: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    # If the value is lower for a model, the model is better

    LOWER_BETTER = ['MAE', 'MSE', 'SMAPE', 'MAPE', 'CENTRALITY']
    # If the value is higher for a model, the model is better
    HIGHER_BETTER = ['LIKELIHOOD', 'SYNTHETIC', 'PR-AUC', 'Best F-1', 'VUS']

    METRIC_NAMES = [i.split('_')[0] for i in models_performance_matrix.columns]
    SORT_DIRECTION = []
    for mn in METRIC_NAMES:
        if mn in HIGHER_BETTER:
            SORT_DIRECTION.append('Desc')
        elif mn in LOWER_BETTER:
            SORT_DIRECTION.append('Asc')
        else:
            raise ValueError('Undefined metric sort direction.')

    ranks = np.zeros(models_performance_matrix.shape).T
    # print(f'models_performance_matrix is {models_performance_matrix},columns is {models_performance_matrix.columns}')
    for i, metric_name in enumerate(models_performance_matrix.columns):
        # print(f'models_performance_matrix.loc[:, metric_name].to_numpy() is \n{models_performance_matrix.loc[:, metric_name].to_numpy()}')
        print(f'metric_namei is {metric_name}')
        if SORT_DIRECTION[i] == 'Asc':
            print('------ASC-------')
            print(models_performance_matrix[metric_name].values)
            print(np.argsort(models_performance_matrix[metric_name].values))
            ranks[i, :] = np.argsort(
                models_performance_matrix.loc[:, metric_name].to_numpy())
            print('------ASC over-------')
        elif SORT_DIRECTION[i] == 'Desc':
            print('------Desc-------')
            models_performance_matrix.loc[:, metric_name].to_numpy()
            ranks[i, :] = np.argsort(
                -models_performance_matrix.loc[:, metric_name].to_numpy())
            print('------Desc over-------')
    rank_prauc = ranks[0, :]  # Rank based on PR-AUC
    rank_f1 = ranks[1, :]  # Rank based on F-1
    rank_vus = ranks[2, :]  # Rank based on VUS
    ranks_by_metrics = ranks[3:, ]  # Ranks 

    return ranks_by_metrics, rank_prauc, rank_f1, rank_vus


def get_eval_batchsizes(model_name: str) -> int:
    """Return evaluation batch sizes of algorithm
    """
    _VALID_MODEL_NAMES = ['RNN', 'DGHL', 'LSTMVAE', 'MD', 'RM',
                          'NN', 'LOF','ABOD','KDE','COF','CBLOF','SOS']  # TODO: Should be stored somewhere centrally

    model_type = model_name.split('_')[0]

    if model_type == 'RNN':
        return RNN_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'DGHL':
        return DGHL_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'NN':
        return NN_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'MD':
        return MD_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'RM':
        return RM_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'LSTMVAE':
        return LSTMVAE_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'LOF':
        return LOF_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'ABOD':
        return ABOD_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'KDE':
        return KDE_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'CBLOF':
        return CBLOF_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'COF':
        return COF_TRAIN_PARAM_GRID['eval_batch_size'][0]
    elif model_type == 'SOS':
        return SOS_TRAIN_PARAM_GRID['eval_batch_size'][0]
    else:
        raise AttributeError(
            f'Model type must be one of {_VALID_MODEL_NAMES}, but {model_type} was passed!'
        )


def _adjust_scores_with_padding(scores: np.ndarray,
                                padding_size: int = 0,
                                padding_type: str = 'right'):
    if scores.ndim == 1: scores = scores[None, :]

    if (padding_type == 'right') and (padding_size > 0):
        scores = scores[:, :-padding_size]
    elif (padding_type == 'left') and (padding_size > 0):
        scores = scores[:, padding_size:]
    return scores


######################################################
# Helper prediction functions for predict
######################################################


def _predict_base(batch, model):
    Y, Y_hat, mask = model.forward(batch)
    if isinstance(Y, t.Tensor): Y = Y.detach().cpu().numpy()
    if isinstance(Y_hat, t.Tensor): Y_hat = Y_hat.detach().cpu().numpy()
    if isinstance(mask, t.Tensor): mask = mask.detach().cpu().numpy()
    Y_sigma = np.NaN * np.ones(batch['Y'].shape)
    return Y, Y_hat, Y_sigma, mask


def _predict_dghl(batch, model):
    return _predict_base(batch, model)


def _predict_md(batch, model):
    return _predict_base(batch, model)


def _predict_nn(batch, model):
    return _predict_base(batch, model)


def _predict_rm(batch, model):
    return _predict_base(batch, model)


def _predict_lof(batch, model):
    return _predict_base(batch, model)


def _predict_kde(batch, model):
    return _predict_base(batch, model)


def _predict_abod(batch, model):
    return _predict_base(batch, model)

def _predict_sos(batch, model):
    return _predict_base(batch, model)

def _predict_cof(batch, model):
    return _predict_base(batch, model)


def _predict_cblof(batch, model):
    return _predict_base(batch, model)


def _predict_rnn(batch, model):
    batch_size, n_features, window_size = batch['Y'].shape
    Y, Y_hat, mask = model.forward(batch)
    Y, Y_hat, mask = Y.detach().cpu().numpy(), Y_hat.detach().cpu().numpy(
    ), mask.detach().cpu().numpy()
    Y = Y.reshape(n_features, -1)[:, :window_size]  # to [n_features, n_time]
    Y_hat = Y_hat.reshape(n_features,
                          -1)[:, :window_size]  # to [n_features, n_time]
    mask = mask.reshape(n_features,
                        -1)[:, :window_size]  # to [n_features, n_time]

    # Add mask dimension
    Y = Y[None, :, :]
    Y_hat = Y_hat[None, :, :]
    mask = mask[None, :, :]

    Y_sigma = np.NaN * np.ones(batch['Y'].shape)
    return Y, Y_hat, Y_sigma, mask


def _predict_lstmvae(batch, model):
    Y, Y_mu, mask, Y_sigma, *_ = model.forward(batch)
    Y, Y_hat, mask, Y_sigma = Y.detach().cpu().numpy(), Y_mu.detach().cpu(
    ).numpy(), mask.detach().cpu().numpy(), Y_sigma.detach().cpu().numpy()
    return Y, Y_hat, Y_sigma, mask
