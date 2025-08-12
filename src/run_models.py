"""Run experiments for last-layer inference models."""

import argparse
from pathlib import Path
import pickle

import math
import numpy as np
from typing import Tuple
import torch
import logging
import json

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from numpy.lib.stride_tricks import sliding_window_view
import torch.nn.functional as F
from models.last_layer_models import fit_last_layer_closed_form, get_metrics_lli_closed_form

from models.mc_dropout import fit_mc_dropout, get_metrics_mc_dropout, get_coverage_mc_dropout
from models.bnn import fit_bnn, get_metrics_bnn
from models.gibbs_sampler import gibbs_sampler, get_pred_post_dist, get_prediction_interval_coverage
from models.vi import fit_vi_post_hoc, fit_vi_post_hoc, predictive_posterior, run_last_layer_vi_closed_form
from models.sg_mcmc import train_sg_mcmc
from tqdm import tqdm
from scipy.stats import norm
from utils.data_utils import create_synthetic_train_data, f

from utils.coverage import get_coverage_gaussian, get_coverage_y_hats

np.random.seed(100)

PARAMS_SYNTH = {'sigma_eps': 0.3,
                'num_points': 200, 
                'xs_range': [-4,4], 
                'num_epochs': 100, 
                'sigma_0': 0.3, 
                'CI_levels': np.linspace(0.001, 0.99, 100)}


logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Train prob. model on save predictive means, 
    predictive variance, pred. samples and prediction intervals."""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Method for prob. neural network.",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Method for prob. neural network.",
    )

    parser.add_argument(
        "--model_dims",
        type=int,
        nargs="+",
        required=True,
        help="Directory to save outputs to",
    )

    parser.add_argument(
        "--outpath",
        type=Path,
        required=True,
        help="Directory to save outputs to",
    )

    args = parser.parse_args()
    params = vars(args)


    # create training data
    if params['data'] == 'synthetic':

        xs, ys, xs_train, ys_train, xs_val, ys_val = create_synthetic_train_data(xs_range =  PARAMS_SYNTH["xs_range"], 
                                                                                 num_points = PARAMS_SYNTH['num_points'],  
                                                                                 sigma_eps = PARAMS_SYNTH['sigma_eps'])

        data = TensorDataset(xs_train, ys_train)
        train_set, val_set = torch.utils.data.random_split(data, [80, 20])
        dataloader_train = DataLoader(train_set, shuffle = True, batch_size=5)
        dataloader_val = DataLoader(val_set, shuffle = True, batch_size=20)
    else:
        ValueError('Please provide a valid dataset name from: synthetic.')

    if params['method'] == 'mc_dropout':
        # train model
        logger.info("Start training {params['method']}....")
        mc_net, out_dict = fit_mc_dropout(model_dims =params['model_dims'], 
                                          xs_pred = xs_val, 
                                          dataloader_train = dataloader_train, 
                                          dataloader_val = dataloader_val,
                                          num_epochs = PARAMS_SYNTH['num_epochs'] )

        logger.info("...computing RMSE...")
        rmse_mc_mean, rmse_mc_std, _, _, ys_samples_mc = get_metrics_mc_dropout(model = mc_net, 
                                                                                xs_val = xs_val, 
                                                                                ys_val = ys_val)

        logger.info("...computing prediction intervals and coverage...")
        coverage =  get_coverage_mc_dropout(ys_samples_mc, ys_val, PARAMS_SYNTH['CI_levels'])

         # save
        out_dict['rmse_mean'] = rmse_mc_mean, 
        out_dict['rmse_std'] = rmse_mc_std,
        out_dict['coverage'] = coverage

        torch.save(
            {
                "model_state_dict": mc_net.state_dict(),
            },
           params['outpath'] / f"{params['method']}_checkpoint.t7"
        )

        with open(params['outpath'] / f"out_dict_{params['method']}.pkl", "wb") as f:
            pickle.dump(out_dict, f)

        logger.info(f"... everything saved under {params['outpath']}.")

       

    elif params['method'] == 'bnn':
        bnn, out_dict = fit_bnn(num_epochs = PARAMS_SYNTH['num_epochs']*2, 
                                          xs_pred = xs_val,
                                          dataloader_train = dataloader_train, 
                                          dataloader_val = dataloader_val)
        
        bnn_samples, rmse_bnn_mean, rmse_bnn_std = get_metrics_bnn(bnn = bnn, xs_val = xs_val, ys_val = ys_val)

        coverage = get_coverage_y_hats(y_samples = torch.stack(bnn_samples).squeeze(), 
                                        y_true = ys_val, 
                                        levels = PARAMS_SYNTH['CI_levels'])

        # save
        out_dict['rmse_mean'] = rmse_bnn_mean, 
        out_dict['rmse_std'] = rmse_bnn_std,
        out_dict['coverage'] = coverage

        torch.save(
            {
                "model_state_dict": bnn.state_dict(),
            },
           params['outpath'] / f"{params['method']}_checkpoint.t7"
        )

        with open(params['outpath'] / f"out_dict_{params['method']}.pkl", "wb") as f:
            pickle.dump(out_dict, f)

        logger.info(f"... everything saved under {params['outpath']}.")



    
    elif params['method'] == 'lli_vi_closed_full_cov':
        logger.info("Start training {params['method']}....")
        # fit model
        out_dict, net = fit_last_layer_closed_form(model_dims = params['model_dims'], 
                                   dataloader_train = dataloader_train, 
                                   dataloader_val = dataloader_val, 
                                   xs_train = xs_train, 
                                   ys_train = ys_train, 
                                   xs_val = xs_val, 
                                   num_epochs = PARAMS_SYNTH['num_epochs'], 
                                   sigma_0 = PARAMS_SYNTH['sigma_0'])
    
        
        logger.info("...computing RMSE...")
        rmse_lli_mean, rmse_lli_std, lli_pred_mu, lli_pred_sigma = get_metrics_lli_closed_form(mu_N = torch.tensor(out_dict['mu_N']), 
                                                                  Sigma_N = out_dict['Sigma_N'], 
                                                                  model = net, 
                                                                  xs_val = xs_val, 
                                                                  ys_val = ys_val, 
                                                                  sigma_0 = PARAMS_SYNTH['sigma_0'] )
        logger.info("...computing prediction intervals and coverage...")
        coverage_lli = get_coverage_gaussian(pred_mean = lli_pred_mu, 
                                             pred_std = lli_pred_sigma, 
                                             y_true = ys_val.detach().numpy(), 
                                             levels=PARAMS_SYNTH['CI_levels'])
        
        out_dict['rmse_mean'] = rmse_lli_mean, 
        out_dict['rmse_std'] = rmse_lli_std,
        out_dict['coverage'] = coverage_lli

        torch.save(
            {
                "model_state_dict": net.state_dict(),
            },
           params['outpath'] / f"{params['method']}_checkpoint.t7"
        )

        with open(params['outpath'] / f"out_dict_{params['method']}.pkl", "wb") as f:
            pickle.dump(out_dict, f)

        logger.info(f"... everything saved under {params['outpath']}.")

    elif params['method'] == 'lli_vi_ridge':
        
        pass
    elif params['method'] == 'lli_vi_horseshoe':
        pass

    elif params['method'] == 'lli_vi_fac_ridge':
        pass

    elif params['method'] == 'lli_vi_fac_horseshoe':
        pass

    elif params['method'] == 'lli_gibbs_ridge':
        pass
    
    else:
        ValueError('Please provide a valid method from: \n mc_dropout, bnn, ' \
                    'lli_vi_ridge, lli_vi_horseshoe,\n lli_vi_fac_ridge,' \
                    ' lli_vi_fac_horseshoe, lli_gibbs_ridge. ')
        

if __name__ == "__main__":
    main()
