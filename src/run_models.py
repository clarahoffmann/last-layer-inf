"""Run experiments for last-layer inference models."""

import argparse
from pathlib import Path
import pickle

import math
import numpy as np
import torch
import logging

from torch.utils.data import DataLoader, TensorDataset
from models.last_layer_models import fit_last_layer_closed_form, get_metrics_lli_closed_form, LLI, LastLayerVIClosedForm, LastLayerVIRidge, LastLayerVIHorseshoe, LastLayerVIRidgeFullCov, LastLayerVIHorseshoeFullCov

from models.mc_dropout import fit_mc_dropout, get_metrics_mc_dropout, get_coverage_mc_dropout
from models.bnn import fit_bnn, get_metrics_bnn
from models.sg_mcmc import train_sg_mcmc_gauss, predict_sg_mcmc_gauss
from models.gibbs_sampler import gibbs_sampler, get_metrics_ridge, get_prediction_interval_coverage
from models.hmc import make_log_prob_flat, HMC, unpack_samples
from models.vi import run_last_layer_vi_closed_form, run_last_layer_vi_ridge, run_last_layer_vi_horseshoe, run_last_layer_vi_ridge_full_fac, run_last_layer_vi_horseshoe_full_fac
from utils.data_utils import create_synthetic_train_data, create_concrete_train_data

from utils.coverage import get_coverage_gaussian, get_coverage_y_hats

np.random.seed(100)

PARAMS_SYNTH = {'sigma_eps': 0.3,
                'num_points': 200, 
                'xs_range': [-4,4], 
                'num_epochs': 100, 
                'sigma_0': 0.3, 
                'CI_levels': np.linspace(0.001, 0.99, 10), 
                'ys_grid': torch.arange(-5,5, 0.01) # grid at which to evaluate samples of dens.
                }

PARAMS_CONCRETE = {'sigma_eps': 0.05,
                'num_points': 200, 
                'xs_range': [-4,4], 
                'num_epochs': 100, 
                'sigma_0': 0.3, 
                'CI_levels': np.linspace(0.001, 0.99, 10), 
                'ys_grid': torch.arange(-5,5, 0.01) # grid at which to evaluate samples of dens.
                }

PARAMS_RIDGE_SYNTHETIC = {'a_sigma': 2,
                'b_sigma': 2,
                'a_tau': 2,
                'b_tau': 2,
                'num_iter': 2000,
                'warm_up': 1000,
                }

PARAMS_RIDGE_CONCRETE = {'a_sigma': 2,
                'b_sigma': 2,
                'a_tau': 2,
                'b_tau': 2,
                'num_iter': 3000,
                'warm_up': 1000,
                }

PARAMS_SG_MCMC = {'T': 2000,
                  'warm_up': 1000,
                  'a': 0.1, 
                  'b': 1}


logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_backbone(params, logger):
    ckpt_backbone = Path(f"./results/lli_closed_form_{params['data']}_checkpoint.t7")
    
    if ckpt_backbone.is_file():
        logger.info(f"Loading backbone from {ckpt_backbone}...")
        lli_net = LLI(params['model_dims'])
        checkpoint = torch.load(ckpt_backbone)
        lli_net.load_state_dict(checkpoint['model_state_dict'])
        lli_net.eval()
        logger.info("...backbone loaded.")

        return lli_net
    else:
        Exception('Please train a backbone first by calling'
                    + 'the script with method lli_closed_form.')



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

        data_file = Path("./results/synthetic_data.npz")
        if data_file.is_file():
            logger.info("Loading existing synthetic training data.")
            data = np.load(data_file)
            xs_train = torch.tensor(data["xs_train"])
            ys_train = torch.tensor(data["ys_train"])
            xs_val   = torch.tensor(data["xs_val"])
            ys_val   = torch.tensor(data["ys_val"])

        else:
            logger.info("Creating new synthetic training data.")
            xs, ys, xs_train, ys_train, xs_val, ys_val = create_synthetic_train_data(xs_range =  PARAMS_SYNTH["xs_range"], 
                                                                                 num_points = PARAMS_SYNTH['num_points'],  
                                                                                 sigma_eps = PARAMS_SYNTH['sigma_eps'])
            logger.info("Save synthetic training data.")
            np.savez_compressed(
                data_file,
                xs=xs,
                ys=ys,
                xs_train=xs_train,
                ys_train=ys_train,
                xs_val=xs_val,
                ys_val=ys_val
            )

        data = TensorDataset(xs_train, ys_train)
        train_set, val_set = torch.utils.data.random_split(data, [80, 20])
        dataloader_train = DataLoader(train_set, shuffle = True, batch_size=5)
        dataloader_val = DataLoader(val_set, shuffle = True, batch_size=5)
    
    elif params['data'] == 'concrete':
        
        data_file = Path("./results/concrete_data.npz")
        
        if data_file.is_file():
            logger.info("Loading existing concrete training data.")
            data = np.load(data_file)
            xs_train = torch.tensor(data["xs_train"])
            ys_train = torch.tensor(data["ys_train"])
            xs_val   = torch.tensor(data["xs_val"])
            ys_val   = torch.tensor(data["ys_val"])
            
        else:
            xs, ys, xs_train, ys_train, xs_val, ys_val = create_concrete_train_data(num_train_points = 700)
            logger.info("Save synthetic training data.")
            np.savez_compressed(
                data_file,
                xs=xs,
                ys=ys,
                xs_train=xs_train,
                ys_train=ys_train,
                xs_val=xs_val,
                ys_val=ys_val
            )

        data = TensorDataset(xs_train, ys_train)
        train_set, val_set = torch.utils.data.random_split(data, [int(0.8*700), int(0.2*700)])
        dataloader_train = DataLoader(train_set, shuffle = True, batch_size=64)
        dataloader_val = DataLoader(val_set, shuffle = True, batch_size=64)

    else:
        ValueError('Please provide a valid dataset name from: synthetic.')

    if params['method'] == 'mc_dropout':
        # train model
        logger.info(f"Start training {params['method']}....")
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
        coverage =  get_coverage_mc_dropout(ys_samples_mc.squeeze(), ys_val, PARAMS_SYNTH['CI_levels'])

         # save
        out_dict['rmse_mean'] = rmse_mc_mean.item() 
        out_dict['rmse_std'] = rmse_mc_std.item()
        out_dict['coverage'] = coverage

        torch.save(
            {
                "model_state_dict": mc_net.state_dict(),
            },
           params['outpath'] / f"{params['method']}_{params['data']}checkpoint.t7"
        )

    elif params['method'] == 'bnn':
        if params['data'] == 'synthetic':
            input_dim = 1
            hidden_dim = 100
        elif params['data'] == 'concrete':
            input_dim = 8
            hidden_dim = 150
        bnn, out_dict = fit_bnn(num_epochs = PARAMS_SYNTH['num_epochs']*2, 
                                          xs_pred = xs_val,
                                          dataloader_train = dataloader_train, 
                                          dataloader_val = dataloader_val, 
                                          input_dim = input_dim, 
                                          hidden_dim = hidden_dim)
        
        bnn_samples, rmse_bnn_mean, rmse_bnn_std = get_metrics_bnn(bnn = bnn, 
                                                                   xs_val = xs_val, 
                                                                   ys_val = ys_val)

        coverage = get_coverage_y_hats(y_samples = torch.stack(bnn_samples).squeeze(), 
                                        y_true = ys_val, 
                                        levels = PARAMS_SYNTH['CI_levels'])

        # save
        out_dict['rmse_mean'] = rmse_bnn_mean.item()
        out_dict['rmse_std'] = rmse_bnn_std.item()
        out_dict['coverage'] = coverage

        torch.save(
            {
                "model_state_dict": bnn.state_dict(),
            },
           params['outpath'] / f"{params['method']}_{params['data']}checkpoint.t7"
        )
    
    elif params['method'] == 'lli_closed_form':
        logger.info(f"Start training {params['method']}....")
        # fit model
        if params['data'] == 'synthetic':
            sigma_0 = PARAMS_SYNTH['sigma_0']
        else:
            sigma_0 = PARAMS_CONCRETE['sigma_0']

        out_dict, net = fit_last_layer_closed_form(model_dims = params['model_dims'], 
                                   dataloader_train = dataloader_train, 
                                   dataloader_val = dataloader_val, 
                                   xs_train = xs_train, 
                                   ys_train = ys_train, 
                                   xs_val = xs_val, 
                                   num_epochs = PARAMS_SYNTH['num_epochs'], 
                                   sigma_0 = sigma_0)
    
        
        logger.info("...computing RMSE...")
        rmse_lli_mean, rmse_lli_std, lli_pred_mu, lli_pred_sigma = get_metrics_lli_closed_form(mu_N = torch.tensor(out_dict['mu_N']), 
                                                                  Sigma_N = out_dict['Sigma_N'], 
                                                                  model = net, 
                                                                  xs_val = xs_val, 
                                                                  ys_val = ys_val, 
                                                                  sigma_0 = sigma_0 )

        logger.info("...computing prediction intervals and coverage...")
        coverage_lli = get_coverage_gaussian(pred_mean = lli_pred_mu.squeeze(), 
                                             pred_std = lli_pred_sigma.squeeze(), 
                                             y_true = ys_val.detach().numpy().squeeze(), 
                                             levels=PARAMS_SYNTH['CI_levels'])
        
        out_dict['rmse_mean'] = rmse_lli_mean.item()
        out_dict['rmse_std'] = rmse_lli_std.item()
        out_dict['coverage'] = coverage_lli

        torch.save(
            {
                "model_state_dict": net.state_dict(),
            },
           params['outpath'] / f"{params['method']}_{params['data']}_checkpoint.t7"
        )

    elif params['method'] == 'lli_gibbs_ridge':

        if params['data'] == 'synthetic':
            params_ridge = PARAMS_RIDGE_SYNTHETIC
        elif params['data'] == 'concrete':
            params_ridge = PARAMS_RIDGE_CONCRETE

        # load from checkpoint if we have trained a deep feature projector already.
        lli_net = load_backbone(params, logger)
        
        with torch.no_grad():
            Psi = lli_net.get_ll_embedd(xs_train).detach()

        logger.info('Run Gibbs sampler...')
        w_sample, _, sigma_sq_sample = gibbs_sampler(Psi = Psi, 
                                                    ys = ys_train, 
                                                    a_tau = params_ridge['a_tau'], 
                                                    b_tau = params_ridge['b_tau'], 
                                                    a_sigma = params_ridge['a_sigma'], 
                                                    b_sigma = params_ridge['b_sigma'], 
                                                    num_iter = params_ridge['num_iter'], 
                                                    warm_up = params_ridge['warm_up'])
        logger.info('... finished.')
                                
        out_dict = {'params': params_ridge,
                    'w_sample': w_sample,
                    'sigma_sq_sample': sigma_sq_sample}
        
        logger.info("...computing RMSE...")
        pdf_gibbs_ridge, rmse_lli_gibbs_mean, rmse_ll_gibbs_std, pred_mu, pred_var = get_metrics_ridge(model = lli_net, 
                                                                                    xs_val = xs_val, 
                                                                                    ys_val = ys_val,  
                                                                                    w_sample = w_sample, 
                                                                                    sigma_sq_sample = sigma_sq_sample, 
                                                                                    ys_grid = PARAMS_SYNTH['ys_grid'])
        pred_std = np.sqrt(pred_var)
        logger.info("...computing prediction intervals and coverage...")
        coverage_gibbs_ridge = get_prediction_interval_coverage(ys_grid = PARAMS_SYNTH['ys_grid'],
                                                                ys_true =ys_val,
                                                                p_hats = torch.tensor(pdf_gibbs_ridge), 
                                                                levels = PARAMS_SYNTH['CI_levels'])

        for key, value in zip(['rmse_mean', 'rmse_std', 'pred_mu', 'pred_std', 'coverage'], 
                              [rmse_lli_gibbs_mean, rmse_ll_gibbs_std, np.array(pred_mu),np.sqrt(np.array(pred_std)),
                                coverage_gibbs_ridge]):
            out_dict[key] = value

    elif params['method'] == 'lli_vi_closed_form':

        lli_net = load_backbone(params, logger)
        
        for param in lli_net.parameters():
            param.requires_grad = False
        with torch.no_grad():
            Psi = lli_net.get_ll_embedd(xs_train)

        logger.info('Run VI...')
        last_layer_vi = LastLayerVIClosedForm(dim_last_layer=Psi.shape[1], dim_output=1)

        last_layer_vi, elbos = run_last_layer_vi_closed_form(model = last_layer_vi, 
                                    ys_train = ys_train, 
                                    Psi = Psi, sigma_eps_sq = PARAMS_SYNTH['sigma_eps']**2, 
                                    lr = 1e-2, temperature = 1, num_epochs = 1000)
        logger.info('...finished.')
        
        logger.info('Predict on validation data....')
        with torch.no_grad():
            Psi_val = lli_net.get_ll_embedd(xs_val).detach()
        
        pred_mu = (Psi_val @  last_layer_vi.mu.T)
        L  = last_layer_vi.get_L()
        Z = Psi_val @ L.squeeze()  
        pred_std = torch.sqrt((Z ** 2).sum(dim=1) + PARAMS_SYNTH['sigma_eps']**2)
        logger.info('... finished.')

        out_dict = {'pred_mu': pred_mu.detach().numpy(),
                    'pred_sigma': pred_std.detach().numpy(),
                    'elbos': elbos
                    }
        
        logger.info('Get metrics...')

        rmse_lli = (ys_val.squeeze() - pred_mu.squeeze()).pow(2).sqrt()
        rmse_mean = rmse_lli.mean().item()
        rmse_std = rmse_lli.std().item()
        coverage = get_coverage_gaussian(pred_mean = pred_mu.detach().numpy().squeeze(), 
                                         pred_std = pred_std.detach().numpy().squeeze(), 
                                         y_true = ys_val.detach().numpy().squeeze(), 
                                         levels=PARAMS_SYNTH['CI_levels'])
        logger.info('...finished.')

        for key, value in zip(['rmse_mean', 'rmse_std', 'coverage'], [rmse_mean, rmse_std, coverage]):
            out_dict[key] = value


    elif params['method'] == 'lli_vi_ridge':
        
        lli_net = load_backbone(params, logger)
        
        for param in lli_net.parameters():
            param.requires_grad = False
        with torch.no_grad():
            Psi = lli_net.get_ll_embedd(xs_train)

        last_layer_vi = LastLayerVIRidge(in_features=Psi.shape[1], out_features=1)
        optimizer_vi = torch.optim.Adam(last_layer_vi.parameters(), lr=1e-3)

        logger.info('Run VI...')
        last_layer_vi, elbos = run_last_layer_vi_ridge(model = last_layer_vi, 
                                Psi = Psi, 
                                ys_train = ys_train, 
                                optimizer_vi = optimizer_vi, 
                                num_epochs = 30000)
        logger.info('...finished.')
        
        logger.info('Predict on validation data....')
        with torch.no_grad():
            Psi_val = lli_net.get_ll_embedd(xs_val)
            pred_mu = (Psi_val @  last_layer_vi.mu.T)
            Sigma_q  = last_layer_vi.get_Sigma_q().squeeze()
            Z = torch.diag(Psi_val @ torch.diag(Sigma_q[:Psi_val.shape[1]]) @ Psi_val.T)
            pred_std = torch.sqrt(Z + torch.exp(last_layer_vi.log_sigma_eps_sq))
        
        logger.info('... finished.')

        out_dict = {'pred_mu': pred_mu.detach().numpy(),
                    'pred_sigma': pred_std.detach().numpy(),
                    'elbos': elbos
                    }
        
        logger.info('Get metrics...')
        rmse_lli = (ys_val.reshape(-1,1) - pred_mu).pow(2).sqrt()
        rmse_mean = rmse_lli.mean().item()
        rmse_std = rmse_lli.std().item()
        coverage = get_coverage_gaussian(pred_mean = pred_mu.detach().numpy().squeeze(), 
                                         pred_std = pred_std.detach().numpy().squeeze(), 
                                         y_true = ys_val.detach().numpy().squeeze(), 
                                         levels=PARAMS_SYNTH['CI_levels'])
        logger.info('...finished.')

        for key, value in zip(['rmse_mean', 'rmse_std', 'coverage'], [rmse_mean, rmse_std, coverage]):
            out_dict[key] = value

    elif params['method'] == 'lli_vi_horseshoe':
        lli_net = load_backbone(params, logger)
        
        for param in lli_net.parameters():
            param.requires_grad = False
        with torch.no_grad():
            Psi = lli_net.get_ll_embedd(xs_train)

        last_layer_vi_hs = LastLayerVIHorseshoe(in_features=Psi.shape[1], out_features=1)
        optimizer_vi = torch.optim.Adam(last_layer_vi_hs.parameters(), lr=1e-3)

        logger.info('Run VI...')
        last_layer_vi_hs, elbos = run_last_layer_vi_horseshoe(model = last_layer_vi_hs, 
                                    Psi = Psi, 
                                    ys_train = ys_train, 
                                    optimizer_vi = optimizer_vi, num_epochs = 30000, lr=1e-3)
        logger.info('...finished.')
        
        logger.info('Predict on validation data....')
        with torch.no_grad():
            Psi_val = lli_net.get_ll_embedd(xs_val)
            pred_mu = (Psi_val @  last_layer_vi_hs.mu.T)
            Sigma_q  = last_layer_vi_hs.get_Sigma_q().squeeze()
            Z = torch.diag(Psi_val @ torch.diag(Sigma_q[:Psi_val.shape[1]]) @ Psi_val.T)
            pred_std = torch.sqrt(Z  + torch.exp(last_layer_vi_hs.log_sigma_eps_sq_mu)) #+ PARAMS_SYNTH['sigma_eps'
        logger.info('...finished.')

        out_dict = {'pred_mu': pred_mu.detach().numpy(),
                    'pred_sigma': pred_std.detach().numpy(),
                    'elbos': elbos
                    }

        logger.info('Get metrics...')
        rmse_lli = (ys_val.reshape(-1,1) - pred_mu).pow(2).sqrt()
        rmse_mean = rmse_lli.mean().item()
        rmse_std = rmse_lli.std().item()
        coverage = get_coverage_gaussian(pred_mean = pred_mu.detach().numpy().squeeze(), 
                                         pred_std = pred_std.detach().numpy().squeeze(), 
                                         y_true = ys_val.detach().numpy().squeeze(), 
                                         levels=PARAMS_SYNTH['CI_levels'])
        
        logger.info('...finished.')

        for key, value in zip(['rmse_mean', 'rmse_std', 'coverage'], [rmse_mean, rmse_std, coverage]):
            out_dict[key] = value

    elif params['method'] == 'lli_vi_ridge_full_fac':

        lli_net = load_backbone(params, logger)
        
        for param in lli_net.parameters():
            param.requires_grad = False
        with torch.no_grad():
            Psi = lli_net.get_ll_embedd(xs_train)
            _, L = Psi.shape

        last_layer_vi = LastLayerVIRidgeFullCov(in_features=Psi.shape[1], out_features=1)
        optimizer_vi = torch.optim.Adam(last_layer_vi.parameters(), lr=1e-3)

        logger.info('Run VI...')
        last_layer_vi, elbos = run_last_layer_vi_ridge_full_fac(last_layer_vi, Psi, ys_train, optimizer_vi, num_epochs = 20000)
        logger.info('...finished.')

        logger.info('Predict on validation data....')
        with torch.no_grad():
            Psi_val = lli_net.get_ll_embedd(xs_val)
            pred_mu = (Psi_val @ last_layer_vi.mu.T)
            Sigma_q, _, _ = last_layer_vi.get_Sigma_q()
            Sigma_w = Sigma_q[:L, :L]
            Z = torch.diag(Psi_val @ Sigma_w @ Psi_val.T) + 1e-6
            pred_std = torch.sqrt(Z + torch.exp(last_layer_vi.log_sigma_eps_sq))
        logger.info('...finished.')

        out_dict = {'pred_mu': pred_mu.detach().numpy(),
                    'pred_sigma': pred_std.detach().numpy(),
                    'elbos': elbos
                    }
        
        logger.info('Get metrics...')
        rmse_lli = (ys_val.reshape(-1,1) - pred_mu).pow(2).sqrt()
        rmse_mean = rmse_lli.mean().item()
        rmse_std = rmse_lli.std().item()
        coverage = get_coverage_gaussian(pred_mean = pred_mu.detach().numpy().squeeze(), 
                                         pred_std = pred_std.detach().numpy().squeeze(), 
                                         y_true = ys_val.detach().numpy().squeeze(), 
                                         levels=PARAMS_SYNTH['CI_levels'])
        
        logger.info('...finished.')

        for key, value in zip(['rmse_mean', 'rmse_std', 'coverage'], [rmse_mean, rmse_std, coverage]):
            out_dict[key] = value


    elif params['method'] == 'lli_vi_horseshoe_full_fac':
        lli_net = load_backbone(params, logger)
        
        for param in lli_net.parameters():
            param.requires_grad = False
        with torch.no_grad():
            Psi = lli_net.get_ll_embedd(xs_train)
        
        last_layer_vi = LastLayerVIHorseshoeFullCov(in_features=Psi.shape[1], out_features=1, rank_B = 15)
        optimizer_vi = torch.optim.Adam(last_layer_vi.parameters(), lr=1e-3)

        logger.info(f'Run VI...{Psi.shape}')
        last_layer_vi, elbos = run_last_layer_vi_horseshoe_full_fac(last_layer_vi, Psi, ys_train, optimizer_vi, num_epochs = 80000)
        logger.info('...finished')

        logger.info('Predict on validation data....')
        with torch.no_grad():
            Psi_val = lli_net.get_ll_embedd(xs_val)
            pred_mu = (Psi_val @ last_layer_vi.mu.T).detach().numpy().squeeze()
            Sigma_q, _, _ = last_layer_vi.get_Sigma_q()
            L = last_layer_vi.in_features
            Sigma_w = Sigma_q[:L, :L]
            Z = torch.sum((Psi_val @ Sigma_w) * Psi_val, dim=1)
            # change below if something is weird
            sigma_eps_sq = torch.exp(last_layer_vi.log_sigma_eps_sq_mu).squeeze()  
            pred_std = torch.sqrt( Z + sigma_eps_sq).detach().numpy()
        logger.info('...finished.')

        out_dict = {'pred_mu': pred_mu,
                    'pred_sigma': pred_std,
                    'elbos': elbos
                    }

        logger.info('Get metrics...')
        rmse_lli = (ys_val.reshape(-1,1) - pred_mu).pow(2).sqrt()
        rmse_mean = rmse_lli.mean().item()
        rmse_std = rmse_lli.std().item()
        
        # check again! 
        # I think we need to compute the coverage here differently
        coverage = get_coverage_gaussian(pred_mean = pred_mu.squeeze(), 
                                         pred_std = pred_std.squeeze(), 
                                         y_true = ys_val.detach().numpy().squeeze(), 
                                         levels=PARAMS_SYNTH['CI_levels'])
        logger.info('...finished.')
        
        for key, value in zip(['rmse_mean', 'rmse_std', 'coverage'], [rmse_mean, rmse_std, coverage]):
            out_dict[key] = value
    
    elif params['method'] == 'lli_sg_mcmc_gauss':

        if params['data'] == 'synthetic':
            sigma_eps = PARAMS_SYNTH['sigma_eps']
            sigma_eps_sq = PARAMS_SYNTH['sigma_eps']**2
            batch_size = 5

        elif params['data'] == 'concrete':
            sigma_eps = PARAMS_CONCRETE['sigma_eps']
            sigma_eps_sq =PARAMS_CONCRETE['sigma_eps']**2
            batch_size = 5

        lli_net = load_backbone(params, logger)
        
        for param in lli_net.parameters():
            param.requires_grad = False
        with torch.no_grad():
            Psi = lli_net.get_ll_embedd(xs_train)

        gamma = math.log(100)/math.log(PARAMS_SG_MCMC['T']+1)

        w_samples, epsilon_ts =  train_sg_mcmc_gauss(Psi = Psi, 
                            lli_net = lli_net, 
                            dataloader_train = dataloader_train, 
                            batch_size = batch_size, 
                            sigma_eps_sq = sigma_eps_sq, 
                            a = PARAMS_SG_MCMC['a'],
                            b = PARAMS_SG_MCMC['b'], 
                            gamma = gamma,
                            T = PARAMS_SG_MCMC['T'],
                            warm_up = PARAMS_SG_MCMC['warm_up'])
        
        pred_mu, pred_std = predict_sg_mcmc_gauss(lli_net = lli_net, 
                              xs_val = xs_val, 
                              w_sample = w_samples, 
                              sigma_eps = sigma_eps,
                              epsilon_ts = torch.tensor(epsilon_ts))

        out_dict = {'pred_mu': pred_mu.detach().numpy(),
                    'pred_sigma': pred_std.detach().numpy(),
                    'w_samples': w_samples,
                    'epsilon_ts': epsilon_ts,
                    }
        
        coverage_sg_mcmc = get_coverage_gaussian(pred_mean = pred_mu.detach().numpy().squeeze(), 
                                    pred_std = pred_std.detach().numpy().squeeze(), 
                                    y_true = ys_val.detach().numpy().squeeze(), 
                                    levels=PARAMS_SYNTH['CI_levels' 
                                    ])
        out_dict['coverage'] = coverage_sg_mcmc

    elif params['method'] == 'lli_hmc_horseshoe':
        """We choose the following HMC parameters for the synthetic data
        and for the concrete data
        # number leapfrog steps: synth = 18
        # number iterations: synth T = 5000
        # step size: synth = 0.0045
        With these, the acceptance rate is approx. 65% as recommend for HMC."""

        if params['data'] == 'synthetic':
            T = 5000
            L = 18
            step_size = torch.tensor(0.0045)

        elif params['data'] == 'concrete':
            T = 8000
            L = 20
            step_size = torch.tensor(0.0015)

        lli_net = load_backbone(params, logger)

        for param in lli_net.parameters():
            param.requires_grad = False
        with torch.no_grad():
            Psi = lli_net.get_ll_embedd(xs_train)
            _, L = Psi.shape

        log_prob = make_log_prob_flat(ys_train, Psi, tau_sq=1)
        hmc = HMC(dim=2*L+1, T=T, L=L, step_size=step_size)
        hmc.register_log_prob(log_prob)
        x0 = torch.cat([torch.zeros(L), torch.zeros(L), torch.tensor([0.0])]) # Initial state

        samples, accept_prob, _ = hmc(x0)
        print("Acceptance probability:", accept_prob)

        w_samples, _, sigma_eps_sq_samples = unpack_samples(samples, L)

        with torch.no_grad():
            Psi_val = lli_net.get_ll_embedd(xs_val)
            pred_mu_samples = (Psi_val @  w_samples.T)
            pred_mu = pred_mu_samples.mean(dim = -1)
            pred_mus = Psi_val @ w_samples.T
            pred_std = torch.sqrt(torch.var(pred_mus, dim=-1) + sigma_eps_sq_samples.mean())  
        
        out_dict = {'pred_mu': pred_mu.detach().numpy(),
                    'pred_sigma': pred_std.detach().numpy(),
                    'w_samples': w_samples,
                    'sigma_eps_sq_samples': sigma_eps_sq_samples,
                    }

        # check again! 
        # I think we need to compute the coverage here differently
        coverage = get_coverage_gaussian(pred_mean = pred_mu.detach().numpy().squeeze(), 
                                         pred_std = pred_std.detach().numpy().squeeze(), 
                                         y_true = ys_val.detach().numpy().squeeze(), 
                                         levels=PARAMS_SYNTH['CI_levels'])
        out_dict['coverage'] = coverage
    
    else:
        ValueError('Please provide a valid method from: \n mc_dropout, bnn, ' \
                    'lli_vi_ridge, lli_vi_horseshoe,\n lli_vi_fac_ridge,' \
                    ' lli_vi_fac_horseshoe, lli_gibbs_ridge, sg_mcmc_gauss ')
        
    with open(params['outpath'] / f"out_dict_{params['method']}_{params['data']}.pkl", "wb") as f:
            pickle.dump(out_dict, f)

    logger.info(f"... everything saved under {params['outpath']}.")   

if __name__ == "__main__":
    main()
