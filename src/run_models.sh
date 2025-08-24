#!/bin/bash

# synthetic data
# run lli_closed_form first to obtain the backbone for all LLI models and training data
poetry run python ./run_models.py --method lli_closed_form --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic
poetry run python ./run_models.py --method mc_dropout --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic
poetry run python ./run_models.py --method bnn --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic
poetry run python ./run_models.py --method lli_gibbs_ridge --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic

poetry run python ./run_models.py --method lli_vi_closed_form --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic
poetry run python ./run_models.py --method lli_vi_ridge --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic
poetry run python ./run_models.py --method lli_vi_horseshoe --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic

poetry run python ./run_models.py --method lli_vi_ridge_full_fac --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic
poetry run python ./run_models.py --method lli_vi_horseshoe_full_fac --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic

poetry run python ./run_models.py --method lli_sg_mcmc_gauss --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic
poetry run python ./run_models.py --method lli_hmc_horseshoe --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic

# concrete compressive strength
poetry run python ./run_models.py --method lli_closed_form --outpath ./results --model_dims 8 200 200 100 50 1 --data concrete
poetry run python ./run_models.py --method mc_dropout --outpath ./results --model_dims 8 200 200 100 50 1 --data concrete
poetry run python ./run_models.py --method bnn --outpath ./results --model_dims 8 200 200 100 50 1 --data concrete
poetry run python ./run_models.py --method lli_gibbs_ridge --outpath ./results --model_dims 8 200 200 100 50 1 --data concrete

poetry run python ./run_models.py --method lli_vi_closed_form --outpath ./results --model_dims 8 200 200 100 50 1  --data concrete
poetry run python ./run_models.py --method lli_vi_ridge --outpath ./results --model_dims 8 200 200 100 50 1 --data concrete
poetry run python ./run_models.py --method lli_vi_horseshoe --outpath ./results --model_dims 8 200 200 100 50 1 --data concrete

poetry run python ./run_models.py --method lli_vi_ridge_full_fac --outpath ./results --model_dims 8 200 200 100 50 1 --data concrete
poetry run python ./run_models.py --method lli_vi_horseshoe_full_fac --outpath ./results --model_dims 8 200 200 100 50 1 --data concrete

poetry run python ./run_models.py --method lli_sg_mcmc_gauss --outpath ./results --model_dims 8 200 200 100 50 1 --data concrete
poetry run python ./run_models.py --method lli_hmc_horseshoe --outpath ./results --model_dims 8 200 200 100 50 1 --data concrete