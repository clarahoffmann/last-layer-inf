#!/bin/bash

poetry run python ./run_models.py --method lli_vi_closed_full_cov --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic
poetry run python ./run_models.py --method mc_dropout --outpath ./results --model_dims 1 100 100 100 50 1 --data synthetic