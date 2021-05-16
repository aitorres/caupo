#!/bin/bash

echo "*** Activating virtual environment ***"
source ./env/bin/activate
printf "***Virtual environment activated***\n\n"

echo "*** Locking Twitter fetch script ***"
python -m caupo.database lock
printf "*** Twitter fetch script locked ***\n\n"

echo "*** Running measurements on `date` ***"
nohup nice -n -19 python -u -m caupo.initial_experiments.measure_embeddings
nohup nice -n -19 python -u -m caupo.initial_experiments.measure_proper_k_for_embeddings
nohup nice -n -19 python -u -m caupo.initial_experiments.measure_possible_eps_values_for_embeddings
nohup nice -n -19 python -u -m caupo.initial_experiments.measure_hyperparameters_for_dbscan
nohup nice -n -19 python -u -m caupo.initial_experiments.measure_optics_clusters
nohup nice -n -19 python -u -m caupo.initial_experiments.measure_hdbscan_clusters
nohup nice -n -19 python -u -m caupo.initial_experiments.measure_mean_shift_clusters
nohup nice -n -19 python -u -m caupo.initial_experiments.measure_affinity_propagation_clusters
printf "*** Measurements finished on `date` ***\n\n"

echo "*** Unlocking Twitter fetch script***"
python -m caupo.database unlock
printf "*** Twitter fetch script unlocked ***\n"
