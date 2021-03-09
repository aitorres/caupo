#!/bin/bash

echo "*** Activating virtual environment ***"
source ../env/bin/activate
printf "***Virtual environment activated***\n\n"

echo "*** Locking Twitter fetch script ***"
python database.py lock
printf "*** Twitter fetch script locked ***\n\n"

echo "*** Running measurements on `date` ***"
nohup nice -n -19 python -u measure_embeddings.py
nohup nice -n -19 python -u measure_proper_k_for_embeddings.py
nohup nice -n -19 python -u measure_possible_eps_values_for_embeddings.py
nohup nice -n -19 python -u measure_hyperparameters_for_dbscan.py
nohup nice -n -19 python -u measure_optics_clusters.py
nohup nice -n -19 python -u measure_hdbscan_clusters.py
nohup nice -n -19 python -u measure_mean_shift_clusters.py
nohup nice -n -19 python -u measure_affinity_propagation_clusters.py
printf "*** Measurements finished on `date` ***\n\n"

echo "*** Unlocking Twitter fetch script***"
python database.py unlock
printf "*** Twitter fetch script unlocked ***\n"
