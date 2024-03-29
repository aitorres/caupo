# CAUPO

Cluster Analysis of Unsupervised Political Opinions - Undergraduate senior thesis @ Universidad Simón Bolívar

## Order of initial experiments

In order to reproduce the workflow of initial experiments in the project, you can follow this suggested order (although, since each experiment is independent from others, you can run just one, or several in any order you want):

1. `measure_embeddings.py`
2. `measure_proper_k_for_embeddings.py`
3. `measure_possible_eps_values_for_embeddings.py`
4. `measure_hyperparameters_for_dbscan.py`
5. `measure_optics_clusters.py`
6. `measure_hdbscan_clusters.py`
7. `measure_mean_shift_clusters.py`
8. `measure_affinity_propagation_clusters.py`

You can run the following script which has been added to the repository in order to easen up each measurement run with some extra considerations, such as turning off Twitter fetching while tests run

```bash
nohup nice -n -19 ./run_initial_experiments.sh &
```

## Things to remember

To run Python scripts that are costly / resource intensive:

- Use `nohup` so that internet interruptions don't... interrupt your script (delete previous `nohup.out` files)
- Use `nice -n -19` to give your script maximum priority (unless you don't need to)
- Use `python -u` to disable the stdout buffer and allow for exceptions and errors to be recorded immediately in log files
- Use `&` to daemonize your script

Example:

```bash
nohup nice -n -19 python -u measure_embeddings.y &
```

## Interesting links

- MongoDB cheat sheet: https://gist.github.com/bradtraversy/f407d642bdc3b31681bc7e56d95485b6

## Run the API

To run the API (back-end), use this command:

```bash
gunicorn -w 2 --threads 2 --preload --bind 0.0.0.0:5000 backend.app:app
```

Adjust the port and other parameters of Gunicorn as needed.
