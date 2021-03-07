# caupo
Cluster Analysis of Unsupervised Political Opinions - Undergraduate senior thesis @ Universidad Simón Bolívar

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
