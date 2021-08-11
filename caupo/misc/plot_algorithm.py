import argparse
import os
from typing import Callable, Dict, NoReturn

import numpy as np
import pandas as pd
import sklearn.datasets as data
from sklearn.cluster import KMeans

from caupo.utils import plot_clusters


def generate_dataset() -> np.ndarray:
    moons, _ = data.make_moons(n_samples=50, noise=0.05)
    blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75, 2.25), (1.0, 2.0)], cluster_std=0.25)
    return np.vstack([moons, blobs])


def kmeans(data: pd.DataFrame, output: str) -> None:
    model = KMeans(n_clusters=3)
    labels = model.fit_predict(data)

    plot_clusters(
        data,
        output,
        "Clústers obtenidos por K-Means (k=3, data sintética)",
        labels=labels,
        point_size=4,
    )


def get_algorithms() -> Dict[str, Callable[[str, str], NoReturn]]:
    return {
        'kmeans': kmeans,
    }


def main() -> None:
    algorithm_names = list(get_algorithms().keys())

    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", type=str, choices=algorithm_names)
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    handler = get_algorithms()[args.algorithm]
    data = generate_dataset()
    os.makedirs("outputs/plots/", exist_ok=True)
    handler(data, f"outputs/plots/{args.output}.png")


if __name__ == "__main__":
    main()
