"""
Auxiliary module for storage and easy calculation of statistics and metrics
that are provided by data generated by other modules, namely `caupo.cluster_tags`
"""

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import ludovico

from caupo.embeddings import get_embedder_function_short_names

VALID_FREQUENCIES = [
    'daily',
    'weekly',
    'monthly',
]


def calculate_valid_entries(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, finds amount of valid entries"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    data = data.loc[data["frequency"] == frequency]
    data["valid_entries"] = data["sil_score"].apply(lambda x: "NaN" if str(x) == "None" else x).astype("float32")
    data.dropna()

    grouped_data = data[["algorithm", "embedder", "valid_entries"]].groupby(["algorithm", "embedder"])
    return grouped_data.count()


def calculate_average_n_clusters(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, finds average n_clusters data for a given frequency"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    data = data.loc[data["frequency"] == frequency]

    grouped_data = data[["algorithm", "embedder", "n_clusters"]].groupby(["algorithm", "embedder"])
    return grouped_data.mean().sort_values(by=["n_clusters"], ascending=False)


def calculate_average_noise_percentage(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, finds average noise_percentage data for a given frequency"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    data = data.loc[data["frequency"] == frequency]

    grouped_data = data[["algorithm", "embedder", "noise_percentage"]].groupby(["algorithm", "embedder"])
    return grouped_data.mean().sort_values(by=["noise_percentage"], ascending=False)


def calculate_average_avg_cluster_size(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, finds average avg_cluster_size data for a given frequency"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    data = data.loc[data["frequency"] == frequency]

    grouped_data = data[["algorithm", "embedder", "avg_cluster_size"]].groupby(["algorithm", "embedder"])
    return grouped_data.mean().sort_values(by=["avg_cluster_size"], ascending=False)


def calculate_average_silhouette(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, finds average silhouette data for a given frequency"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    data = data.loc[data["frequency"] == frequency]
    data["sil_score"] = data["sil_score"].apply(lambda x: "NaN" if str(x) == "None" else x).astype("float32")

    grouped_data = data[["algorithm", "embedder", "sil_score"]].groupby(["algorithm", "embedder"])
    return grouped_data.mean().sort_values(by=["sil_score"], ascending=False)


def calculate_average_davies_bouldin(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, finds average davies bouldin data for a given frequency"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    data = data.loc[data["frequency"] == frequency]
    data["db_score"] = data["db_score"].apply(lambda x: "NaN" if str(x) == "None" else x).astype("float32")

    grouped_data = data[["algorithm", "embedder", "db_score"]].groupby(["algorithm", "embedder"])
    return grouped_data.mean().sort_values(by=["db_score"], ascending=False)


def calculate_average_calinski_harabasz(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, finds average davies bouldin data for a given frequency"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    data = data.loc[data["frequency"] == frequency]
    data["ch_score"] = data["ch_score"].apply(lambda x: "NaN" if str(x) == "None" else x).astype("float32")

    grouped_data = data[["algorithm", "embedder", "ch_score"]].groupby(["algorithm", "embedder"])
    return grouped_data.mean().sort_values(by=["ch_score"], ascending=False)


def calculate_consolidated_data(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, calculates a consolidated dataframe"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    avg_silhouette_scores = calculate_average_silhouette(frequency, data.copy())
    valid_entries = calculate_valid_entries(frequency, data.copy())

    consolidated = pd.concat([avg_silhouette_scores, valid_entries], axis=1)

    max_entries_value = np.max(consolidated['valid_entries'].tolist())
    consolidated["weighted_score"] = (consolidated["sil_score"] * consolidated["valid_entries"]) / max_entries_value

    return consolidated.sort_values(by=["weighted_score", "sil_score"], ascending=False)


def consolidate_three_averages(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, consolidates the average of the three measurements"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    avg_silhouette_scores = calculate_average_silhouette(frequency, data.copy())
    avg_davies_bouldin = calculate_average_davies_bouldin(frequency, data.copy())
    avg_calinski_harabasz = calculate_average_calinski_harabasz(frequency, data.copy())

    consolidated = pd.concat(
        [avg_silhouette_scores, avg_davies_bouldin, avg_calinski_harabasz],
        axis=1
    ).round(3).reset_index().rename(
        columns={
            'embedder': 'Modelo',
            'algorithm': 'Algoritmo',
            'sil_score': 'Silueta',
            'db_score': 'Davies-Bouldin',
            'ch_score': 'Calinski-Harabasz',
        }
    )
    short_names = get_embedder_function_short_names()
    consolidated["Modelo"] = [
        short_names[modelo]
        for modelo in consolidated["Modelo"].tolist()
    ]

    return consolidated.sort_values(by=["Silueta"], ascending=False)


def consolidate_three_weighted_averages(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw result data, consolidates the weighted average of the three measurements
    according to the valid entries they did
    """

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    avg_silhouette_scores = calculate_average_silhouette(frequency, data.copy())
    avg_davies_bouldin = calculate_average_davies_bouldin(frequency, data.copy())
    avg_calinski_harabasz = calculate_average_calinski_harabasz(frequency, data.copy())
    valid_entries = calculate_valid_entries(frequency, data.copy())

    consolidated = pd.concat(
        [avg_silhouette_scores, avg_davies_bouldin, avg_calinski_harabasz, valid_entries],
        axis=1
    ).reset_index()

    max_entries_value = np.max(consolidated['valid_entries'].tolist())
    consolidated["sil_score"] = (consolidated["sil_score"] * consolidated["valid_entries"]) / max_entries_value
    consolidated["db_score"] = (consolidated["db_score"] * consolidated["valid_entries"]) / max_entries_value
    consolidated["ch_score"] = (consolidated["ch_score"] * consolidated["valid_entries"]) / max_entries_value

    consolidated = consolidated.rename(
        columns={
            'embedder': 'Modelo',
            'algorithm': 'Algoritmo',
            'sil_score': 'Silueta',
            'db_score': 'Davies-Bouldin',
            'ch_score': 'Calinski-Harabasz',
            'valid_entries': 'Resultados válidos',
        }
    )
    short_names = get_embedder_function_short_names()
    consolidated["Modelo"] = [
        short_names[modelo]
        for modelo in consolidated["Modelo"].tolist()
    ]

    return consolidated.sort_values(by=["Silueta"], ascending=False).round(3)


def consolidate_cluster_nature_values(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """
    Given raw result data, consolidates the weighted average of the three measurements
    according to the valid entries they did
    """

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    avg_n_clusters = calculate_average_n_clusters(frequency, data.copy())
    avg_noise_percentage = calculate_average_noise_percentage(frequency, data.copy())
    avg_cluster_size = calculate_average_avg_cluster_size(frequency, data.copy())

    consolidated = pd.concat(
        [avg_n_clusters, avg_noise_percentage, avg_cluster_size],
        axis=1
    ).reset_index()

    consolidated = consolidated.rename(
        columns={
            'embedder': 'Modelo',
            'algorithm': 'Algoritmo',
            'n_clusters': 'Cantidad de clústers',
            'noise_percentage': 'Ruido (%)',
            'cluster_size': 'Tamaño de clústers',
        }
    )
    short_names = get_embedder_function_short_names()
    consolidated["Modelo"] = [
        short_names[modelo]
        for modelo in consolidated["Modelo"].tolist()
    ]

    return consolidated.round(3)


def read_csv(file_path: Path) -> pd.DataFrame:
    """Given a path to a file, reads the file and returns a dataframe"""

    return pd.read_csv(file_path)


def main() -> None:
    """Read input arguments and calculates and returns results"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", metavar="FREQUENCY", type=str, default="daily",
                        choices=VALID_FREQUENCIES)
    args = parser.parse_args()
    print(f"Received frequency `{args.frequency}`")

    file_path = Path(f"outputs/cluster_tags/{args.frequency}/results.csv")
    output_file_path = Path(f"outputs/cluster_tags/{args.frequency}/aggregated_results.csv")
    output_table_file_path = Path(f"outputs/cluster_tags/{args.frequency}/tables.txt")
    assert file_path.exists(), f"The file {file_path} does not exist"

    print("Reading data")
    data = read_csv(file_path)

    print("Eliminating DBSCAN from data")
    data = data[data["algorithm"] != "DBSCAN"]

    print("Eliminating 50 dimension models from data")
    data = data[~data["embedder"].str.contains("50")]
    frequency_name = "diaria" if args.frequency == 'daily' else 'mensual'

    # Get average of silhouette score
    consolidated_data = calculate_consolidated_data(args.frequency, data.copy())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Avg. Silhouette Score & valid entries for each algorithm and embedding, over all entries")
        print(consolidated_data)
    consolidated_data.to_csv(output_file_path)

    table_list: List[str] = []

    # Get consolidated table with three measurements
    consolidated_three_averages_data = consolidate_three_averages(args.frequency, data.copy())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Avg metrics for each algorithm and embedding, over all entries")
        print(consolidated_three_averages_data)
    print(f"Printing TeX table for Three averages with frequency={args.frequency}")
    table_three_averages = ludovico.generate_comparison_for_two_columns(
        consolidated_three_averages_data,
        "Modelo",
        "Algoritmo",
        ["Silueta", "Davies-Bouldin", "Calinski-Harabasz"],
        add_hlines=True,
        data_highlight={
            'Silueta': 'max',
            'Davies-Bouldin': 'min',
            'Calinski-Harabasz': 'max',
        },
        table_width=1,
        table_label="tabla_tres_metricas",
        table_name=(
            "Promedio de métricas de validación interna según configuración "
            f"experimental con frecuencia {frequency_name}"
        ),
        table_long_name=(
            "Promedio de métricas de validación interna (coeficiente de silueta, "
            "coeficiente de Davies-Bouldin y coeficiente de Calinski-Harabasz) según "
            f"algoritmo y modelo utilizados con frecuencia {frequency_name}."
        )
    )
    table_list.append(table_three_averages)
    print(table_three_averages)

    # Get consolidated table with three weighted measurements
    consolidated_three_weighted_averages_data = consolidate_three_weighted_averages(args.frequency, data.copy())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Weighted avg metrics for each algorithm and embedding, over all entries")
        print(consolidated_three_weighted_averages_data)
    print(f"Printing TeX table for Three weighted averages with frequency={args.frequency}")
    table_three_weighted_averages = ludovico.generate_comparison_for_two_columns(
        consolidated_three_weighted_averages_data,
        "Modelo",
        "Algoritmo",
        ["Silueta", "Davies-Bouldin", "Calinski-Harabasz", "Resultados válidos"],
        add_hlines=True,
        data_highlight={
            'Silueta': 'max',
            'Davies-Bouldin': 'min',
            'Calinski-Harabasz': 'max',
            'Resultados válidos': 'max',
        },
        table_width=1,
        table_label="tabla_tres_metricas_ponderadas",
        table_name=(
            "Promedio ponderado de métricas de validación interna por resultados válidos "
            f"según configuración experimental con frecuencia {frequency_name}"
        ),
        table_long_name=(
            "Promedio ponderado de métricas de validación interna (coeficiente de silueta, "
            "coeficiente de Davies-Bouldin y coeficiente de Calinski-Harabasz) por resultados válidos"
            f" según algoritmo y modelo utilizados con frecuencia {frequency_name}."
        )
    )
    table_list.append(table_three_weighted_averages)
    print(table_three_weighted_averages)

    # Get consolidated table with cluster nature measurements
    consolidated_cluster_data = consolidate_cluster_nature_values(args.frequency, data.copy())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Cluster nature metrics for each algorithm and embedding, over all entries")
        print(consolidated_cluster_data)
    print(f"Printing TeX table for Cluster nature metrics with frequency={args.frequency}")
    table_cluster_data = ludovico.generate_comparison_for_two_columns(
        consolidated_cluster_data,
        "Modelo",
        "Algoritmo",
        ["Cantidad de clústers", "Tamaño de clústers", "Ruido (%)",],
        add_hlines=True,
        table_width=1,
        table_label="tabla_nat_clusters",
        table_name=(
            "Promedios de cantidad de clústers, tamaño de clústers, y porcentaje de ruido "
            f"según configuración experimental con frecuencia {frequency_name}"
        ),
        table_long_name=(
            "Promedios de cantidad de clústers, tamaño de clústers, y porcentaje de ruido"
            f" según algoritmo y modelo utilizados con frecuencia {frequency_name}."
        )
    )
    table_list.append(table_cluster_data)
    print(table_cluster_data)

    # Storing tables
    with open(output_table_file_path, "w") as file_handler:
        file_handler.writelines([f"{table}\n\n" for table in table_list])

if __name__ == "__main__":
    main()
