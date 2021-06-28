"""
Auxiliary module for storage and easy calculation of statistics and metrics
that are provided by data generated by other modules, namely `caupo.cluster_tags`
"""

import argparse
from pathlib import Path

import pandas as pd

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


def calculate_average_silhouette(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, finds average silhouette data for a given frequency"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    data = data.loc[data["frequency"] == frequency]
    data["sil_score"] = data["sil_score"].apply(lambda x: "NaN" if str(x) == "None" else x).astype("float32")

    grouped_data = data[["algorithm", "embedder", "sil_score"]].groupby(["algorithm", "embedder"])
    return grouped_data.mean().sort_values(by=["sil_score"], ascending=False)


def calculate_consolidated_data(frequency: str, data: pd.DataFrame) -> pd.DataFrame:
    """Given raw result data, calculates a consolidated dataframe"""

    assert frequency in VALID_FREQUENCIES, "Unknown frequency value"

    avg_silhouette_scores = calculate_average_silhouette(frequency, data.copy())
    valid_entries = calculate_valid_entries(frequency, data.copy())

    return pd.concat([avg_silhouette_scores, valid_entries], axis=1).sort_values(
        by=["sil_score"], ascending=False)


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
    assert file_path.exists(), f"The file {file_path} does not exist"

    data = read_csv(file_path)

    # Get average of silhouette score
    consolidated_data = calculate_consolidated_data(args.frequency, data.copy())
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print("Avg. Silhouette Score & valid entries for each algorithm and embedding, over all entries")
        print(consolidated_data)
    consolidated_data.to_csv(output_file_path)


if __name__ == "__main__":
    main()
