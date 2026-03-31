import pandas as pd
from scipy.stats import spearmanr, pearsonr
import os
import json

from utils.config import EVALUATION_RESULTS_PATH


def calculate_intrinsic_correlations(df: pd.DataFrame) -> dict:
    # Drop "Alibaba-NLP__gte-multilingual-base" and "jinaai__jina-embeddings-v2-small-en" rows
    df = df[~df["Model"].isin(["Alibaba-NLP__gte-multilingual-base", "jinaai__jina-embeddings-v2-small-en"])]

    # Keep only the mteb and intrinsic metric columns, in order to calculate the correlations
    df_mteb_angular_loss = df[["**AVG_MTEB**", "angular_loss"]].dropna()
    df_mteb_angular_loss_weighted = df[["**AVG_MTEB**", "angular_loss_weighted"]].dropna()

    df_mteb_positional_loss = df[["**AVG_MTEB**", "positional_loss"]].dropna()
    df_mteb_positional_loss_weighted = df[["**AVG_MTEB**", "positional_loss_weighted"]].dropna()

    df_mteb_spearman_loss = df[["**AVG_MTEB**", "spearman_loss"]].dropna()
    df_mteb_spearman_loss_weighted = df[["**AVG_MTEB**", "spearman_loss_weighted"]].dropna()

    # Calculate Spearman and Pearson correlations for angular_loss
    spearman_angular_score = spearmanr(
        df_mteb_angular_loss["**AVG_MTEB**"].values, 
        df_mteb_angular_loss["angular_loss"].values
    )
    pearson_angular_score = pearsonr(
        df_mteb_angular_loss["**AVG_MTEB**"].values, 
        df_mteb_angular_loss["angular_loss"].values
    )

    # Calculate Spearman and Pearson correlations for angular_loss_weighted
    spearman_angular_score_weighted = spearmanr(
        df_mteb_angular_loss_weighted["**AVG_MTEB**"].values, 
        df_mteb_angular_loss_weighted["angular_loss_weighted"].values
    )
    pearson_angular_score_weighted = pearsonr(
        df_mteb_angular_loss_weighted["**AVG_MTEB**"].values, 
        df_mteb_angular_loss_weighted["angular_loss_weighted"].values
    )

    # Calculate Spearman and Pearson correlations for positional_loss
    spearman_positional_score = spearmanr(
        df_mteb_positional_loss["**AVG_MTEB**"].values, 
        df_mteb_positional_loss["positional_loss"].values
    )
    pearson_positional_score = pearsonr(
        df_mteb_positional_loss["**AVG_MTEB**"].values, 
        df_mteb_positional_loss["positional_loss"].values
    )

    # Calculate Spearman and Pearson correlations for positional_loss_weighted
    spearman_positional_score_weighted = spearmanr(
        df_mteb_positional_loss_weighted["**AVG_MTEB**"].values, 
        df_mteb_positional_loss_weighted["positional_loss_weighted"].values
    )
    pearson_positional_score_weighted = pearsonr(
        df_mteb_positional_loss_weighted["**AVG_MTEB**"].values, 
        df_mteb_positional_loss_weighted["positional_loss_weighted"].values
    )

    # Calculate Spearman and Pearson correlations for spearman_loss
    spearman_spearman_score = spearmanr(
        df_mteb_spearman_loss["**AVG_MTEB**"].values, 
        df_mteb_spearman_loss["spearman_loss"].values
    )
    pearson_spearman_score = pearsonr(
        df_mteb_spearman_loss["**AVG_MTEB**"].values, 
        df_mteb_spearman_loss["spearman_loss"].values
    )

    # Calculate Spearman and Pearson correlations for spearman_loss_weighted
    spearman_spearman_score_weighted = spearmanr(
        df_mteb_spearman_loss_weighted["**AVG_MTEB**"].values, 
        df_mteb_spearman_loss_weighted["spearman_loss_weighted"].values
    )
    pearson_spearman_score_weighted = pearsonr(
        df_mteb_spearman_loss_weighted["**AVG_MTEB**"].values, 
        df_mteb_spearman_loss_weighted["spearman_loss_weighted"].values
    )

    return {
        "angular_loss": {
            "spearman": spearman_angular_score.statistic,
            "pearson": pearson_angular_score.statistic
        },
        "angular_loss_weighted": {
            "spearman": spearman_angular_score_weighted.statistic,
            "pearson": pearson_angular_score_weighted.statistic
        },
        "positional_loss": {
            "spearman": spearman_positional_score.statistic,
            "pearson": pearson_positional_score.statistic
        },
        "positional_loss_weighted": {
            "spearman": spearman_positional_score_weighted.statistic,
            "pearson": pearson_positional_score_weighted.statistic
        },
        "spearman_loss": {
            "spearman": spearman_spearman_score.statistic,
            "pearson": pearson_spearman_score.statistic
        },
        "spearman_loss_weighted": {
            "spearman": spearman_spearman_score_weighted.statistic,
            "pearson": pearson_spearman_score_weighted.statistic
        }
    }


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(EVALUATION_RESULTS_PATH, "comparison_results.csv"))
    correlations = calculate_intrinsic_correlations(df)

    with open(os.path.join(EVALUATION_RESULTS_PATH, "intrinsic_correlations.json"), "w") as f:
        json.dump(correlations, f, indent=4)
        