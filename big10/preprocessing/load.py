import polars as pl
from typing import Tuple
from big10.constants import BASE_DIR

def normalize_stats(stats, exclude_columns=[]):
    new_columns = []
    for col in stats.columns:
        if col in exclude_columns:
            continue
        # Normalize and round to the nearest integer
        normalized_col = ((stats[col] - stats[col].min()) / (stats[col].max() - stats[col].min()) * 100).round()
        new_columns.append(normalized_col.alias(col))
    return stats.with_columns(new_columns)


def load_and_preprocess_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    stats = pl.read_parquet(f"{BASE_DIR}/data/stats_final.parquet")
    games = pl.read_parquet(f"{BASE_DIR}/data/games_all.parquet")
    drives = pl.read_parquet(f"{BASE_DIR}/data/drives_all.parquet")
    weather = pl.read_parquet(f"{BASE_DIR}/data/weather_all.parquet")
    winprobs = pl.read_parquet(f"{BASE_DIR}/data/win_probs.parquet")
    
    stats = normalize_stats(stats, exclude_columns=["team", "season", "week", "conference"])
    
    return stats, games, drives, weather, winprobs