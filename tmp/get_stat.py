from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Action semantics in this dataset:
# 1 = exposure without click
# 2 = click
IMPRESSION_ACTION = 1
CLICK_ACTION = 2


def _safe_get(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, dict):
        return entry.get(key, default)
    try:
        value = entry[key]
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        return value
    except Exception:
        return default


def _to_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        if value.shape == ():
            return int(value.item())
        return int(value.reshape(-1)[0])
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        if value.shape == ():
            return float(value.item())
        return float(value.reshape(-1)[0])
    try:
        return float(value)
    except Exception:
        return default


def _as_int_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.int64)
    arr = np.asarray(value, dtype=np.int64)
    if arr.ndim == 0:
        return np.array([int(arr)], dtype=np.int64)
    return arr


def _as_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=np.float32)
    return arr


def _iter_entries(value: Any):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return value
    return []


def _extract_user_id_int(series: pd.Series) -> pd.Series:
    """Convert user ids like 'user_123' to integer 123 without building a mapping table."""
    extracted = series.astype(str).str.extract(r"(\d+)$", expand=False)
    user_id_int = pd.to_numeric(extracted, errors="coerce")
    return user_id_int.astype("Int64")


def _count_actions(label_arr: Any) -> tuple[int, int]:
    """Count exposure and click actions in one row's label list."""
    impressions = 0
    clicks = 0

    if label_arr is None:
        return impressions, clicks

    if not isinstance(label_arr, (list, tuple, np.ndarray, pd.Series)):
        return impressions, clicks

    for e in label_arr:
        if not isinstance(e, dict):
            continue

        at_raw = e.get("action_type")
        if at_raw is None:
            continue

        at = int(at_raw)
        if at == IMPRESSION_ACTION:
            impressions += 1
        elif at == CLICK_ACTION:
            clicks += 1

    return impressions, clicks


def build_item_global_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build item-level global stats for downstream training usage."""
    action_counts = df["label"].apply(_count_actions)
    cnt_df = pd.DataFrame(
        action_counts.tolist(), columns=["impression_count", "click_count"]
    )

    work = pd.concat(
        [df[["item_id", "timestamp"]].reset_index(drop=True), cnt_df], axis=1
    )

    item_stats = (
        work.groupby("item_id", as_index=False)
        .agg(
            impression_count=("impression_count", "sum"),
            click_count=("click_count", "sum"),
            last_tt=("timestamp", "max"),
        )
        .sort_values(
            ["impression_count", "click_count", "item_id"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    denom = item_stats["impression_count"] + item_stats["click_count"]
    item_stats["ctr"] = np.where(denom > 0, item_stats["click_count"] / denom, 0.0)
    return item_stats


def build_prepared_data(df: pd.DataFrame) -> pd.DataFrame:
    """Create lightweight table input for dataset with integer user id."""
    prepared = df.copy()
    prepared["user_id"] = _extract_user_id_int(prepared["user_id"])

    # Keep row-level click label for simple training usage.
    prepared["is_click"] = prepared["label"].apply(
        lambda arr: int(
            any(int(e.get("action_type")) == CLICK_ACTION for e in (arr or []))
        )
    )
    prepared["action_time"] = prepared["label"].apply(
        lambda arr: max((int(e.get("action_time", 0)) for e in (arr or [])), default=0)
    )
    prepared.drop(columns=["label"], inplace=True)
    return prepared


def build_feature_meta(df: pd.DataFrame) -> dict[str, Any]:
    """Dynamically extract FEATURE_META that matches current dataset/get_features API."""
    item_sparse_max: dict[int, int] = defaultdict(int)
    item_dense: dict[int, dict[str, Any]] = {}
    item_dense_values: dict[int, list[float]] = defaultdict(list)
    item_multihot_max: dict[int, int] = defaultdict(int)

    user_sparse_max: dict[int, int] = defaultdict(int)
    user_multihot_max: dict[int, int] = defaultdict(int)
    user_embedding_dims: dict[int, int] = defaultdict(int)
    user_weighted_multihot_max: dict[int, int] = defaultdict(int)

    seq_feature_ids: dict[str, set[int]] = {
        "action_seq": set(),
        "content_seq": set(),
        "item_seq": set(),
    }
    seq_max_vals: dict[str, dict[int, int]] = {
        "action_seq": defaultdict(int),
        "content_seq": defaultdict(int),
        "item_seq": defaultdict(int),
    }
    seq_lens: dict[str, list[int]] = {
        "action_seq": [],
        "content_seq": [],
        "item_seq": [],
    }

    for row in df.itertuples(index=False):
        # item_feature
        for e in _iter_entries(getattr(row, "item_feature", None)):
            fid = _to_int(_safe_get(e, "feature_id"), default=-1)
            if fid < 0:
                continue
            int_value = _safe_get(e, "int_value")
            float_value = _safe_get(e, "float_value")
            int_array = _as_int_array(_safe_get(e, "int_array"))

            if int_value is not None:
                item_sparse_max[fid] = max(item_sparse_max[fid], _to_int(int_value, 0))
            elif float_value is not None:
                item_dense[fid] = {"type": "float_value"}
                item_dense_values[fid].append(_to_float(float_value, default=0.0))
            elif int_array.size > 0:
                item_multihot_max[fid] = max(
                    item_multihot_max[fid], int(int_array.max())
                )

        # user_feature
        for e in _iter_entries(getattr(row, "user_feature", None)):
            fid = _to_int(_safe_get(e, "feature_id"), default=-1)
            if fid < 0:
                continue
            int_value = _safe_get(e, "int_value")
            int_array = _as_int_array(_safe_get(e, "int_array"))
            float_array = _as_float_array(_safe_get(e, "float_array"))

            if int_value is not None and int_array.size == 0 and float_array.size == 0:
                user_sparse_max[fid] = max(user_sparse_max[fid], _to_int(int_value, 0))
            elif int_array.size > 0 and float_array.size > 0:
                user_weighted_multihot_max[fid] = max(
                    user_weighted_multihot_max[fid], int(int_array.max())
                )
            elif int_array.size > 0:
                user_multihot_max[fid] = max(
                    user_multihot_max[fid], int(int_array.max())
                )
            elif float_array.size > 0:
                user_embedding_dims[fid] = max(
                    user_embedding_dims[fid], int(float_array.shape[0])
                )

        # seq_feature
        seq_feature = getattr(row, "seq_feature", None)
        for seq_name in ("action_seq", "content_seq", "item_seq"):
            entries = _iter_entries(_safe_get(seq_feature, seq_name, []))
            for e in entries:
                fid = _to_int(_safe_get(e, "feature_id"), default=-1)
                if fid < 0:
                    continue
                arr = _as_int_array(_safe_get(e, "int_array"))
                seq_feature_ids[seq_name].add(fid)
                seq_lens[seq_name].append(int(arr.shape[0]))
                if arr.size > 0:
                    seq_max_vals[seq_name][fid] = max(
                        seq_max_vals[seq_name][fid], int(arr.max())
                    )

    def _stats(values: list[int]) -> dict[str, float | int]:
        if len(values) == 0:
            return {"min": 0, "max": 0, "mean": 0.0, "p50": 0, "p95": 0}
        arr = np.asarray(values, dtype=np.int64)
        return {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "p50": int(np.quantile(arr, 0.5)),
            "p95": int(np.quantile(arr, 0.95)),
        }

    def _dense_binning(values: list[float], num_bins: int = 32) -> dict[str, Any]:
        if len(values) == 0:
            return {
                "enabled": False,
                "use_log1p": False,
                "bin_edges": [],
                "num_bins": 1,
            }

        arr = np.asarray(values, dtype=np.float64)
        if arr.shape[0] <= 1:
            skewness = 0.0
        else:
            centered = arr - arr.mean()
            denom = np.power(arr.std() + 1e-12, 3)
            skewness = float((centered**3).mean() / denom)

        use_log1p = bool(np.all(arr >= 0.0) and skewness > 1.0)
        transformed = np.log1p(arr) if use_log1p else arr
        edges = np.quantile(transformed, np.linspace(0.0, 1.0, num_bins + 1))
        edges = np.unique(edges)

        if edges.shape[0] <= 1:
            return {
                "enabled": True,
                "use_log1p": use_log1p,
                "bin_edges": [],
                "num_bins": 1,
            }

        return {
            "enabled": True,
            "use_log1p": use_log1p,
            "bin_edges": [float(x) for x in edges.tolist()],
            "num_bins": int(edges.shape[0] - 1),
        }

    for fid in list(item_dense.keys()):
        item_dense[fid]["binning"] = _dense_binning(item_dense_values[fid], num_bins=32)

    feature_meta = {
        "item_feature": {
            "sparse": {
                int(fid): {"vocab_size": int(max_v) + 1}
                for fid, max_v in sorted(item_sparse_max.items())
            },
            "dense": {int(fid): meta for fid, meta in sorted(item_dense.items())},
            "multihot": {
                int(fid): {"vocab_size": int(max_v) + 1}
                for fid, max_v in sorted(item_multihot_max.items())
            },
        },
        "user_feature": {
            "sparse": {
                int(fid): {"vocab_size": int(max_v) + 1}
                for fid, max_v in sorted(user_sparse_max.items())
            },
            "multihot": {
                int(fid): {"vocab_size": int(max_v) + 1}
                for fid, max_v in sorted(user_multihot_max.items())
            },
            "embedding": {
                int(fid): {"type": "float_array", "dim": int(dim)}
                for fid, dim in sorted(user_embedding_dims.items())
            },
            "weighted_multihot": {
                int(fid): {"vocab_size": int(max_v) + 1}
                for fid, max_v in sorted(user_weighted_multihot_max.items())
            },
        },
        "seq_feature": {
            seq_name: {
                "feature_ids": sorted(int(fid) for fid in seq_feature_ids[seq_name]),
                "value_type": "int_array",
                "seq_len_stats": _stats(seq_lens[seq_name]),
                "max_vals": {
                    int(fid): int(max_v)
                    for fid, max_v in sorted(seq_max_vals[seq_name].items())
                },
            }
            for seq_name in ("action_seq", "content_seq", "item_seq")
        },
        "api_fit": {
            "item_id_vocab_size": int(df["item_id"].nunique()) + 1,
            "delta_tt_bucket_size": 64,
            "time_shared_embedding_anchor": "action_tt_hour",
        },
    }
    return feature_meta



def main() -> None:
    root_path = Path(__file__).parent.parent.parent
    input_path = Path(root_path / "data/raw/sample_data.parquet")
    out_dir = Path(root_path / "data/processed/")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)

    # prepared = build_prepared_data(df)
    item_stats = build_item_global_stats(df)
    feature_meta = build_feature_meta(df)

    # prepared_parquet = out_dir / "prepared_data.parquet"
    item_stats_parquet = out_dir / "item_global_stats.parquet"
    feature_meta_json = out_dir / "feature_meta.json"
    summary_json = out_dir / "global_summary.json"

    # prepared.to_parquet(prepared_parquet, index=False)
    item_stats.to_parquet(item_stats_parquet, index=False)
    with feature_meta_json.open("w", encoding="utf-8") as f:
        json.dump(feature_meta, f, ensure_ascii=False, indent=2)

    summary = {
        "n_rows": len(df),
        "n_items": int(df["item_id"].nunique()),
        "n_users_raw": int(df["user_id"].nunique()),
        # "n_users_int_non_null": int(prepared["user_id"].notna().sum()),
        "global_impression_actions": int(
            sum(_count_actions(arr)[0] for arr in df["label"])
        ),
        "global_click_actions": int(sum(_count_actions(arr)[1] for arr in df["label"])),
        "global_ctr": float(
            item_stats["click_count"].sum()
            / (item_stats["impression_count"].sum() + item_stats["click_count"].sum())
        ),
        "output_files": {
            # "prepared_data_parquet": str(prepared_parquet),
            "item_global_stats_parquet": str(item_stats_parquet),
            "feature_meta_json": str(feature_meta_json),
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Loaded rows: {len(df)}")
    # print(f"Prepared data saved to: {prepared_parquet}")
    print(f"Item stats saved to: {item_stats_parquet}")
    print(f"Feature meta saved to: {feature_meta_json}")
    print(f"Summary saved to: {summary_json}")


if __name__ == "__main__":
    main()
