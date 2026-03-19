from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "sample_data.parquet"
PROCESSED_DIR = DATA_DIR / "processed"
PREPARED_PATH = PROCESSED_DIR / "prepared_train.parquet"
ITEM_STATS_PATH = PROCESSED_DIR / "item_stats.parquet"
FEATURE_META_PATH = PROCESSED_DIR / "feature_meta.json"
SUMMARY_PATH = PROCESSED_DIR / "summary.json"

ACTION_REV = True
CONTENT_REV = True
ITEM_REV = True
SEQ_TT_ID = {
    "action_seq": 28,
    "content_seq": 41,
    "item_seq": 29,
}


def _extract_tt(timestamp: int) -> tuple[int, int]:
    dt = pd.to_datetime(timestamp, unit="s", utc=True)
    return int(dt.hour) + 1, int(dt.dayofweek) + 1


def _process_tt_bucket(delta_tt: int, max_bucket: int) -> int:
    if delta_tt < 0:
        raise ValueError(f"Invalid delta_tt: {delta_tt}")
    return min(int(np.log2(delta_tt + 1)), max_bucket)


def _left_pad_truncate(arr: np.ndarray, max_seq_len: int, dtype) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    arr = arr.astype(dtype, copy=False)
    if arr.shape[0] >= max_seq_len:
        return arr[-max_seq_len:]
    out = np.zeros(max_seq_len, dtype=dtype)
    out[-arr.shape[0] :] = arr
    return out


def _build_seq_time_diffs(seq_tt: np.ndarray) -> np.ndarray:
    if seq_tt.ndim != 1:
        raise ValueError(f"Expected 1D timestamp sequence, got shape {seq_tt.shape}")
    diffs = np.zeros_like(seq_tt)
    if seq_tt.shape[0] <= 1:
        return diffs
    prev_tt = seq_tt[:-1]
    curr_tt = seq_tt[1:]
    valid = (curr_tt != 0) & (prev_tt != 0)
    raw_delta = curr_tt - prev_tt
    valid_delta = raw_delta[valid]
    if valid_delta.size > 0 and (valid_delta < 0).sum() > (valid_delta > 0).sum():
        delta = prev_tt - curr_tt
    else:
        delta = raw_delta
    delta = np.clip(delta, a_min=0, a_max=None)
    diffs[1:] = np.where(valid, delta, 0)
    return diffs


def _parse_user_id(user_id: str) -> int:
    parts = user_id.split("_")
    if len(parts) < 2:
        raise ValueError(f"Invalid user_id: {user_id}")
    return int(parts[-1])


def _get_dense_bin(value: float, edges: list[float]) -> int:
    raw_bin = np.searchsorted(np.asarray(edges, dtype=np.float64), value, side="left")
    return int(raw_bin) + 1


def _entry_get(entry, key: str):
    if not isinstance(entry, dict):
        raise ValueError(f"Expected dict entry, got {type(entry)}")
    return entry[key]


def _seq_get(seq_feature, seq_name: str):
    if not isinstance(seq_feature, dict):
        raise ValueError(f"Expected dict seq_feature, got {type(seq_feature)}")
    return seq_feature[seq_name]


def build_item_stats(df: pd.DataFrame) -> pd.DataFrame:
    def count_action(label_arr) -> tuple[int, int]:
        if not label_arr:
            raise ValueError("Empty label array")
        impressions = 0
        clicks = 0
        for e in label_arr:
            action_type = int(e["action_type"])
            impressions += 1 if action_type == 1 else 0
            clicks += 1 if action_type == 2 else 0
        return impressions, clicks

    action_counts = df["label"].apply(count_action)
    cnt_df = pd.DataFrame(action_counts.tolist(), columns=["impression_count", "click_count"])
    tmp = pd.concat([df[["item_id", "timestamp"]].reset_index(drop=True), cnt_df], axis=1)
    item_stats = (
        tmp.groupby("item_id", as_index=False)
        .agg(
            impression_count=("impression_count", "sum"),
            click_count=("click_count", "sum"),
            last_tt=("timestamp", "max"),
        )
        .sort_values(["impression_count", "click_count", "last_tt"], ascending=[False, False, False])
        .reset_index(drop=True)
    )
    denom = item_stats["impression_count"] + item_stats["click_count"]
    item_stats["ctr"] = np.where(denom != 0, item_stats["click_count"] / denom, 0.0)
    return item_stats[["item_id", "impression_count", "click_count", "ctr"]]


def build_feature_meta(df: pd.DataFrame) -> dict:
    def process_dense_bins(dense_values: dict[int, list[float]], num_bins: int = 32) -> dict[int, list[float]]:
        ret = {}
        for fid, values in dense_values.items():
            if len(values) == 0:
                raise ValueError(f"No values found for dense feature_id {fid}")
            arr = np.asarray(values, dtype=np.float64)
            centered = arr - arr.mean()
            denom = np.power(arr.std() + 1e-12, 3)
            skewness = float((centered**3).mean() / denom)
            use_log1p = bool(np.all(arr >= 0.0) and skewness > 1.0)
            transformed = np.log1p(arr) if use_log1p else arr
            edges = np.quantile(transformed, np.linspace(0.0, 1.0, num_bins + 1))
            edges = np.unique(edges)
            ret[fid] = [float(x) for x in edges.tolist()]
        return ret

    item_sparse_max: dict[int | str, int] = defaultdict(int)
    item_multihot_max: dict[int, int] = defaultdict(int)
    user_sparse_max: dict[int, int] = defaultdict(int)
    user_multihot_max: dict[int, int] = defaultdict(int)
    user_weighted_multihot_max: dict[int, int] = defaultdict(int)
    user_embedding_dim: dict[int, int] = defaultdict(int)
    action_seq_max: dict[int, int] = defaultdict(int)
    content_seq_max: dict[int, int] = defaultdict(int)
    item_seq_max: dict[int, int] = defaultdict(int)
    item_dense_values: dict[int, list[float]] = defaultdict(list)

    for row in df.itertuples(index=False):
        item_sparse_max["item_id"] = max(item_sparse_max["item_id"], int(row.item_id))
        item_features = row.item_feature
        user_features = row.user_feature
        action_seq = _seq_get(row.seq_feature, "action_seq")
        content_seq = _seq_get(row.seq_feature, "content_seq")
        item_seq = _seq_get(row.seq_feature, "item_seq")

        for it in item_features:
            fid = int(_entry_get(it, "feature_id"))
            float_value = _entry_get(it, "float_value")
            int_array = _entry_get(it, "int_array")
            int_value = _entry_get(it, "int_value")
            if float_value is None and int_array is None and int_value is None:
                raise ValueError(f"Empty feature array for item feature_id {fid}")
            if int_value is not None:
                item_sparse_max[fid] = max(item_sparse_max[fid], int(int_value))
            elif int_array is not None and len(int_array) > 0:
                item_multihot_max[fid] = max(item_multihot_max[fid], max(int_array))
            elif float_value is not None:
                item_dense_values[fid].append(float(float_value))

        for it in user_features:
            fid = int(_entry_get(it, "feature_id"))
            float_array = _entry_get(it, "float_array")
            int_array = _entry_get(it, "int_array")
            int_value = _entry_get(it, "int_value")
            if float_array is None and int_array is None and int_value is None:
                raise ValueError(f"Empty feature array for user feature_id {fid}")
            if int_value is not None:
                user_sparse_max[fid] = max(user_sparse_max[fid], int(int_value))
            elif int_array is not None and float_array is None and len(int_array) > 0:
                user_multihot_max[fid] = max(user_multihot_max[fid], max(int_array))
            elif float_array is not None and int_array is None:
                if fid not in user_embedding_dim:
                    user_embedding_dim[fid] = len(float_array)
                elif user_embedding_dim[fid] != len(float_array):
                    raise ValueError(
                        f"Embedding dimension mismatch for user feature_id {fid}, get {len(float_array)} but expected {user_embedding_dim[fid]}"
                    )
            elif float_array is not None and int_array is not None and len(int_array) > 0:
                if len(int_array) != len(float_array):
                    raise ValueError(f"Weighted multihot length mismatch for user feature_id {fid}")
                user_weighted_multihot_max[fid] = max(user_weighted_multihot_max[fid], max(int_array))

        for it in action_seq:
            fid = int(_entry_get(it, "feature_id"))
            int_array = _entry_get(it, "int_array")
            if int_array is not None and len(int_array) > 0:
                action_seq_max[fid] = max(action_seq_max[fid], max(int_array))

        for it in content_seq:
            fid = int(_entry_get(it, "feature_id"))
            int_array = _entry_get(it, "int_array")
            if int_array is not None and len(int_array) > 0:
                content_seq_max[fid] = max(content_seq_max[fid], max(int_array))

        for it in item_seq:
            fid = int(_entry_get(it, "feature_id"))
            int_array = _entry_get(it, "int_array")
            if int_array is not None and len(int_array) > 0:
                item_seq_max[fid] = max(item_seq_max[fid], max(int_array))

    return {
        "item_feature": {
            "sparse": {str(fid): int(vocab_size) for fid, vocab_size in item_sparse_max.items()},
            "multihot": {int(fid): int(vocab_size) for fid, vocab_size in item_multihot_max.items()},
            "dense": {int(fid): bins for fid, bins in process_dense_bins(item_dense_values).items()},
        },
        "user_feature": {
            "sparse": {int(fid): int(vocab_size) for fid, vocab_size in user_sparse_max.items()},
            "multihot": {int(fid): int(vocab_size) for fid, vocab_size in user_multihot_max.items()},
            "weighted_multihot": {
                int(fid): int(vocab_size) for fid, vocab_size in user_weighted_multihot_max.items()
            },
            "embedding": {int(fid): int(dim) for fid, dim in user_embedding_dim.items()},
        },
        "seq_feature": {
            "action_seq": {int(fid): int(vocab_size) for fid, vocab_size in action_seq_max.items()},
            "content_seq": {int(fid): int(vocab_size) for fid, vocab_size in content_seq_max.items()},
            "item_seq": {int(fid): int(vocab_size) for fid, vocab_size in item_seq_max.items()},
        },
    }


def build_summary(df: pd.DataFrame, item_stats: pd.DataFrame) -> dict:
    global_impression_count = item_stats["impression_count"].sum()
    global_click_count = item_stats["click_count"].sum()
    denom = global_impression_count + global_click_count
    global_ctr = global_click_count / denom if denom != 0 else 0.0
    return {
        "num_users": int(df["user_id"].nunique()),
        "num_items": int(df["item_id"].nunique()),
        "global_impression_count": int(global_impression_count),
        "global_click_count": int(global_click_count),
        "global_ctr": float(global_ctr),
    }


def build_prepared_data(df: pd.DataFrame, feature_meta: dict, max_seq_len: int, max_delta_tt_bucket: int) -> pd.DataFrame:
    rows: list[dict] = []

    item_sparse_fids = tuple(sorted(feature_meta["item_feature"]["sparse"]))
    item_dense_fids = tuple(sorted(feature_meta["item_feature"]["dense"]))
    item_multihot_fids = tuple(sorted(feature_meta["item_feature"]["multihot"]))
    user_sparse_fids = tuple(sorted(feature_meta["user_feature"]["sparse"]))
    user_embedding_fids = tuple(sorted(feature_meta["user_feature"]["embedding"]))
    user_multihot_fids = tuple(sorted(feature_meta["user_feature"]["multihot"]))
    user_weighted_multihot_fids = tuple(sorted(feature_meta["user_feature"]["weighted_multihot"]))
    action_seq_fids = tuple(sorted(feature_meta["seq_feature"]["action_seq"]))
    content_seq_fids = tuple(sorted(feature_meta["seq_feature"]["content_seq"]))
    item_seq_fids = tuple(sorted(feature_meta["seq_feature"]["item_seq"]))

    user_embedding_dims = feature_meta["user_feature"]["embedding"]

    for row in df.itertuples(index=False):
        row_dict: dict = {}
        timestamp = int(row.timestamp)
        label = row.label
        if not isinstance(label, np.ndarray):
            raise ValueError(f"Expected label array, got {type(label)}")
        if len(label) == 0:
            raise ValueError("Empty label array")
        first_label = label[0]
        action_type = int(_entry_get(first_label, "action_type"))
        action_time = int(_entry_get(first_label, "action_time"))
        if action_time < timestamp:
            raise ValueError(f"Invalid action_time={action_time} timestamp={timestamp}")

        row_dict["item_id"] = int(row.item_id)
        row_dict["user_id"] = _parse_user_id(str(row.user_id))
        row_dict["timestamp"] = timestamp
        row_dict["label"] = int(action_type == 2)
        row_dict["action_time"] = action_time
        action_tt_hour, action_tt_dow = _extract_tt(action_time)
        row_dict["action_tt_hour"] = action_tt_hour
        row_dict["action_tt_dow"] = action_tt_dow
        row_dict["delta_tt_bucket"] = _process_tt_bucket(action_time - timestamp, max_delta_tt_bucket)

        for fid in item_sparse_fids:
            row_dict[f"item_sparse_{fid}"] = 0
        for fid in item_dense_fids:
            row_dict[f"item_dense_bin_{fid}"] = 0
        for fid in item_multihot_fids:
            row_dict[f"item_multihot_{fid}"] = np.zeros(max_seq_len, dtype=np.int64)
        for fid in user_sparse_fids:
            row_dict[f"user_sparse_{fid}"] = 0
        for fid in user_embedding_fids:
            row_dict[f"user_embedding_{fid}"] = np.zeros(user_embedding_dims[fid], dtype=np.float32)
        for fid in user_multihot_fids:
            row_dict[f"user_multihot_{fid}"] = np.zeros(max_seq_len, dtype=np.int64)
        for fid in user_weighted_multihot_fids:
            row_dict[f"user_weighted_multihot_{fid}_ids"] = np.zeros(max_seq_len, dtype=np.int64)
            row_dict[f"user_weighted_multihot_{fid}_weights"] = np.zeros(max_seq_len, dtype=np.float32)

        for fid in action_seq_fids:
            if fid == SEQ_TT_ID["action_seq"]:
                continue
            row_dict[f"action_seq_{fid}"] = np.zeros(max_seq_len, dtype=np.int64)
        for fid in content_seq_fids:
            if fid == SEQ_TT_ID["content_seq"]:
                continue
            row_dict[f"content_seq_{fid}"] = np.zeros(max_seq_len, dtype=np.int64)
        for fid in item_seq_fids:
            if fid == SEQ_TT_ID["item_seq"]:
                continue
            row_dict[f"item_seq_{fid}"] = np.zeros(max_seq_len, dtype=np.int64)

        item_features = row.item_feature
        user_features = row.user_feature
        seq_feature = row.seq_feature
        if not isinstance(item_features, np.ndarray):
            raise ValueError(f"Expected item_feature array, got {type(item_features)}")
        if not isinstance(user_features, np.ndarray):
            raise ValueError(f"Expected user_feature array, got {type(user_features)}")
        if not isinstance(seq_feature, dict):
            raise ValueError(f"Expected seq_feature dict, got {type(seq_feature)}")

        for it in item_features:
            fid = int(_entry_get(it, "feature_id"))
            float_value = _entry_get(it, "float_value")
            int_array = _entry_get(it, "int_array")
            int_value = _entry_get(it, "int_value")
            if int_value is not None:
                row_dict[f"item_sparse_{fid}"] = int(int_value)
            elif int_array is not None and len(int_array) > 0:
                row_dict[f"item_multihot_{fid}"] = _left_pad_truncate(np.asarray(int_array, dtype=np.int64), max_seq_len, np.int64)
            elif float_value is not None:
                row_dict[f"item_dense_bin_{fid}"] = _get_dense_bin(float(float_value), feature_meta["item_feature"]["dense"][fid])
            else:
                raise ValueError(f"Empty feature array for item feature_id {fid}")

        for it in user_features:
            fid = int(_entry_get(it, "feature_id"))
            float_array = _entry_get(it, "float_array")
            int_array = _entry_get(it, "int_array")
            int_value = _entry_get(it, "int_value")
            if int_value is not None:
                row_dict[f"user_sparse_{fid}"] = int(int_value)
            elif int_array is not None and float_array is None:
                row_dict[f"user_multihot_{fid}"] = _left_pad_truncate(np.asarray(int_array, dtype=np.int64), max_seq_len, np.int64)
            elif float_array is not None and int_array is None:
                arr = np.asarray(float_array, dtype=np.float32)
                if arr.shape[0] != user_embedding_dims[fid]:
                    raise ValueError(
                        f"Embedding dimension mismatch for user feature_id {fid}, get {arr.shape[0]} but expected {user_embedding_dims[fid]}"
                    )
                row_dict[f"user_embedding_{fid}"] = arr
            elif float_array is not None and int_array is not None:
                ids = np.asarray(int_array, dtype=np.int64)
                weights = np.asarray(float_array, dtype=np.float32)
                if ids.shape[0] != weights.shape[0]:
                    raise ValueError(f"Weighted multihot length mismatch for user feature_id {fid}")
                row_dict[f"user_weighted_multihot_{fid}_ids"] = _left_pad_truncate(ids, max_seq_len, np.int64)
                row_dict[f"user_weighted_multihot_{fid}_weights"] = _left_pad_truncate(weights, max_seq_len, np.float32)
            else:
                raise ValueError(f"Empty feature array for user feature_id {fid}")

        for seq_name, seq_fids, is_reversed in (
            ("action_seq", action_seq_fids, ACTION_REV),
            ("content_seq", content_seq_fids, CONTENT_REV),
            ("item_seq", item_seq_fids, ITEM_REV),
        ):
            tt_fid = SEQ_TT_ID[seq_name]
            seq_entries = {
                int(_entry_get(it, "feature_id")): _entry_get(it, "int_array")
                for it in _seq_get(seq_feature, seq_name)
            }

            tt_array = seq_entries.get(tt_fid)
            if tt_array is None or len(tt_array) == 0:
                row_dict[f"{seq_name}_timestamp"] = np.zeros(max_seq_len, dtype=np.int64)
                row_dict[f"{seq_name}_hour"] = np.zeros(max_seq_len, dtype=np.int64)
                row_dict[f"{seq_name}_dow"] = np.zeros(max_seq_len, dtype=np.int64)
                row_dict[f"{seq_name}_time_diff"] = np.zeros(max_seq_len, dtype=np.int64)
                continue

            seq_tt = np.asarray(tt_array, dtype=np.int64)
            if is_reversed:
                seq_tt = seq_tt[::-1].copy()
            seq_tt = _left_pad_truncate(seq_tt, max_seq_len, np.int64)
            row_dict[f"{seq_name}_timestamp"] = seq_tt

            valid_tt = seq_tt[seq_tt > 0]
            hours = np.zeros(max_seq_len, dtype=np.int64)
            dows = np.zeros(max_seq_len, dtype=np.int64)
            if valid_tt.shape[0] > 0:
                tt_pairs = np.asarray([_extract_tt(int(ts)) for ts in valid_tt], dtype=np.int64)
                hours[-valid_tt.shape[0] :] = tt_pairs[:, 0]
                dows[-valid_tt.shape[0] :] = tt_pairs[:, 1]
            row_dict[f"{seq_name}_hour"] = hours
            row_dict[f"{seq_name}_dow"] = dows
            row_dict[f"{seq_name}_time_diff"] = _build_seq_time_diffs(seq_tt)

            for fid in seq_fids:
                if fid == tt_fid:
                    continue
                int_array = seq_entries.get(fid)
                if int_array is None or len(int_array) == 0:
                    continue
                arr = np.asarray(int_array, dtype=np.int64)
                if is_reversed:
                    arr = arr[::-1].copy()
                row_dict[f"{seq_name}_{fid}"] = _left_pad_truncate(arr, max_seq_len, np.int64)

        rows.append(row_dict)

    return pd.DataFrame(rows)


def main():
    max_seq_len = 10
    max_delta_tt_bucket = 15

    print("Loading data...")
    df = pd.read_parquet(RAW_PATH)
    print(f"Loaded {len(df)} rows")

    item_stats = build_item_stats(df)
    print(f"Built item stats for {len(item_stats)} items")

    feature_meta = build_feature_meta(df)
    print("Built feature meta")

    prepared = build_prepared_data(df, feature_meta=feature_meta, max_seq_len=max_seq_len, max_delta_tt_bucket=max_delta_tt_bucket)
    print(f"Built prepared data with {len(prepared)} rows")

    summary = build_summary(prepared, item_stats)
    print(f"Built summary: {summary}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prepared.to_parquet(PREPARED_PATH, index=False)
    item_stats.to_parquet(ITEM_STATS_PATH, index=False)

    with FEATURE_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=4, ensure_ascii=False)

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(f"Saved prepared data to {PREPARED_PATH}")
    print(f"Saved item stats to {ITEM_STATS_PATH}")
    print(f"Saved feature meta to {FEATURE_META_PATH}")
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
