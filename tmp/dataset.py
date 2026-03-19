from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from demo_hyformer.basic import BaseFeature, DenseFeature, SequenceFeature, SparseFeature, WeightedMultiHotFeature
from demo_hyformer.utils.pylogger import RankedLogger

log = RankedLogger(__name__)

root_dir = Path(__file__).parent.parent.parent
data_path = root_dir / "data" / "raw" / "sample_data.parquet"


def _load_feature_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # JSON object keys are strings; convert known feature-id keyed maps back to int keys.
    for group in ("sparse", "dense", "multihot"):
        d = meta.get("item_feature", {}).get(group, {})
        meta["item_feature"][group] = {int(k): v for k, v in d.items()}

    for group in ("sparse", "multihot", "embedding", "weighted_multihot"):
        d = meta.get("user_feature", {}).get(group, {})
        meta["user_feature"][group] = {int(k): v for k, v in d.items()}

    for seq_name in ("action_seq", "content_seq", "item_seq"):
        seq = meta.get("seq_feature", {}).get(seq_name, {})
        max_vals = seq.get("max_vals", {})
        seq["max_vals"] = {int(k): int(v) for k, v in max_vals.items()}
        seq["feature_ids"] = [int(x) for x in seq.get("feature_ids", [])]

    return meta


FEATURE_META = _load_feature_meta(root_dir / "data" / "processed" / "feature_meta.json")


def _entry_get(entry, key, default=None):
    if isinstance(entry, dict):
        return entry.get(key, default)
    try:
        value = entry[key]
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        return value
    except Exception:
        return default


def _to_int(value, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return int(value.item())
        if value.size == 0:
            return default
        return int(value.reshape(-1)[0])
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return float(value.item())
        if value.size == 0:
            return default
        return float(value.reshape(-1)[0])
    try:
        return float(value)
    except Exception:
        return default


def _parse_raw_id(value) -> int:
    if isinstance(value, str):
        if "_" in value:
            return _to_int(value.split("_")[-1], default=0)
        return _to_int(value, default=0)
    return _to_int(value, default=0)


def _normalize_int_array(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.int64)
    arr = np.array(value, dtype=np.int64, copy=True)
    if arr.ndim == 0:
        return np.array([int(arr)], dtype=np.int64)
    return arr


def _normalize_float_array(value) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    arr = np.array(value, dtype=np.float32, copy=True)
    if arr.ndim == 0:
        return np.array([float(arr)], dtype=np.float32)
    return arr


def _log_bucket(delta_seconds: int | float, max_bucket: int = 63) -> int:
    if delta_seconds is None:
        return 0
    v = int(delta_seconds)
    if v <= 0:
        return 0
    return min(int(np.log2(v + 1)) + 1, max_bucket)


class DemoDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        max_seq_len: int,
        dense_num_bins: int = 32,
    ):
        super().__init__()
        self.df = df
        self.dense_num_bins = dense_num_bins
        self.max_seq_len = max_seq_len

        self._item_sparse_fids = tuple(FEATURE_META["item_feature"]["sparse"].keys())
        self._item_dense_fids = tuple(FEATURE_META["item_feature"]["dense"].keys())
        self._item_multihot_fids = tuple(FEATURE_META["item_feature"]["multihot"].keys())

        self._user_sparse_fids = tuple(FEATURE_META["user_feature"]["sparse"].keys())
        self._user_embedding_fids = tuple(FEATURE_META["user_feature"]["embedding"].keys())
        self._user_multihot_fids = tuple(FEATURE_META["user_feature"]["multihot"].keys())
        self._user_weighted_fids = tuple(FEATURE_META["user_feature"]["weighted_multihot"].keys())

        self._item_sparse_set = set(self._item_sparse_fids)
        self._item_dense_set = set(self._item_dense_fids)
        self._item_multihot_set = set(self._item_multihot_fids)

        self._user_sparse_set = set(self._user_sparse_fids)
        self._user_embedding_set = set(self._user_embedding_fids)
        self._user_multihot_set = set(self._user_multihot_fids)
        self._user_weighted_set = set(self._user_weighted_fids)

        self._seq_time_fids = {"action_seq": 28, "content_seq": 41, "item_seq": 29}
        self._seq_fids = {
            "action_seq": tuple(FEATURE_META["seq_feature"]["action_seq"]["feature_ids"]),
            "content_seq": tuple(FEATURE_META["seq_feature"]["content_seq"]["feature_ids"]),
            "item_seq": tuple(FEATURE_META["seq_feature"]["item_seq"]["feature_ids"]),
        }

        self._user_embedding_dims = self._infer_user_embedding_dims()
        self._dense_binning_state = self._fit_item_dense_binning(fid=17)
        self._item_multihot_max_lens, self._user_multihot_max_lens, self._user_weighted_max_lens = (
            self._infer_nonseq_list_max_lens()
        )

    def __len__(self):
        return len(self.df)

    def _infer_user_embedding_dims(self) -> dict[int, int]:
        dims: dict[int, int] = dict.fromkeys(self._user_embedding_fids, 0)
        for user_feats in self.df["user_feature"].values:
            for e in user_feats:
                fid = _to_int(_entry_get(e, "feature_id", -1), default=-1)
                if fid not in dims:
                    continue
                arr = _normalize_float_array(_entry_get(e, "float_array"))
                if arr.size > 0:
                    dims[fid] = max(dims[fid], int(arr.shape[0]))
        # Fallback to dimension 1 to avoid empty shape tensors in edge cases.
        return {fid: (dim if dim > 0 else 1) for fid, dim in dims.items()}

    def _fit_item_dense_binning(self, fid: int) -> dict:
        dense_meta = FEATURE_META.get("item_feature", {}).get("dense", {}).get(fid, {})
        offline = dense_meta.get("binning")
        if isinstance(offline, dict) and offline.get("enabled", False):
            edges = np.asarray(offline.get("bin_edges", []), dtype=np.float64)
            return {
                "fid": fid,
                "enabled": True,
                "use_log1p": bool(offline.get("use_log1p", False)),
                "bin_edges": (edges if edges.shape[0] > 1 else None),
                "num_bins": int(offline.get("num_bins", 1)),
            }

        values: list[float] = []
        for item_feats in self.df["item_feature"].values:
            for e in item_feats:
                if _to_int(_entry_get(e, "feature_id", -1), default=-1) != fid:
                    continue
                v = _entry_get(e, "float_value")
                if v is None:
                    continue
                values.append(_to_float(v, default=0.0))

        if len(values) == 0:
            return {
                "fid": fid,
                "enabled": False,
                "use_log1p": False,
                "bin_edges": None,
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

        quantiles = np.linspace(0.0, 1.0, self.dense_num_bins + 1)
        edges = np.quantile(transformed, quantiles)
        edges = np.unique(edges)

        if edges.shape[0] <= 1:
            return {
                "fid": fid,
                "enabled": True,
                "use_log1p": use_log1p,
                "bin_edges": None,
                "num_bins": 1,
            }

        return {
            "fid": fid,
            "enabled": True,
            "use_log1p": use_log1p,
            "bin_edges": edges,
            "num_bins": int(edges.shape[0] - 1),
        }

    def _bin_dense_value(self, value: float | None) -> int:
        if value is None:
            return 0
        state = self._dense_binning_state
        if not state["enabled"]:
            return 0
        v = float(value)
        if state["use_log1p"] and v >= 0.0:
            v = float(np.log1p(v))
        edges = state["bin_edges"]
        if edges is None:
            return 1
        # 0 is reserved for padding/unknown; valid bins start from 1.
        return int(np.digitize(v, edges[1:-1], right=True) + 1)

    @staticmethod
    def _left_pad_or_truncate_1d(t: torch.Tensor, target_len: int, pad_value: int | float = 0) -> torch.Tensor:
        if t.numel() > target_len:
            return t[-target_len:]
        if t.numel() < target_len:
            pad = torch.full((target_len - t.numel(),), pad_value, dtype=t.dtype)
            return torch.cat([pad, t], dim=0)
        return t

    def _infer_nonseq_list_max_lens(self) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
        item_multihot_max_lens = dict.fromkeys(self._item_multihot_fids, 1)
        user_multihot_max_lens = dict.fromkeys(self._user_multihot_fids, 1)
        user_weighted_max_lens = dict.fromkeys(self._user_weighted_fids, 1)

        for item_features in self.df["item_feature"].values:
            for e in item_features:
                fid = _to_int(_entry_get(e, "feature_id", -1), default=-1)
                if fid not in item_multihot_max_lens:
                    continue
                ids = _normalize_int_array(_entry_get(e, "int_array"))
                item_multihot_max_lens[fid] = max(item_multihot_max_lens[fid], int(ids.shape[0]))

        for user_features in self.df["user_feature"].values:
            for e in user_features:
                fid = _to_int(_entry_get(e, "feature_id", -1), default=-1)
                if fid in user_multihot_max_lens:
                    ids = _normalize_int_array(_entry_get(e, "int_array"))
                    user_multihot_max_lens[fid] = max(user_multihot_max_lens[fid], int(ids.shape[0]))
                if fid in user_weighted_max_lens:
                    ids = _normalize_int_array(_entry_get(e, "int_array"))
                    user_weighted_max_lens[fid] = max(user_weighted_max_lens[fid], int(ids.shape[0]))

        return item_multihot_max_lens, user_multihot_max_lens, user_weighted_max_lens

    def _process_tt(self, tt):
        # Supports scalar and sequence timestamps; keep 0 as padding index.
        if isinstance(tt, (list, tuple, np.ndarray, torch.Tensor)):
            arr = np.asarray(tt, dtype=np.int64)
            if arr.ndim == 0:
                arr = arr.reshape(1)
            hour = np.zeros_like(arr, dtype=np.int64)
            dow = np.zeros_like(arr, dtype=np.int64)
            valid = arr > 0
            if np.any(valid):
                dt = pd.to_datetime(arr[valid], unit="s", utc=True)
                hour[valid] = dt.hour.to_numpy(dtype=np.int64) + 1
                dow[valid] = dt.dayofweek.to_numpy(dtype=np.int64) + 1
            return hour, dow

        ts = int(tt)
        if ts <= 0:
            return 0, 0
        dt = pd.to_datetime(ts, unit="s", utc=True)
        return int(dt.hour) + 1, int(dt.dayofweek) + 1

    def _get_item_features(self, item_features: np.ndarray) -> tuple[dict, dict, dict]:
        sparse_features: dict[str, torch.Tensor] = {
            f"item_sparse_{fid}": torch.tensor(0, dtype=torch.long) for fid in self._item_sparse_fids
        }
        dense_features: dict[str, torch.Tensor] = {
            f"item_dense_{fid}": torch.tensor(0.0, dtype=torch.float32) for fid in self._item_dense_fids
        }
        # Dense bin feature is modeled as sparse to support unified embedding style.
        dense_features["item_dense_bin_17"] = torch.tensor(0, dtype=torch.long)
        multihot_features: dict[str, torch.Tensor] = {
            f"item_multihot_{fid}": torch.zeros(self._item_multihot_max_lens[fid], dtype=torch.long)
            for fid in self._item_multihot_fids
        }

        for e in item_features:
            fid = _to_int(_entry_get(e, "feature_id", -1), default=-1)
            if fid in self._item_sparse_set:
                v = _entry_get(e, "int_value", 0)
                sparse_features[f"item_sparse_{fid}"] = torch.tensor(_to_int(v, default=0), dtype=torch.long)
            elif fid in self._item_dense_set:
                fv = _to_float(_entry_get(e, "float_value", 0.0), default=0.0)
                dense_features[f"item_dense_{fid}"] = torch.tensor(fv, dtype=torch.float32)
                dense_features[f"item_dense_bin_{fid}"] = torch.tensor(self._bin_dense_value(fv), dtype=torch.long)
            elif fid in self._item_multihot_set:
                ids = _normalize_int_array(_entry_get(e, "int_array"))
                ids_tensor = torch.from_numpy(ids).long()
                target_len = self._item_multihot_max_lens[fid]
                multihot_features[f"item_multihot_{fid}"] = self._left_pad_or_truncate_1d(ids_tensor, target_len)

        return sparse_features, dense_features, multihot_features

    def _get_user_features(self, user_features: np.ndarray) -> tuple[dict, dict, dict, dict]:
        sparse_features: dict[str, torch.Tensor] = {
            f"user_sparse_{fid}": torch.tensor(0, dtype=torch.long) for fid in self._user_sparse_fids
        }
        embedding_features: dict[str, torch.Tensor] = {
            f"user_embedding_{fid}": torch.zeros(self._user_embedding_dims[fid], dtype=torch.float32)
            for fid in self._user_embedding_fids
        }
        multihot_features: dict[str, torch.Tensor] = {
            f"user_multihot_{fid}": torch.zeros(self._user_multihot_max_lens[fid], dtype=torch.long)
            for fid in self._user_multihot_fids
        }
        weighted_multihot_features: dict[str, dict[str, torch.Tensor]] = {
            f"user_weighted_multihot_{fid}": {
                "ids": torch.zeros(self._user_weighted_max_lens[fid], dtype=torch.long),
                "weights": torch.zeros(self._user_weighted_max_lens[fid], dtype=torch.float32),
            }
            for fid in self._user_weighted_fids
        }

        for e in user_features:
            fid = _to_int(_entry_get(e, "feature_id", -1), default=-1)
            if fid in self._user_sparse_set:
                v = _entry_get(e, "int_value", 0)
                sparse_features[f"user_sparse_{fid}"] = torch.tensor(_to_int(v, default=0), dtype=torch.long)
            elif fid in self._user_embedding_set:
                arr = _normalize_float_array(_entry_get(e, "float_array"))
                embedding_features[f"user_embedding_{fid}"] = torch.from_numpy(arr).float()
            elif fid in self._user_multihot_set:
                ids = _normalize_int_array(_entry_get(e, "int_array"))
                ids_tensor = torch.from_numpy(ids).long()
                target_len = self._user_multihot_max_lens[fid]
                multihot_features[f"user_multihot_{fid}"] = self._left_pad_or_truncate_1d(ids_tensor, target_len)
            elif fid in self._user_weighted_set:
                ids = _normalize_int_array(_entry_get(e, "int_array"))
                weights = _normalize_float_array(_entry_get(e, "float_array"))
                if weights.size == 0 and ids.size > 0:
                    weights = np.ones(ids.shape[0], dtype=np.float32)
                ids_tensor = torch.from_numpy(ids).long()
                weights_tensor = torch.from_numpy(weights).float()
                target_len = self._user_weighted_max_lens[fid]
                weighted_multihot_features[f"user_weighted_multihot_{fid}"] = {
                    "ids": self._left_pad_or_truncate_1d(ids_tensor, target_len),
                    "weights": self._left_pad_or_truncate_1d(weights_tensor, target_len, pad_value=0.0),
                }

        return (
            sparse_features,
            embedding_features,
            multihot_features,
            weighted_multihot_features,
        )

    def _process_seq(self, seq_features: np.ndarray) -> tuple[dict, dict, dict]:
        seq_out: dict[str, dict[str, torch.Tensor]] = {
            "action_seq": {
                f"action_seq_{fid}": torch.empty(0, dtype=torch.long) for fid in self._seq_fids["action_seq"]
            },
            "content_seq": {
                f"content_seq_{fid}": torch.empty(0, dtype=torch.long) for fid in self._seq_fids["content_seq"]
            },
            "item_seq": {f"item_seq_{fid}": torch.empty(0, dtype=torch.long) for fid in self._seq_fids["item_seq"]},
        }

        for seq_name in ("action_seq", "content_seq", "item_seq"):
            arr = _entry_get(seq_features, seq_name, [])
            if arr is None:
                arr = []
            time_fid = self._seq_time_fids[seq_name]
            timestamp_tensor = torch.empty(0, dtype=torch.long)

            for e in arr:
                fid = _to_int(_entry_get(e, "feature_id", -1), default=-1)
                # if fid not in self._seq_fids[seq_name]:
                #     continue
                ids = _normalize_int_array(_entry_get(e, "int_array"))
                name = f"{seq_name}_{fid}"
                seq_out[seq_name][name] = torch.from_numpy(ids).long()
                if fid == time_fid:
                    timestamp_tensor = seq_out[seq_name][name]

            if timestamp_tensor.numel() > 1:
                valid_ts = timestamp_tensor[timestamp_tensor > 0]
                if valid_ts.numel() > 1 and valid_ts[0] > valid_ts[-1]:
                    for name, values in list(seq_out[seq_name].items()):
                        if values.ndim == 1 and values.numel() > 1:
                            seq_out[seq_name][name] = torch.flip(values, dims=[0])
                    timestamp_tensor = seq_out[seq_name][f"{seq_name}_{time_fid}"]

            if timestamp_tensor.numel() > 0:
                hour, dow = self._process_tt(timestamp_tensor.numpy())
                seq_out[seq_name][f"{seq_name}_hour"] = torch.from_numpy(hour).long()
                seq_out[seq_name][f"{seq_name}_dow"] = torch.from_numpy(dow).long()
                seq_out[seq_name][f"{seq_name}_time_diffs"] = self._build_seq_time_diffs(timestamp_tensor)
            else:
                seq_out[seq_name][f"{seq_name}_hour"] = torch.empty(0, dtype=torch.long)
                seq_out[seq_name][f"{seq_name}_dow"] = torch.empty(0, dtype=torch.long)
                seq_out[seq_name][f"{seq_name}_time_diffs"] = torch.empty(0, dtype=torch.long)

            # Truncate to max_seq_len (keep most recent = last elements) then left-pad with zeros.
            max_len = self.max_seq_len
            for name in list(seq_out[seq_name].keys()):
                t = seq_out[seq_name][name]
                if t.ndim != 1:
                    continue
                seq_len = t.numel()
                if seq_len > max_len:
                    seq_out[seq_name][name] = t[-max_len:]
                elif seq_len < max_len:
                    pad = torch.zeros(max_len - seq_len, dtype=t.dtype)
                    seq_out[seq_name][name] = torch.cat([pad, t])

        return seq_out["action_seq"], seq_out["content_seq"], seq_out["item_seq"]

    @staticmethod
    def _build_seq_time_diffs(timestamp_tensor: torch.Tensor) -> torch.Tensor:
        if timestamp_tensor.numel() == 0:
            return torch.empty(0, dtype=torch.long)

        diffs = torch.zeros_like(timestamp_tensor)
        if timestamp_tensor.numel() == 1:
            return diffs

        prev_ts = timestamp_tensor[:-1]
        curr_ts = timestamp_tensor[1:]
        valid = (prev_ts > 0) & (curr_ts > 0)

        raw_delta = curr_ts - prev_ts
        valid_delta = raw_delta[valid]
        if valid_delta.numel() > 0 and (valid_delta < 0).sum() > (valid_delta > 0).sum():
            delta = prev_ts - curr_ts
        else:
            delta = raw_delta
        delta = torch.clamp(delta, min=0)

        diffs[1:] = torch.where(valid, delta, torch.zeros_like(delta))
        return diffs

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Scalar context features.
        item_id: int = _parse_raw_id(row["item_id"])
        label_arr = row["label"]
        if len(label_arr) > 0:
            action_type = _to_int(_entry_get(label_arr[0], "action_type", 1), default=1)
            action_timestamp = _to_int(_entry_get(label_arr[0], "action_time", 0), default=0)
        else:
            action_type = 1
            action_timestamp = 0
        is_click: int = 1 if action_type == 2 else 0
        session_timestamp: int = int(row["timestamp"])
        delta_timestamp = action_timestamp - session_timestamp

        # Timestamp derived scalar features.
        action_hour, action_dow = self._process_tt(action_timestamp)

        # Non-sequence features.
        item_sparse_features, item_dense_features, item_multihot_features = self._get_item_features(row["item_feature"])
        (
            user_sparse_features,
            user_embedding_features,
            user_multihot_features,
            user_weighted_multihot_features,
        ) = self._get_user_features(row["user_feature"])

        # Sequence features.
        action_seq, content_seq, item_seq = self._process_seq(row["seq_feature"])

        action_seq_ts = action_seq.get(f"action_seq_{self._seq_time_fids['action_seq']}")
        if action_seq_ts is not None and action_seq_ts.numel() > 0:
            valid_ts = action_seq_ts[action_seq_ts > 0]
            if valid_ts.numel() > 0:
                last_action_seq_ts = int(valid_ts[-1].item())
            else:
                last_action_seq_ts = 0
        else:
            last_action_seq_ts = 0

        action_seq_last_delta = session_timestamp - last_action_seq_ts if last_action_seq_ts > 0 else 0

        delta_tt_bucket = _log_bucket(delta_timestamp)
        action_seq_last_delta_bucket = _log_bucket(action_seq_last_delta)

        non_seq = {
            "item_id": torch.tensor(item_id, dtype=torch.long),
            "action_tt_hour": torch.tensor(action_hour, dtype=torch.long),
            "action_tt_dow": torch.tensor(action_dow, dtype=torch.long),
            "delta_tt_bucket": torch.tensor(delta_tt_bucket, dtype=torch.long),
            "action_seq_last_delta_bucket": torch.tensor(action_seq_last_delta_bucket, dtype=torch.long),
        }
        non_seq.update(item_sparse_features)
        non_seq.update(item_dense_features)
        non_seq.update(item_multihot_features)
        non_seq.update(user_sparse_features)
        non_seq.update(user_embedding_features)
        non_seq.update(user_multihot_features)
        non_seq.update(user_weighted_multihot_features)

        sample_dict = {
            "non_seq": non_seq,
            "action_seq": action_seq,
            "content_seq": content_seq,
            "item_seq": item_seq,
            "seq_time_diffs": {
                "action_seq": action_seq.get("action_seq_time_diffs", torch.empty(0, dtype=torch.long)),
                "content_seq": content_seq.get("content_seq_time_diffs", torch.empty(0, dtype=torch.long)),
                "item_seq": item_seq.get("item_seq_time_diffs", torch.empty(0, dtype=torch.long)),
            },
            "timestamp": torch.tensor(action_timestamp, dtype=torch.long),
            "label": torch.tensor(is_click, dtype=torch.long),
        }

        return sample_dict


class DemoDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size,
        max_seq_len,
        val_ratio: float = 0.2,
        num_workers=os.cpu_count() // 4,  # type: ignore
        data_path: Path = data_path,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.data_path = data_path
        self.num_workers = num_workers
        self.val_ratio = val_ratio

    def _train_val_split(self, df: pd.DataFrame, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        unique_users = df["user_id"].unique()
        shuffled = np.random.permutation(unique_users)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_users = set(shuffled[:n_val])
        is_val = df["user_id"].isin(val_users)
        return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)

    def setup(self, stage: str | None = None):
        if stage not in (None, "fit", "test"):
            return

        log.info(f"Loading dataset from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        log.info(f"Dataset loaded with {len(df)} samples")

        train_df, val_df = self._train_val_split(df, val_ratio=self.val_ratio)

        if stage in (None, "fit"):
            self.train_dataset = DemoDataset(
                df=train_df,
                max_seq_len=self.max_seq_len,
            )
            self.val_dataset = DemoDataset(
                df=val_df,
                max_seq_len=self.max_seq_len,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        raise NotImplementedError("Test dataloader is not implemented in this demo.")


def _resolve_vocab_size(meta: dict) -> int:
    if "vocab_size" in meta:
        return int(meta["vocab_size"])
    if "max_val" in meta:
        return int(meta["max_val"]) + 1
    if "max_arr_len" in meta:
        return int(meta["max_arr_len"]) + 1
    if "fixed_arr_len" in meta:
        return int(meta["fixed_arr_len"]) + 1
    # Keep demo usable when metadata is incomplete; replace with exact vocab_size later.
    return 4096


def get_features(
    embed_dim: int = 16,
    padding_idx: int = 0,
    item_id_vocab_size: int = 5000,
    dense_bin_vocab_size: int = 33,
) -> Tuple[
    list[BaseFeature],
    list[SequenceFeature],
    list[SequenceFeature],
    list[SequenceFeature],
]:
    """Build feature definitions for unified modeling of non-seq and multi-seq features.

    Returns:
        non_seq_features, action_seq_features, content_seq_features, item_seq_features
    """
    non_seq_features: list[BaseFeature] = []
    hour_shared_name = "action_tt_hour"
    dow_shared_name = "action_tt_dow"
    delta_shared_name = "delta_tt_bucket"

    non_seq_features.append(
        SparseFeature(
            name="item_id",
            vocab_size=item_id_vocab_size,
            embed_dim=embed_dim,
            padding_idx=padding_idx,
        )
    )

    non_seq_features.append(
        SparseFeature(
            name=hour_shared_name,
            vocab_size=64,
            embed_dim=embed_dim,
            padding_idx=padding_idx,
        )
    )
    non_seq_features.append(
        SparseFeature(
            name=dow_shared_name,
            vocab_size=8,
            embed_dim=embed_dim,
            padding_idx=padding_idx,
        )
    )
    non_seq_features.append(
        SparseFeature(
            name="delta_tt_bucket",
            vocab_size=64,
            embed_dim=embed_dim,
            padding_idx=padding_idx,
        )
    )
    non_seq_features.append(
        SparseFeature(
            name="action_seq_last_delta_bucket",
            vocab_size=64,
            embed_dim=embed_dim,
            shared_with=delta_shared_name,
            padding_idx=padding_idx,
        )
    )

    # Item sparse features.
    for fid, meta in FEATURE_META["item_feature"]["sparse"].items():
        non_seq_features.append(
            SparseFeature(
                name=f"item_sparse_{fid}",
                vocab_size=int(meta["vocab_size"]),
                embed_dim=embed_dim,
                padding_idx=padding_idx,
            )
        )

    # Item dense-bin feature (fid=17) modeled as sparse feature for embedding.
    non_seq_features.append(
        SparseFeature(
            name="item_dense_bin_17",
            vocab_size=int(dense_bin_vocab_size),
            embed_dim=embed_dim,
            padding_idx=padding_idx,
        )
    )

    # Item multihot features use SequenceFeature with pooling.
    for fid, meta in FEATURE_META["item_feature"]["multihot"].items():
        non_seq_features.append(
            SequenceFeature(
                name=f"item_multihot_{fid}",
                vocab_size=_resolve_vocab_size(meta),
                embed_dim=embed_dim,
                pooling="concat",
                padding_idx=padding_idx,
            )
        )

    # User sparse features.
    for fid, meta in FEATURE_META["user_feature"]["sparse"].items():
        non_seq_features.append(
            SparseFeature(
                name=f"user_sparse_{fid}",
                vocab_size=int(meta["vocab_size"]),
                embed_dim=embed_dim,
                padding_idx=padding_idx,
            )
        )

    # User multihot features as SequenceFeature.
    for fid, meta in FEATURE_META["user_feature"]["multihot"].items():
        non_seq_features.append(
            SequenceFeature(
                name=f"user_multihot_{fid}",
                vocab_size=_resolve_vocab_size(meta),
                embed_dim=embed_dim,
                pooling="concat",
                padding_idx=padding_idx,
            )
        )

    # User weighted multi-hot features.
    for fid, meta in FEATURE_META["user_feature"]["weighted_multihot"].items():
        non_seq_features.append(
            WeightedMultiHotFeature(
                name=f"user_weighted_multihot_{fid}",
                sparse_vocab_size=_resolve_vocab_size(meta),
                embed_dim=embed_dim,
                pooling="weighted_concat",
                padding_idx=padding_idx,
            )
        )

    # User dense embedding-vector features.
    for fid, meta in FEATURE_META["user_feature"]["embedding"].items():
        non_seq_features.append(
            DenseFeature(
                name=f"user_embedding_{fid}",
                input_dim=int(meta["dim"]),
                embed_dim=embed_dim,
            )
        )

    def _build_seq_features(seq_name: str, seq_meta: dict) -> list[SequenceFeature]:
        out: list[SequenceFeature] = []
        max_vals = seq_meta["max_vals"]
        time_fids = {"action_seq": 28, "content_seq": 41, "item_seq": 29}
        for fid in seq_meta["feature_ids"]:
            if fid == time_fids[seq_name]:
                # Raw timestamps are not embedded directly; use derived hour/dow sequences instead.
                continue
            out.append(
                SequenceFeature(
                    name=f"{seq_name}_{fid}",
                    vocab_size=int(max_vals[fid]) + 1,
                    embed_dim=embed_dim,
                    pooling="concat",
                    padding_idx=padding_idx,
                )
            )

        out.append(
            SequenceFeature(
                name=f"{seq_name}_hour",
                vocab_size=64,
                embed_dim=embed_dim,
                pooling="concat",
                shared_with=hour_shared_name,
                padding_idx=padding_idx,
            )
        )
        out.append(
            SequenceFeature(
                name=f"{seq_name}_dow",
                vocab_size=8,
                embed_dim=embed_dim,
                pooling="concat",
                shared_with=dow_shared_name,
                padding_idx=padding_idx,
            )
        )
        return out

    action_seq_features = _build_seq_features("action_seq", FEATURE_META["seq_feature"]["action_seq"])
    content_seq_features = _build_seq_features("content_seq", FEATURE_META["seq_feature"]["content_seq"])
    item_seq_features = _build_seq_features("item_seq", FEATURE_META["seq_feature"]["item_seq"])

    return (
        non_seq_features,
        action_seq_features,
        content_seq_features,
        item_seq_features,
    )


def _to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def get_semantic_groups(mode: Optional[int] = None):
    if not mode:
        return None
    if mode == 0:
        semantic_groups = [
            ["item_id"] + [f"item_sparse_{i}" for i in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 75, 77, 78, 79]],
            ["action_tt_hour", "action_tt_dow", "delta_tt_bucket", "action_seq_last_delta_bucket"],
            ["item_dense_bin_17"],
            ["item_multihot_14"],
            [f"user_sparse_{i}" for i in [1, 3, 4]],
            [f"user_sparse_{i}" for i in [50, 51, 52, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]],
            [f"user_sparse_{i}" for i in [76, 80, 82]],
            [f"user_sparse_{i}" for i in range(86, 106)],
            [f"user_multihot_{i}" for i in [5, 18, 53, 54, 67, 74]],
            [f"user_weighted_multihot_{i}" for i in range(69, 74)],
            [f"user_weighted_multihot_{i}" for i in range(83, 86)],
        ]
    else:
        raise ValueError(f"Unsupported semantic group mode: {mode}")
    return semantic_groups


if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent
    data_path = Path(root_dir, "data/raw/sample_data.parquet")
    output_dir = Path(root_dir, "data/processed")

    df = pd.read_parquet(data_path)
    dataset = DemoDataset(df, max_seq_len=50)
    sample = dataset[0]

    with (output_dir / "dataset_getitem_sample.json").open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(sample), f, ensure_ascii=False, indent=2)

    non_seq_features, action_seq_features, content_seq_features, item_seq_features = get_features(
        item_id_vocab_size=int(FEATURE_META.get("api_fit", {}).get("item_id_vocab_size", 5000)),
        dense_bin_vocab_size=int(
            FEATURE_META.get("item_feature", {}).get("dense", {}).get(17, {}).get("binning", {}).get("num_bins", 32)
        )
        + 1,
    )

    print(f"getitem sample saved to: {output_dir / 'dataset_getitem_sample.json'}")

    get_features_dict = {
        "non_seq": [f.__repr__() for f in non_seq_features],
        "action_seq": [f.__repr__() for f in action_seq_features],
        "content_seq": [f.__repr__() for f in content_seq_features],
        "item_seq": [f.__repr__() for f in item_seq_features],
    }
    with (output_dir / "get_features.json").open("w", encoding="utf-8") as f:
        json.dump(get_features_dict, f, ensure_ascii=False, indent=2)
