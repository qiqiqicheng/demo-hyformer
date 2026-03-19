from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from demo_hyformer.basic import BaseFeature, DenseFeature, SequenceFeature, SparseFeature, WeightedMultiHotFeature
from demo_hyformer.utils.pylogger import RankedLogger

log = RankedLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"

ACTION_REV = True
CONTENT_REV = True
ITEM_REV = True
SEQ_TT_ID = {
    "action_seq": 28,
    "content_seq": 41,
    "item_seq": 29,
}


def _load_feature_meta(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    item_sparse = meta["item_feature"]["sparse"]
    meta["item_feature"]["sparse"] = {(int(k) if k != "item_id" else k): v for k, v in item_sparse.items()}

    for group in ("multihot", "dense"):
        meta["item_feature"][group] = {int(k): v for k, v in meta["item_feature"][group].items()}

    for group in ("sparse", "multihot", "weighted_multihot", "embedding"):
        meta["user_feature"][group] = {int(k): v for k, v in meta["user_feature"][group].items()}

    for seq_name in ("action_seq", "content_seq", "item_seq"):
        meta["seq_feature"][seq_name] = {int(k): int(v) for k, v in meta["seq_feature"][seq_name].items()}

    return meta


FEATURE_META = _load_feature_meta(DATA_DIR / "processed" / "feature_meta.json")
SUMMARY = json.load(open(DATA_DIR / "processed" / "summary.json"))


class DemoDataset(Dataset):
    def __init__(self, columns: dict[str, np.ndarray]):
        super().__init__()
        self.columns = columns
        self.length = len(columns["timestamp"])

        self.item_sparse_fids = tuple(
            fid
            for fid in sorted(FEATURE_META["item_feature"]["sparse"], key=lambda x: (isinstance(x, str), x))
            if fid != "item_id"
        )
        self.item_dense_fids = tuple(sorted(FEATURE_META["item_feature"]["dense"]))
        self.item_multihot_fids = tuple(sorted(FEATURE_META["item_feature"]["multihot"]))

        self.user_sparse_fids = tuple(sorted(FEATURE_META["user_feature"]["sparse"]))
        self.user_embedding_fids = tuple(sorted(FEATURE_META["user_feature"]["embedding"]))
        self.user_multihot_fids = tuple(sorted(FEATURE_META["user_feature"]["multihot"]))
        self.user_weighted_multihot_fids = tuple(sorted(FEATURE_META["user_feature"]["weighted_multihot"]))

        self.seq_fids = {
            seq_name: tuple(sorted(FEATURE_META["seq_feature"][seq_name]))
            for seq_name in ("action_seq", "content_seq", "item_seq")
        }

    def __len__(self):
        return self.length

    def _scalar(self, name: str, idx: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
        return torch.tensor(self.columns[name][idx], dtype=dtype)

    def _array(self, name: str, idx: int, dtype: torch.dtype) -> torch.Tensor:
        value = self.columns[name][idx]
        arr = np.asarray(value)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array for {name}, got shape {arr.shape}")
        return torch.as_tensor(arr, dtype=dtype)

    def __getitem__(self, idx) -> dict:
        non_seq: dict[str, object] = {
            "item_id": self._scalar("item_id", idx),
            "user_id": self._scalar("user_id", idx),
            "action_tt_hour": self._scalar("action_tt_hour", idx),
            "action_tt_dow": self._scalar("action_tt_dow", idx),
            "delta_tt_bucket": self._scalar("delta_tt_bucket", idx),
        }

        for fid in self.item_sparse_fids:
            non_seq[f"item_sparse_{fid}"] = self._scalar(f"item_sparse_{fid}", idx)
        for fid in self.item_dense_fids:
            non_seq[f"item_dense_bin_{fid}"] = self._scalar(f"item_dense_bin_{fid}", idx)
        for fid in self.item_multihot_fids:
            non_seq[f"item_multihot_{fid}"] = self._array(f"item_multihot_{fid}", idx, torch.long)

        for fid in self.user_sparse_fids:
            non_seq[f"user_sparse_{fid}"] = self._scalar(f"user_sparse_{fid}", idx)
        for fid in self.user_embedding_fids:
            non_seq[f"user_embedding_{fid}"] = self._array(f"user_embedding_{fid}", idx, torch.float32)
        for fid in self.user_multihot_fids:
            non_seq[f"user_multihot_{fid}"] = self._array(f"user_multihot_{fid}", idx, torch.long)
        for fid in self.user_weighted_multihot_fids:
            non_seq[f"user_weighted_multihot_{fid}"] = {
                "ids": self._array(f"user_weighted_multihot_{fid}_ids", idx, torch.long),
                "weights": self._array(f"user_weighted_multihot_{fid}_weights", idx, torch.float32),
            }

        action_seq = {
            "action_seq_timestamp": self._array("action_seq_timestamp", idx, torch.long),
            "action_seq_hour": self._array("action_seq_hour", idx, torch.long),
            "action_seq_dow": self._array("action_seq_dow", idx, torch.long),
        }
        for fid in self.seq_fids["action_seq"]:
            if fid == SEQ_TT_ID["action_seq"]:
                continue
            action_seq[f"action_seq_{fid}"] = self._array(f"action_seq_{fid}", idx, torch.long)

        content_seq = {
            "content_seq_timestamp": self._array("content_seq_timestamp", idx, torch.long),
            "content_seq_hour": self._array("content_seq_hour", idx, torch.long),
            "content_seq_dow": self._array("content_seq_dow", idx, torch.long),
        }
        for fid in self.seq_fids["content_seq"]:
            if fid == SEQ_TT_ID["content_seq"]:
                continue
            content_seq[f"content_seq_{fid}"] = self._array(f"content_seq_{fid}", idx, torch.long)

        item_seq = {
            "item_seq_timestamp": self._array("item_seq_timestamp", idx, torch.long),
            "item_seq_hour": self._array("item_seq_hour", idx, torch.long),
            "item_seq_dow": self._array("item_seq_dow", idx, torch.long),
        }
        for fid in self.seq_fids["item_seq"]:
            if fid == SEQ_TT_ID["item_seq"]:
                continue
            item_seq[f"item_seq_{fid}"] = self._array(f"item_seq_{fid}", idx, torch.long)

        seq_time_diffs = {
            "action_seq_time_diff": self._array("action_seq_time_diff", idx, torch.long),
            "content_seq_time_diff": self._array("content_seq_time_diff", idx, torch.long),
            "item_seq_time_diff": self._array("item_seq_time_diff", idx, torch.long),
        }

        return {
            "non_seq": non_seq,
            "action_seq": action_seq,
            "content_seq": content_seq,
            "item_seq": item_seq,
            "seq_time_diffs": seq_time_diffs,
            "timestamp": self._scalar("timestamp", idx, torch.int32),
            "label": self._scalar("label", idx, torch.int32),
        }


class DemoDataModule(L.LightningDataModule):
    def __init__(
        self,
        batch_size,
        max_seq_len,
        max_delta_tt_bucket,
        val_ratio: float = 0.2,
        num_workers=os.cpu_count() // 4,  # type: ignore
        data_path: Path = DATA_DIR / "processed" / "prepared_data.parquet",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.max_delta_tt_bucket = max_delta_tt_bucket
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.data_path = data_path

    def _train_val_split(self, df: pd.DataFrame, val_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        unique_users = df["user_id"].unique()
        shuffled = np.random.permutation(unique_users)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val_users = set(shuffled[:n_val])
        is_val = df["user_id"].isin(val_users)
        return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)

    def _build_columns(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        return {column: df[column].to_numpy(copy=True) for column in df.columns}

    def setup(self, stage: str | None = None):
        if stage not in (None, "fit", "test"):
            return

        log.info(f"Loading dataset from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        log.info(f"Dataset loaded with {len(df)} samples")

        train_df, val_df = self._train_val_split(df, val_ratio=self.val_ratio)

        if stage in (None, "fit"):
            self.train_dataset = DemoDataset(self._build_columns(train_df))
            self.val_dataset = DemoDataset(self._build_columns(val_df))

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
        # TODO: finish test_dataloader on test dataset
        raise NotImplementedError


def get_features(
    emb_dim: int, max_delta_tt_bucket: int
) -> Tuple[
    list[BaseFeature], list[BaseFeature], list[BaseFeature], list[BaseFeature], list[BaseFeature], list[BaseFeature]
]:
    non_seq_sparse_features = []
    non_seq_multi_features = []
    non_seq_embedding_features = []
    action_seq_features = []
    content_seq_features = []
    item_seq_features = []

    non_seq_sparse_features.extend([
        SparseFeature(
            name="item_id", vocab_size=FEATURE_META["item_feature"]["sparse"]["item_id"] + 1, embed_dim=emb_dim
        ),
        SparseFeature(name="action_tt_hour", vocab_size=25, embed_dim=emb_dim),
        SparseFeature(name="action_tt_dow", vocab_size=8, embed_dim=emb_dim),
        SparseFeature(name="delta_tt_bucket", vocab_size=max_delta_tt_bucket + 1, embed_dim=emb_dim),
    ])

    for fid, vocab_size in FEATURE_META["item_feature"]["sparse"].items():
        if fid == "item_id":
            continue
        non_seq_sparse_features.append(
            SparseFeature(name=f"item_sparse_{fid}", vocab_size=vocab_size + 1, embed_dim=emb_dim)
        )

    for fid, vocab_size in FEATURE_META["item_feature"]["multihot"].items():
        non_seq_multi_features.append(
            SequenceFeature(
                name=f"item_multihot_{fid}",
                vocab_size=vocab_size + 1,
                embed_dim=emb_dim,
                pooling="concat",
            )
        )

    for fid, bin_edges in FEATURE_META["item_feature"]["dense"].items():
        non_seq_sparse_features.append(
            SparseFeature(
                name=f"item_dense_bin_{fid}",
                vocab_size=len(bin_edges) + 2,
                embed_dim=emb_dim,
            )
        )

    for fid, vocab_size in FEATURE_META["user_feature"]["sparse"].items():
        non_seq_sparse_features.append(
            SparseFeature(name=f"user_sparse_{fid}", vocab_size=vocab_size + 1, embed_dim=emb_dim)
        )

    for fid, vocab_size in FEATURE_META["user_feature"]["multihot"].items():
        non_seq_multi_features.append(
            SequenceFeature(
                name=f"user_multihot_{fid}",
                vocab_size=vocab_size + 1,
                embed_dim=emb_dim,
                pooling="concat",
            )
        )

    for fid, vocab_size in FEATURE_META["user_feature"]["weighted_multihot"].items():
        non_seq_multi_features.append(
            WeightedMultiHotFeature(
                name=f"user_weighted_multihot_{fid}",
                sparse_vocab_size=vocab_size + 1,
                embed_dim=emb_dim,
                pooling="weighted_concat",
            )
        )

    for fid, dim in FEATURE_META["user_feature"]["embedding"].items():
        non_seq_embedding_features.append(DenseFeature(name=f"user_embedding_{fid}", input_dim=dim, embed_dim=emb_dim))

    for fid, vocab_size in FEATURE_META["seq_feature"]["action_seq"].items():
        if fid == SEQ_TT_ID["action_seq"]:
            continue
        action_seq_features.append(
            SequenceFeature(name=f"action_seq_{fid}", vocab_size=vocab_size + 1, embed_dim=emb_dim, pooling="concat")
        )
    action_seq_features.extend([
        SequenceFeature(
            name="action_seq_hour", vocab_size=25, embed_dim=emb_dim, pooling="concat", shared_with="action_tt_hour"
        ),
        SequenceFeature(
            name="action_seq_dow", vocab_size=8, embed_dim=emb_dim, pooling="concat", shared_with="action_tt_dow"
        ),
    ])

    for fid, vocab_size in FEATURE_META["seq_feature"]["content_seq"].items():
        if fid == SEQ_TT_ID["content_seq"]:
            continue
        content_seq_features.append(
            SequenceFeature(name=f"content_seq_{fid}", vocab_size=vocab_size + 1, embed_dim=emb_dim, pooling="concat")
        )
    content_seq_features.extend([
        SequenceFeature(
            name="content_seq_hour", vocab_size=25, embed_dim=emb_dim, pooling="concat", shared_with="action_tt_hour"
        ),
        SequenceFeature(
            name="content_seq_dow", vocab_size=8, embed_dim=emb_dim, pooling="concat", shared_with="action_tt_dow"
        ),
    ])

    for fid, vocab_size in FEATURE_META["seq_feature"]["item_seq"].items():
        if fid == SEQ_TT_ID["item_seq"]:
            continue
        item_seq_features.append(
            SequenceFeature(name=f"item_seq_{fid}", vocab_size=vocab_size + 1, embed_dim=emb_dim, pooling="concat")
        )
    item_seq_features.extend([
        SequenceFeature(
            name="item_seq_hour", vocab_size=25, embed_dim=emb_dim, pooling="concat", shared_with="action_tt_hour"
        ),
        SequenceFeature(
            name="item_seq_dow", vocab_size=8, embed_dim=emb_dim, pooling="concat", shared_with="action_tt_dow"
        ),
    ])

    return (
        non_seq_sparse_features,
        non_seq_multi_features,
        non_seq_embedding_features,
        action_seq_features,
        content_seq_features,
        item_seq_features,
    )


def get_semantic_groups(mode: int | None = None):
    if mode is None:
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
    elif mode == 1:
        pass
    else:
        raise ValueError(f"Unsupported semantic group mode: {mode} with type: {type(mode)}")
    return semantic_groups


if __name__ == "__main__":

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

    data_module = DemoDataModule(batch_size=4, max_seq_len=10, max_delta_tt_bucket=100)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))

    with (DATA_DIR / "processed" / "sample_batch.json").open("w", encoding="utf-8") as f:
        json.dump(_to_serializable(batch), f, indent=2)

    (
        non_seq_sparse_features,
        non_seq_multi_features,
        non_seq_embedding_features,
        action_seq_features,
        item_seq_features,
        content_seq_features,
    ) = get_features(emb_dim=16, max_delta_tt_bucket=100)

    get_features_dict = {
        "non_seq_sparse_features": [f.__repr__() for f in non_seq_sparse_features],
        "non_seq_multi_features": [f.__repr__() for f in non_seq_multi_features],
        "non_seq_embedding_features": [f.__repr__() for f in non_seq_embedding_features],
        "action_seq": [f.__repr__() for f in action_seq_features],
        "content_seq": [f.__repr__() for f in content_seq_features],
        "item_seq": [f.__repr__() for f in item_seq_features],
    }
    with (DATA_DIR / "processed" / "get_features.json").open("w", encoding="utf-8") as f:
        json.dump(get_features_dict, f, ensure_ascii=False, indent=2)
