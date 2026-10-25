from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .embedding_analysis import build_item_level_analysis_table
from .paths import ITEM_LEVEL_DIR, PROCESSED_DIGI_DIR, ensure_output_dirs, get_run_dir
from .popularity_metrics import (
    compute_cold_start_test_diagnostic,
    compute_ranking_metrics_from_predictions,
    compute_stratified_ranking_metrics,
    evaluate_top20_popularity_bias,
)
from .utils import (
    create_run_logger,
    load_experiment_config,
    load_item_mapping,
    save_json,
    save_yaml,
    set_global_seed,
)


class SessionPrefixDataset(Dataset):
    """
    Dataset tiền xử lý sẵn chuỗi prefix sang ma trận numpy cố định (N, max_seq_length)
    để tốc độ nạp batch vào GPU/CPU nhanh gấp 10 lần so với duyệt list Python ở mỗi batch.
    """

    def __init__(self, examples_df: pd.DataFrame, max_seq_length: int = 50):
        super().__init__()
        self.max_seq_length = int(max_seq_length)
        n = len(examples_df)

        self.session_ids = examples_df["session_id"].to_numpy(dtype=np.int64)
        self.targets = examples_df["target_internal_item_id"].to_numpy(dtype=np.int64)
        self.target_raw_ids = examples_df["target_item_id"].to_numpy(dtype=np.int64)
        self.target_groups = examples_df["target_popularity_group"].astype(str).to_numpy()
        self.query_ids = np.arange(n, dtype=np.int64)

        sequences = np.zeros((n, self.max_seq_length), dtype=np.int64)
        lengths = np.empty(n, dtype=np.int64)

        for idx, seq in enumerate(examples_df["prefix_items"]):
            arr = np.asarray(seq, dtype=np.int64)
            if len(arr) > self.max_seq_length:
                arr = arr[-self.max_seq_length :]
            length = max(1, len(arr))
            sequences[idx, :length] = arr[:length]
            lengths[idx] = length

        self.sequences = torch.from_numpy(sequences)
        self.lengths = torch.from_numpy(lengths)
        self.targets_tensor = torch.from_numpy(self.targets)
        self.query_ids_tensor = torch.from_numpy(self.query_ids)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "query_id": self.query_ids_tensor[idx],
            "item_seq": self.sequences[idx],
            "item_seq_len": self.lengths[idx],
            "target": self.targets_tensor[idx],
        }


class BaseSBRModel(nn.Module):
    """
    Lớp cơ sở cho cả 3 mô hình SBR (GRU4Rec, NARM, SASRec).
    Đảm bảo:
    1. Bảng item_embedding có đúng kích thước (num_items + 1, embedding_size) với padding_idx=0.
    2. Cố định và đóng băng tuyệt đối gradient của padding (ID 0) và 7,647 Controlled Cold Items.
    3. Hỗ trợ chấm điểm full-catalog kèm warm_mask để chỉ gợi ý trong 29,936 warm items.
    """

    def __init__(
        self,
        num_items_with_pad: int,
        embedding_size: int,
        trainable_warm_mask: torch.Tensor,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.num_items_with_pad = int(num_items_with_pad)
        self.embedding_size = int(embedding_size)
        self.padding_idx = int(padding_idx)

        self.item_embedding = nn.Embedding(
            self.num_items_with_pad,
            self.embedding_size,
            padding_idx=self.padding_idx,
        )

        # Đăng ký buffer mask boolean cho các warm item hợp lệ (ID 1..N có is_trainable_warm == True)
        mask_bool = trainable_warm_mask.to(dtype=torch.bool).clone()
        mask_bool[self.padding_idx] = False
        self.register_buffer("trainable_warm_mask", mask_bool)

        # Hook khóa cứng gradient = 0 đối với hàng padding (0) và toàn bộ 7,647 hàng Controlled Cold
        self._register_cold_gradient_freeze_hook()

    def _register_cold_gradient_freeze_hook(self) -> None:
        def _hook(grad: torch.Tensor) -> torch.Tensor:
            grad_masked = grad.clone()
            grad_masked[~self.trainable_warm_mask] = 0.0
            return grad_masked

        self.item_embedding.weight.register_hook(_hook)

    def gather_indexes(self, output: torch.Tensor, gather_index: torch.Tensor) -> torch.Tensor:
        """Trích xuất vector ẩn tại vị trí kết thúc thực tế (seq_len - 1) của mỗi phiên."""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def encode_session(
        self, item_seq: torch.Tensor, item_seq_len: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def full_sort_logits(
        self,
        item_seq: torch.Tensor,
        item_seq_len: torch.Tensor,
        mask_cold_and_pad: bool = True,
        mask_fill_value: float = -1.0e9,
    ) -> torch.Tensor:
        """Tính điểm tích vô hướng h_s @ E^T cho toàn bộ catalog và che điểm của cold/padding."""
        seq_output = self.encode_session(item_seq, item_seq_len)
        logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        if mask_cold_and_pad:
            logits = logits.masked_fill(~self.trainable_warm_mask.unsqueeze(0), mask_fill_value)
        return logits


class GRU4RecModel(BaseSBRModel):
    """
    GRU4Rec (Hidasi et al., ICLR 2016) — Kiến trúc chuẩn RecBole:
    Item Embedding -> Dropout -> GRU -> Dense Projection -> Dot Product với Item Embedding.
    """

    def __init__(
        self,
        num_items_with_pad: int,
        trainable_warm_mask: torch.Tensor,
        embedding_size: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        embedding_dropout: float = 0.25,
        hidden_dropout: float = 0.20,
        init_std: float = 0.02,
        padding_idx: int = 0,
    ):
        super().__init__(
            num_items_with_pad=num_items_with_pad,
            embedding_size=embedding_size,
            trainable_warm_mask=trainable_warm_mask,
            padding_idx=padding_idx,
        )
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.init_std = float(init_std)

        self.emb_dropout = nn.Dropout(embedding_dropout)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
            dropout=hidden_dropout if self.num_layers > 1 else 0.0,
        )
        self.out_dropout = nn.Dropout(hidden_dropout)
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.init_std)
        with torch.no_grad():
            self.item_embedding.weight[self.padding_idx].zero_()
        for name, param in self.gru_layers.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
        nn.init.xavier_normal_(self.dense.weight)
        if self.dense.bias is not None:
            nn.init.constant_(self.dense.bias, 0.0)

    def encode_session(
        self, item_seq: torch.Tensor, item_seq_len: torch.Tensor
    ) -> torch.Tensor:
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        last_hidden = self.gather_indexes(gru_output, item_seq_len - 1)
        last_hidden = self.out_dropout(last_hidden)
        seq_output = self.dense(last_hidden)
        return seq_output


class NARMModel(BaseSBRModel):
    """
    NARM (Li et al., CIKM 2017) — Kiến trúc chuẩn RecBole:
    Kết hợp Global Encoder (trạng thái cuối GRU h_t) và Local Encoder (Attention trên toàn chuỗi
    để nắm bắt ý định chính của người dùng trong phiên) -> Ghép [c_local; c_global] -> Bi-linear Dense.
    """

    def __init__(
        self,
        num_items_with_pad: int,
        trainable_warm_mask: torch.Tensor,
        embedding_size: int = 64,
        hidden_size: int = 128,
        n_layers: int = 1,
        embedding_dropout: float = 0.25,
        ct_dropout: float = 0.50,
        init_std: float = 0.02,
        padding_idx: int = 0,
    ):
        super().__init__(
            num_items_with_pad=num_items_with_pad,
            embedding_size=embedding_size,
            trainable_warm_mask=trainable_warm_mask,
            padding_idx=padding_idx,
        )
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)
        self.init_std = float(init_std)

        self.emb_dropout = nn.Dropout(embedding_dropout)
        self.gru = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            bias=False,
            batch_first=True,
        )
        self.a_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.a_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_t = nn.Linear(self.hidden_size, 1, bias=False)
        self.ct_dropout = nn.Dropout(ct_dropout)
        self.b = nn.Linear(2 * self.hidden_size, self.embedding_size, bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=self.init_std)
        with torch.no_grad():
            self.item_embedding.weight[self.padding_idx].zero_()
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
        for linear in (self.a_1, self.a_2, self.v_t, self.b):
            nn.init.xavier_normal_(linear.weight)

    def encode_session(
        self, item_seq: torch.Tensor, item_seq_len: torch.Tensor
    ) -> torch.Tensor:
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_out, _ = self.gru(item_seq_emb_dropout)

        # Global representation: trạng thái ẩn ở bước cuối của chuỗi
        c_global = h_t = self.gather_indexes(gru_out, item_seq_len - 1)

        # Local representation: cơ chế Attention tính trọng số của từng item trong phiên với h_t
        q1 = self.a_1(gru_out)
        q2 = self.a_2(h_t).unsqueeze(1)
        alpha = self.v_t(torch.sigmoid(q1 + q2))  # (B, L, 1)

        mask = item_seq.gt(0).unsqueeze(2)  # (B, L, 1)
        alpha = alpha.masked_fill(~mask, -1.0e9)
        attn_weights = torch.softmax(alpha, dim=1)
        c_local = torch.sum(attn_weights * gru_out, dim=1)

        c_t = torch.cat([c_local, c_global], dim=1)
        c_t = self.ct_dropout(c_t)
        seq_output = self.b(c_t)
        return seq_output


class SASRecTransformerBlock(nn.Module):
    """Một tầng Causal Self-Attention + Point-wise Feed-Forward chuẩn SASRec / RecBole."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float = 1e-12,
        hidden_act: str = "gelu",
    ):
        super().__init__()
        self.num_attention_heads = int(n_heads)
        self.attention_head_size = int(hidden_size // n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

        self.ffn_1 = nn.Linear(hidden_size, inner_size)
        self.ffn_2 = nn.Linear(inner_size, hidden_size)
        self.ffn_dropout = nn.Dropout(hidden_dropout_prob)
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.act_fn = F.gelu if hidden_act.lower() == "gelu" else F.relu

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        q = self._transpose_for_scores(self.query(hidden_states))
        k = self._transpose_for_scores(self.key(hidden_states))
        v = self._transpose_for_scores(self.value(hidden_states))

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
            self.attention_head_size
        )
        attn_scores = attn_scores + attention_mask
        attn_probs = nn.Softmax(dim=-1)(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)

        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.all_head_size,))

        attn_out = self.out_proj(context)
        attn_out = self.out_dropout(attn_out)
        hidden_states = self.layer_norm_1(hidden_states + attn_out)

        ffn_out = self.ffn_2(self.act_fn(self.ffn_1(hidden_states)))
        ffn_out = self.ffn_dropout(ffn_out)
        hidden_states = self.layer_norm_2(hidden_states + ffn_out)
        return hidden_states


class SASRecModel(BaseSBRModel):
    """
    SASRec (Kang & McAuley, ICDM 2018) — Kiến trúc chuẩn RecBole:
    Item Embedding + Positional Embedding -> LayerNorm + Dropout -> Causal Multi-Head Transformer -> Gather Last Position.
    """

    def __init__(
        self,
        num_items_with_pad: int,
        trainable_warm_mask: torch.Tensor,
        max_seq_length: int = 50,
        embedding_size: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        inner_size: int = 256,
        hidden_dropout_prob: float = 0.20,
        attn_dropout_prob: float = 0.20,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1.0e-12,
        initializer_range: float = 0.02,
        padding_idx: int = 0,
    ):
        super().__init__(
            num_items_with_pad=num_items_with_pad,
            embedding_size=embedding_size,
            trainable_warm_mask=trainable_warm_mask,
            padding_idx=padding_idx,
        )
        self.max_seq_length = int(max_seq_length)
        self.initializer_range = float(initializer_range)

        self.position_embedding = nn.Embedding(self.max_seq_length, self.embedding_size)
        self.layer_norm = nn.LayerNorm(self.embedding_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.blocks = nn.ModuleList(
            [
                SASRecTransformerBlock(
                    hidden_size=self.embedding_size,
                    n_heads=n_heads,
                    inner_size=inner_size,
                    hidden_dropout_prob=hidden_dropout_prob,
                    attn_dropout_prob=attn_dropout_prob,
                    layer_norm_eps=layer_norm_eps,
                    hidden_act=hidden_act,
                )
                for _ in range(int(n_layers))
            ]
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        with torch.no_grad():
            self.item_embedding.weight[self.padding_idx].zero_()

    def _get_causal_attention_mask(self, item_seq: torch.Tensor) -> torch.Tensor:
        """Tạo mặt nạ Causal (tam giác dưới) kết hợp mặt nạ Padding cho Self-Attention."""
        attention_mask = (item_seq > 0).long()  # (B, L)
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
        max_len = attention_mask.size(-1)
        subsequent_mask = torch.triu(
            torch.ones((1, max_len, max_len), device=item_seq.device), diagonal=1
        ).eq(0)
        subsequent_mask = subsequent_mask.unsqueeze(1).long()  # (1, 1, L, L)

        combined_mask = extended_mask * subsequent_mask
        combined_mask = (1.0 - combined_mask.to(dtype=torch.float32)) * -10000.0
        return combined_mask

    def encode_session(
        self, item_seq: torch.Tensor, item_seq_len: torch.Tensor
    ) -> torch.Tensor:
        seq_len = item_seq.size(1)
        position_ids = (
            torch.arange(seq_len, dtype=torch.long, device=item_seq.device)
            .unsqueeze(0)
            .expand_as(item_seq)
        )
        hidden_states = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        attention_mask = self._get_causal_attention_mask(item_seq)
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        seq_output = self.gather_indexes(hidden_states, item_seq_len - 1)
        return seq_output


def build_model_from_config(
    config: dict[str, Any],
    item_mapping: pd.DataFrame,
    device: torch.device,
) -> BaseSBRModel:
    """Khởi tạo đúng mô hình (GRU4Rec, NARM hoặc SASRec) cùng mask warm/cold chuẩn từ item_mapping."""
    num_items_with_pad = int(item_mapping["internal_item_id"].max()) + 1
    warm_mask = torch.zeros(num_items_with_pad, dtype=torch.bool)
    warm_ids = item_mapping.loc[
        item_mapping["is_trainable_warm"], "internal_item_id"
    ].to_numpy(dtype=np.int64)
    warm_mask[warm_ids] = True

    model_name = config["model_name"].lower()
    params = config["model_params"]
    padding_idx = int(config.get("padding_idx", 0))
    max_seq_len = int(config.get("max_seq_length", 50))

    if model_name == "gru4rec":
        model = GRU4RecModel(
            num_items_with_pad=num_items_with_pad,
            trainable_warm_mask=warm_mask,
            embedding_size=int(params.get("embedding_size", 64)),
            hidden_size=int(params.get("hidden_size", 128)),
            num_layers=int(params.get("num_layers", 1)),
            embedding_dropout=float(params.get("embedding_dropout", 0.25)),
            hidden_dropout=float(params.get("hidden_dropout", 0.20)),
            init_std=float(params.get("init_std", 0.02)),
            padding_idx=padding_idx,
        )
    elif model_name == "narm":
        model = NARMModel(
            num_items_with_pad=num_items_with_pad,
            trainable_warm_mask=warm_mask,
            embedding_size=int(params.get("embedding_size", 64)),
            hidden_size=int(params.get("hidden_size", 128)),
            n_layers=int(params.get("n_layers", 1)),
            embedding_dropout=float(params.get("embedding_dropout", 0.25)),
            ct_dropout=float(params.get("ct_dropout", 0.50)),
            init_std=float(params.get("init_std", 0.02)),
            padding_idx=padding_idx,
        )
    elif model_name == "sasrec":
        model = SASRecModel(
            num_items_with_pad=num_items_with_pad,
            trainable_warm_mask=warm_mask,
            max_seq_length=max_seq_len,
            embedding_size=int(params.get("embedding_size", 64)),
            n_layers=int(params.get("n_layers", 2)),
            n_heads=int(params.get("n_heads", 2)),
            inner_size=int(params.get("inner_size", 256)),
            hidden_dropout_prob=float(params.get("hidden_dropout_prob", 0.20)),
            attn_dropout_prob=float(params.get("attn_dropout_prob", 0.20)),
            hidden_act=str(params.get("hidden_act", "gelu")),
            layer_norm_eps=float(params.get("layer_norm_eps", 1e-12)),
            initializer_range=float(params.get("initializer_range", 0.02)),
            padding_idx=padding_idx,
        )
    else:
        raise ValueError(f"Mô hình không hợp lệ: {model_name}")

    return model.to(device)


@torch.no_grad()
def predict_topk_for_dataset(
    model: BaseSBRModel,
    dataset: SessionPrefixDataset,
    device: torch.device,
    batch_size: int = 1024,
    topk: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sinh dự đoán Top-K (internal_item_id và score) trên toàn bộ dataset.
    Chỉ cho phép chọn ứng viên trong tập 29,936 warm items (mask cold items & padding = -1e9).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_topk_ids = np.empty((len(dataset), topk), dtype=np.int64)
    all_topk_scores = np.empty((len(dataset), topk), dtype=np.float32)

    offset = 0
    for batch in loader:
        item_seq = batch["item_seq"].to(device, non_blocking=True)
        item_seq_len = batch["item_seq_len"].to(device, non_blocking=True)

        logits = model.full_sort_logits(
            item_seq=item_seq,
            item_seq_len=item_seq_len,
            mask_cold_and_pad=True,
        )
        scores, top_indices = torch.topk(logits, k=topk, dim=-1)

        bsz = item_seq.size(0)
        all_topk_ids[offset : offset + bsz] = top_indices.cpu().numpy()
        all_topk_scores[offset : offset + bsz] = scores.cpu().numpy()
        offset += bsz

    return all_topk_ids, all_topk_scores


def build_top20_dataframe(
    dataset: SessionPrefixDataset,
    topk_internal_ids: np.ndarray,
    topk_scores: np.ndarray,
    internal_to_raw_id: dict[int, int] | np.ndarray,
) -> pd.DataFrame:
    """
    Xuất bảng top20.parquet theo đúng quy ước tại Mục 9.6 tài liệu CAU-TRUC-DU-AN.md:
    Các cột: session_id, query_id, rank (1..20), item_id (raw), internal_item_id, score.
    """
    n_queries, k = topk_internal_ids.shape
    session_col = np.repeat(dataset.session_ids, k)
    query_col = np.repeat(dataset.query_ids, k)
    rank_col = np.tile(np.arange(1, k + 1, dtype=np.int32), n_queries)
    internal_col = topk_internal_ids.ravel()
    score_col = topk_scores.ravel()

    if isinstance(internal_to_raw_id, np.ndarray):
        raw_col = internal_to_raw_id[internal_col]
    else:
        raw_col = np.array([internal_to_raw_id[int(i)] for i in internal_col], dtype=np.int64)

    return pd.DataFrame(
        {
            "session_id": session_col,
            "query_id": query_col,
            "rank": rank_col,
            "item_id": raw_col,
            "internal_item_id": internal_col,
            "score": score_col,
        }
    )


def train_and_evaluate_seed(
    model_name: str,
    seed: int,
    config: dict[str, Any] | None = None,
    processed_dir: Path = PROCESSED_DIGI_DIR,
    max_epochs_override: int | None = None,
    device_str: str | None = None,
) -> dict[str, Any]:
    """
    Thực thi trọn vẹn chu trình Huấn luyện -> Early Stopping trên Validation -> Đánh giá trên Test ->
    Sinh top20.parquet -> Trích xuất item_embeddings.npy -> Kiểm định Cold-start Option B ->
    Xuất bảng phân tích item-level cho một cặp (model_name, seed).
    """
    ensure_output_dirs()
    cfg = copy.deepcopy(config) if config is not None else load_experiment_config(model_name)
    cfg["seed"] = int(seed)
    if max_epochs_override is not None:
        cfg["training"]["epochs"] = int(max_epochs_override)

    set_global_seed(seed)

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = get_run_dir(model_name, seed)
    logger = create_run_logger(run_dir / "training.log", f"{model_name}_seed_{seed}")

    logger.info("=" * 72)
    logger.info(f"BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH {model_name.upper()} | SEED = {seed}")
    logger.info(f"Thiết bị chạy (Device) : {device}")
    logger.info(f"Thư mục lưu kết quả    : {run_dir}")
    logger.info("=" * 72)

    # 1. Đọc item_mapping và dữ liệu train/val/test
    item_mapping = load_item_mapping(processed_dir)
    max_internal_id = int(item_mapping["internal_item_id"].max())
    internal_to_raw = np.zeros(max_internal_id + 1, dtype=np.int64)
    internal_to_raw[item_mapping["internal_item_id"].to_numpy(dtype=np.int64)] = (
        item_mapping["item_id"].to_numpy(dtype=np.int64)
    )

    max_seq_len = int(cfg.get("max_seq_length", 50))
    topk = int(cfg["evaluation"].get("topk", 20))

    train_df = pd.read_parquet(processed_dir / "train_examples.parquet")
    val_df = pd.read_parquet(processed_dir / "validation_examples.parquet")
    test_df = pd.read_parquet(processed_dir / "test_examples.parquet")
    test_last_df = pd.read_parquet(processed_dir / "test_last_item_examples.parquet")
    test_events_full = pd.read_parquet(processed_dir / "test_events_full.parquet")

    train_dataset = SessionPrefixDataset(train_df, max_seq_length=max_seq_len)
    val_dataset = SessionPrefixDataset(val_df, max_seq_length=max_seq_len)
    test_dataset = SessionPrefixDataset(test_df, max_seq_length=max_seq_len)
    test_last_dataset = SessionPrefixDataset(test_last_df, max_seq_length=max_seq_len)

    train_batch_size = int(cfg["training"].get("train_batch_size", 512))
    eval_batch_size = int(cfg["training"].get("eval_batch_size", 1024))

    g = torch.Generator()
    g.manual_seed(int(seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=g,
    )

    # 2. Khởi tạo mô hình và lưu lại bản chụp embedding khởi tạo của Controlled Cold Items
    model = build_model_from_config(cfg, item_mapping, device)
    cold_internal_ids = item_mapping.loc[
        item_mapping["is_controlled_cold"], "internal_item_id"
    ].to_numpy(dtype=np.int64)

    initial_cold_embeddings = (
        model.item_embedding.weight.detach().cpu().numpy()[cold_internal_ids].copy()
    )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Tổng số item trong bảng Embedding: {max_internal_id:,} (+1 hàng padding 0) | "
        f"Warm trainable: {int(item_mapping['is_trainable_warm'].sum()):,} | "
        f"Controlled cold (đóng băng): {len(cold_internal_ids):,}"
    )
    logger.info(f"Tổng số tham số mô hình: {total_params:,}")

    lr = float(cfg["training"].get("learning_rate", 0.001))
    weight_decay = float(cfg["training"].get("weight_decay", 0.0))
    grad_clip = float(cfg["training"].get("grad_clip_norm", 5.0))
    epochs = int(cfg["training"].get("epochs", 30))
    use_amp = bool(cfg["training"].get("use_amp", True)) and (device.type == "cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    monitor_metric = str(cfg["early_stopping"].get("monitor", "mrr@20")).lower()
    patience = int(cfg["early_stopping"].get("patience", 5))
    min_delta = float(cfg["early_stopping"].get("min_delta", 0.0001))

    best_score = -float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history_rows: list[dict[str, Any]] = []
    checkpoint_path = run_dir / "checkpoint.pth"

    # 3. Vòng lặp huấn luyện qua các Epoch
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        total_samples = 0

        progress = tqdm(
            train_loader,
            desc=f"[{model_name.upper()} seed={seed}] Epoch {epoch:02d}/{epochs:02d}",
            leave=False,
        )
        for batch in progress:
            item_seq = batch["item_seq"].to(device, non_blocking=True)
            item_seq_len = batch["item_seq_len"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model.full_sort_logits(
                    item_seq=item_seq,
                    item_seq_len=item_seq_len,
                    mask_cold_and_pad=True,
                )
                loss = F.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bsz = item_seq.size(0)
            running_loss += float(loss.item()) * bsz
            total_samples += bsz
            progress.set_postfix(loss=f"{running_loss / max(1, total_samples):.4f}")

        train_loss = running_loss / max(1, total_samples)

        # Đánh giá trên tập Validation Warm
        val_topk_ids, _ = predict_topk_for_dataset(
            model=model,
            dataset=val_dataset,
            device=device,
            batch_size=eval_batch_size,
            topk=topk,
        )
        val_metrics = compute_ranking_metrics_from_predictions(
            val_topk_ids, val_dataset.targets, k=topk
        )
        elapsed = time.perf_counter() - t0

        current_score = float(val_metrics[monitor_metric])
        is_best = current_score > (best_score + min_delta)

        if is_best:
            best_score = current_score
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(
                {
                    "model_name": model_name.lower(),
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "best_validation_score": float(best_score),
                    "monitor_metric": monitor_metric,
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                },
                checkpoint_path,
            )
        else:
            epochs_no_improve += 1

        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            f"validation_recall@{topk}": float(val_metrics[f"recall@{topk}"]),
            f"validation_mrr@{topk}": float(val_metrics[f"mrr@{topk}"]),
            f"validation_ndcg@{topk}": float(val_metrics[f"ndcg@{topk}"]),
            "validation_score": current_score,
            "is_best": bool(is_best),
            "elapsed_time": float(elapsed),
        }
        history_rows.append(row)

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | Loss: {train_loss:.4f} | "
            f"Val Recall@{topk}: {val_metrics[f'recall@{topk}']:.4f} | "
            f"Val MRR@{topk}: {val_metrics[f'mrr@{topk}']:.4f} | "
            f"Val NDCG@{topk}: {val_metrics[f'ndcg@{topk}']:.4f} | "
            f"Time: {elapsed:.1f}s {'[BEST]' if is_best else ''}"
        )

        if epochs_no_improve >= patience:
            logger.info(
                f"Early stopping kích hoạt tại epoch {epoch} "
                f"(không cải thiện {monitor_metric} sau {patience} epochs). "
                f"Best epoch = {best_epoch} ({best_score:.4f})."
            )
            break

    # 4. Nạp lại Best Checkpoint và kiểm tra tính bất biến của Controlled Cold Embeddings
    best_ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Trích xuất ma trận item embedding (37583, D) từ index 1..37583 (bỏ hàng 0 padding)
    full_weight_np = model.item_embedding.weight.detach().cpu().numpy()
    trained_cold_embeddings = full_weight_np[cold_internal_ids]
    max_cold_drift = float(np.max(np.abs(trained_cold_embeddings - initial_cold_embeddings)))
    if max_cold_drift > 1e-7:
        raise AssertionError(
            f"Rò rỉ gradient vào Controlled Cold Items! max_abs_drift = {max_cold_drift}"
        )
    logger.info(
        f"Kiểm định Option B: Độ lệch tối đa của 7,647 Controlled Cold Embeddings = {max_cold_drift:.8f} (Chưa huấn luyện 100%)."
    )

    item_embeddings = full_weight_np[1 : max_internal_id + 1].astype(np.float32)

    # 5. Đánh giá trên tập Test chính (All-prefix Warm) & xuất top20.parquet
    test_topk_ids, test_topk_scores = predict_topk_for_dataset(
        model=model,
        dataset=test_dataset,
        device=device,
        batch_size=eval_batch_size,
        topk=topk,
    )
    test_metrics_main = compute_ranking_metrics_from_predictions(
        test_topk_ids, test_dataset.targets, k=topk
    )
    test_stratified_df = compute_stratified_ranking_metrics(
        test_topk_ids,
        test_dataset.targets,
        test_dataset.target_groups,
        k=topk,
    )

    # Đánh giá bổ sung trên tập Last-item Test (12,791 phiên)
    test_last_topk_ids, _ = predict_topk_for_dataset(
        model=model,
        dataset=test_last_dataset,
        device=device,
        batch_size=eval_batch_size,
        topk=topk,
    )
    test_metrics_last_item = compute_ranking_metrics_from_predictions(
        test_last_topk_ids, test_last_dataset.targets, k=topk
    )

    # Lưu top20.parquet
    top20_df = build_top20_dataframe(
        dataset=test_dataset,
        topk_internal_ids=test_topk_ids,
        topk_scores=test_topk_scores,
        internal_to_raw_id=internal_to_raw,
    )
    top20_df.to_parquet(run_dir / "top20.parquet", index=False, compression="zstd")

    # 6. Tính chỉ số Popularity Bias đầu ra & Cold-start Diagnostic
    bias_summary, item_rec_catalog, group_exposure_df = evaluate_top20_popularity_bias(
        top20_df=top20_df,
        item_mapping=item_mapping,
        k=topk,
    )
    cold_diagnostic = compute_cold_start_test_diagnostic(
        warm_metrics=test_metrics_main,
        test_events_full=test_events_full,
        warm_test_examples_count=len(test_dataset),
        k=topk,
    )

    # 7. Lưu toàn bộ 8 file chuẩn vào runs/{model}/seed_{seed}/ và bảng item-level
    save_yaml(cfg, run_dir / "config.yaml")

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(run_dir / "training_history.csv", index=False)

    np.save(run_dir / "item_embeddings.npy", item_embeddings)

    embedding_mapping_df = item_mapping.copy()
    embedding_mapping_df.insert(0, "embedding_row_index", np.arange(len(embedding_mapping_df)))
    embedding_mapping_df.to_csv(run_dir / "item_embedding_mapping.csv", index=False)

    item_level_df = build_item_level_analysis_table(
        model_name=model_name,
        seed=seed,
        embeddings=item_embeddings,
        item_rec_catalog=item_rec_catalog,
    )
    item_level_path = ITEM_LEVEL_DIR / f"{model_name.lower()}_seed{seed}.parquet"
    item_level_df.to_parquet(item_level_path, index=False, compression="zstd")

    full_test_metrics_payload = {
        "model": model_name.lower(),
        "architecture": cfg.get("architecture", model_name.upper()),
        "seed": int(seed),
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_validation_score": float(best_score),
        "monitor_metric": monitor_metric,
        "cold_embedding_max_abs_drift": max_cold_drift,
        "test_warm_all_prefix": test_metrics_main,
        "test_warm_last_item": test_metrics_last_item,
        "test_warm_by_popularity_group": test_stratified_df.to_dict(orient="records"),
        "popularity_bias": bias_summary,
        "exposure_by_group": group_exposure_df.to_dict(orient="records"),
        "cold_start_diagnostic": cold_diagnostic,
    }
    save_json(full_test_metrics_payload, run_dir / "test_metrics.json")

    logger.info("-" * 72)
    logger.info(
        f"[KẾT QUẢ TEST WARM] Recall@{topk}: {test_metrics_main[f'recall@{topk}']:.4f} | "
        f"MRR@{topk}: {test_metrics_main[f'mrr@{topk}']:.4f} | "
        f"NDCG@{topk}: {test_metrics_main[f'ndcg@{topk}']:.4f}"
    )
    logger.info(
        f"[TEST LAST-ITEM]    Recall@{topk}: {test_metrics_last_item[f'recall@{topk}']:.4f} | "
        f"MRR@{topk}: {test_metrics_last_item[f'mrr@{topk}']:.4f} | "
        f"NDCG@{topk}: {test_metrics_last_item[f'ndcg@{topk}']:.4f}"
    )
    logger.info(
        f"[POPULARITY BIAS]   Coverage_warm@{topk}: {bias_summary[f'coverage_warm@{topk}']:.4f} | "
        f"Gini_warm@{topk}: {bias_summary[f'gini_warm@{topk}']:.4f} | "
        f"ARP@{topk}: {bias_summary[f'arp@{topk}']:.2f}"
    )
    logger.info(f"Đã xuất đầy đủ kết quả vào: {run_dir} và {item_level_path}")
    logger.info("=" * 72)

    return {
        "run_dir": run_dir,
        "history_df": history_df,
        "test_metrics": full_test_metrics_payload,
        "test_stratified_df": test_stratified_df,
        "group_exposure_df": group_exposure_df,
        "item_level_df": item_level_df,
    }
