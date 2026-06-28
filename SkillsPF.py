# -*- coding: utf-8 -*-

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.nn.utils.weight_norm import weight_norm


class FCNet(nn.Module):
    def __init__(self, dims, act: str = 'ReLU', dropout: float = 0.0):
        super().__init__()
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        if act:
            layers.append(getattr(nn, act)())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BAFN(nn.Module):
    def __init__(
        self,
        l_dim: int,
        c_dim: int,
        h_dim: int,
        h_out: int,
        act: str = 'ReLU',
        dropout: float = 0.2,
        k: int = 3,
        norm_type: str = 'layernorm',
    ):
        super().__init__()
        self.k = k
        self.h_out = h_out
        self.U = FCNet([l_dim, h_dim * self.k], act=act, dropout=dropout)
        self.M = FCNet([c_dim, h_dim * self.k], act=act, dropout=dropout)
        self.q_mat = nn.Parameter(torch.Tensor(1, h_out, 1, h_dim * self.k).normal_())
        self.q_bias = nn.Parameter(torch.Tensor(1, h_out, 1, 1).normal_())
        if k > 1:
            self.pooling = nn.AvgPool1d(k, stride=k)

        norm_type = norm_type.lower()
        if norm_type == 'batchnorm':
            self.norm = nn.BatchNorm1d(h_dim)
        elif norm_type == 'layernorm':
            self.norm = nn.LayerNorm(h_dim)
        elif norm_type == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError(f'Unsupported norm_type: {norm_type}')

    def attention_pooling(self, v, q, att_map):
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v, att_map, q))
        if self.k > 1:
            fusion_logits = fusion_logits.unsqueeze(1)
            fusion_logits = self.pooling(fusion_logits).squeeze(1) * self.k
        return fusion_logits

    def forward(self, l, c, softmax: bool = False, v_mask: bool = True, mask_with: float = 0.0):
        l_num = l.size(1)
        c_num = c.size(1)
        l_ = self.U(l)
        c_ = self.M(c)
        att_maps = torch.einsum('xhyk,bvk,bqk->bhvq', (self.q_mat, l_, c_)) + self.q_bias

        if v_mask:
            mask = (l.abs().sum(2).unsqueeze(1).unsqueeze(3).expand_as(att_maps) == 0)
            att_maps = att_maps.masked_fill(mask, mask_with)

        if softmax:
            p = nn.functional.softmax(att_maps.view(-1, self.h_out, l_num * c_num), dim=2)
            att_maps = p.view(-1, self.h_out, l_num, c_num)

        logits = self.attention_pooling(l_, c_, att_maps[:, 0, :, :])
        for i in range(1, self.h_out):
            logits = logits + self.attention_pooling(l_, c_, att_maps[:, i, :, :])
        logits = self.norm(logits)
        return logits, att_maps


class SkillsPF_model(nn.Module):
    """Skill 级源码语义 + BiGRU 上下文化污点路径语义融合分类器。"""

    def __init__(
        self,
        code_embed_dim: int,
        taint_embed_dim: int,
        gru_hidden_dim: int,
        gru_num_layers: int,
        bafn_output_dim: int,
        dropout: float,
        device,
        norm_type: str = 'layernorm',
        # =========================
        # 【修改位置 1】新增污点路径 BiGRU 参数
        # =========================
        taint_gru_hidden_dim: int = 128,
        taint_gru_num_layers: int = 1,
    ):
        super().__init__()
        self.device = device
        self.taint_embed_dim = taint_embed_dim
        self.taint_gru_hidden_dim = taint_gru_hidden_dim
        self.taint_out_dim = 2 * taint_gru_hidden_dim

        # =========================
        # 源码上下文路径：代码行向量 -> BiGRU -> BAFN
        # =========================
        self.gru = nn.GRU(
            input_size=code_embed_dim,
            hidden_size=gru_hidden_dim,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0.0,
        )

        self.bafn = weight_norm(
            BAFN(
                l_dim=code_embed_dim,
                c_dim=2 * gru_hidden_dim,
                h_dim=bafn_output_dim,
                h_out=2,
                dropout=dropout,
                norm_type=norm_type,
            ),
            name='q_mat',
            dim=None,
        )

        # =========================
        # 【修改位置 2】污点路径语义路径：多条路径向量 -> BiGRU -> Skill级污点语义
        # =========================
        self.taint_gru = nn.GRU(
            input_size=taint_embed_dim,
            hidden_size=taint_gru_hidden_dim,
            num_layers=taint_gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if taint_gru_num_layers > 1 else 0.0,
        )
        self.taint_norm = nn.LayerNorm(self.taint_out_dim)

        # =========================
        # 【修改位置 3】融合维度从 bafn_output_dim + taint_embed_dim
        # 改为 bafn_output_dim + 2 * taint_gru_hidden_dim
        # =========================
        self.fusion_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bafn_output_dim + self.taint_out_dim, 1),
        )

    def _encode_code_context(self, code_tensor: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """源码路径：代码行编码序列 -> GRU上下文 -> BAFN池化。"""
        if len(code_tensor) == 0:
            raise ValueError('code_tensor is empty')
        for item in code_tensor:
            if item.size(0) == 0:
                raise ValueError('A Skill has zero line embeddings. Please filter empty examples first.')

        sent_lengths = torch.tensor([code.size(0) for code in code_tensor], dtype=torch.long, device=self.device)
        padded_code = pad_sequence(code_tensor, batch_first=True)

        sorted_lengths, perm_idx = sent_lengths.sort(dim=0, descending=True)
        sorted_code = padded_code[perm_idx]

        packed_sents = pack_padded_sequence(
            sorted_code,
            lengths=sorted_lengths.detach().cpu().tolist(),
            batch_first=True,
            enforce_sorted=True,
        )

        line_contexts, _ = self.gru(packed_sents)
        line_contexts, _ = pad_packed_sequence(line_contexts, batch_first=True)

        _, unperm_idx = perm_idx.sort(dim=0, descending=False)
        sorted_code = sorted_code[unperm_idx]
        line_contexts = line_contexts[unperm_idx]

        file_level_embeds, sent_att = self.bafn(sorted_code, line_contexts)
        return file_level_embeds, sent_att, sorted_code

    def _encode_taint_context(
        self,
        taint_path_embeds: Optional[List[torch.Tensor]],
        reference_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        【修改位置 4】污点路径分支核心函数。

        输入：
            taint_path_embeds: List[Tensor]
                每个元素对应一个 Skill。
                若该 Skill 有 k 条污点路径，则形状为 [k, taint_embed_dim]；
                若该 Skill 没有污点路径，则形状为 [0, taint_embed_dim] 或 None。

        输出：
            taint_skill_embeds: Tensor，形状为 [batch_size, 2 * taint_gru_hidden_dim]
        """
        batch_size = reference_tensor.size(0)
        device = reference_tensor.device
        dtype = reference_tensor.dtype

        taint_skill_embeds = torch.zeros(batch_size, self.taint_out_dim, dtype=dtype, device=device)
        if taint_path_embeds is None:
            return taint_skill_embeds

        valid_indices: List[int] = []
        valid_seqs: List[torch.Tensor] = []
        valid_lengths: List[int] = []

        for idx, seq in enumerate(taint_path_embeds):
            if seq is None or seq.numel() == 0 or seq.size(0) == 0:
                continue
            seq = seq.to(device=device, dtype=dtype)
            if seq.dim() != 2:
                raise ValueError(f'taint_path_embeds[{idx}] must be 2D [path_num, hidden], got {tuple(seq.shape)}')
            if seq.size(-1) != self.taint_embed_dim:
                raise ValueError(
                    f'taint_path_embeds[{idx}] hidden dim mismatch: '
                    f'expected {self.taint_embed_dim}, got {seq.size(-1)}'
                )
            valid_indices.append(idx)
            valid_seqs.append(seq)
            valid_lengths.append(seq.size(0))

        if not valid_seqs:
            return taint_skill_embeds

        lengths = torch.tensor(valid_lengths, dtype=torch.long, device=device)
        padded = pad_sequence(valid_seqs, batch_first=True)  # [valid_batch, max_paths, taint_embed_dim]

        sorted_lengths, perm_idx = lengths.sort(dim=0, descending=True)
        sorted_padded = padded[perm_idx]

        packed = pack_padded_sequence(
            sorted_padded,
            lengths=sorted_lengths.detach().cpu().tolist(),
            batch_first=True,
            enforce_sorted=True,
        )
        taint_contexts, _ = self.taint_gru(packed)
        taint_contexts, _ = pad_packed_sequence(taint_contexts, batch_first=True)

        _, unperm_idx = perm_idx.sort(dim=0, descending=False)
        taint_contexts = taint_contexts[unperm_idx]
        lengths = lengths[unperm_idx]

        # masked mean pooling：对每个 Skill 的多条上下文化路径向量求平均
        max_len = taint_contexts.size(1)
        mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(dtype=taint_contexts.dtype)
        pooled = (taint_contexts * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1).to(dtype=taint_contexts.dtype)
        pooled = self.taint_norm(pooled)

        for local_i, batch_i in enumerate(valid_indices):
            taint_skill_embeds[batch_i] = pooled[local_i]

        return taint_skill_embeds

    def forward(
        self,
        code_tensor: List[torch.Tensor],
        taint_path_embeds: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数：
            code_tensor:
                List[Tensor]，每个 Skill 的代码行向量序列，形状为 [line_num, code_embed_dim]。
            taint_path_embeds:
                List[Tensor]，每个 Skill 的污点路径向量序列，形状为 [path_num, taint_embed_dim]。

        返回：
            scores: [batch_size, 1]
            sent_att_weights: 当前任务保留占位，形状为 [batch_size, max_line_num]
        """
        file_level_embeds, sent_att, sorted_code = self._encode_code_context(code_tensor)

        # 【修改位置 5】使用 BiGRU 得到上下文化 Skill 级污点路径语义，而不是外部 mean pooling
        taint_skill_embeds = self._encode_taint_context(taint_path_embeds, reference_tensor=file_level_embeds)

        enhanced_embeds = torch.cat([file_level_embeds, taint_skill_embeds], dim=-1)
        scores = self.fusion_fc(enhanced_embeds)

        # 当前任务不使用行级注意力。为了减少无意义数值风险，返回零矩阵占位。
        sent_att_weights = torch.zeros(
            file_level_embeds.size(0),
            sorted_code.size(1),
            dtype=file_level_embeds.dtype,
            device=file_level_embeds.device,
        )
        return scores, sent_att_weights
