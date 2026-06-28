# -*- coding: utf-8 -*-
"""
test_taint_fusion_bafline_bigru.py

【用途】
独立测试脚本：加载 train_taint_fusion_bafline_bigru.py 保存的最佳模型，
只在测试集上进行评估，并保存测试预测结果。

适配训练脚本：
- train_taint_fusion_bafline_bigru.py

关键点：
1. checkpoint 默认路径改为：
   /root/BATaint/output/bafline_taint_fusion_bigru/best_model.pt
2. 测试阶段不再对多条污点路径做 mean pooling；
3. 测试阶段保留 List[Tensor[path_num, hidden]]；
4. 将污点路径序列输入 BAFLineTaintFusion 内部的 taint_gru；
5. 构建模型时传入 taint_gru_hidden_dim 和 taint_gru_num_layers；
6. 加载 checkpoint 前检查 checkpoint 是否为 BiGRU 版，避免误加载旧版模型。
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from math import sqrt
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from SkillsPF import SkillsPF_model
from my_util import (
    SkillCodeTaintDataset,
    bool_str,
    collate_skill_taint_batch,
    load_skill_examples,
    set_seed,
    stratified_split_examples,
    summarize_examples,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Independent test script for BAFLine taint fusion BiGRU model.')

    parser.add_argument('--data_csv', type=str, default='/root/BATaint/datasets/skills_ground_truth_lines_dataset.csv')
    parser.add_argument('--repo_root', type=str, default='/root/BATaint/datasets/repo')
    parser.add_argument('--encoder_name_or_path', type=str, default='/root/BATaint/unixcoder-base-nine')

    # 注意：BiGRU 版训练脚本默认输出到 bafline_taint_fusion_bigru
    parser.add_argument('--output_dir', type=str, default='/root/BATaint/output/bafline_taint_fusion_bigru')
    parser.add_argument('--checkpoint_path', type=str, default='')
    parser.add_argument('--split_csv', type=str, default='')
    parser.add_argument('--test_output_dir', type=str, default='')

    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--code_block_size', type=int, default=75)
    parser.add_argument('--taint_block_size', type=int, default=256)
    parser.add_argument('--max_lines', type=int, default=900)
    parser.add_argument('--max_taint_paths', type=int, default=32)
    parser.add_argument('--encoder_line_batch_size', type=int, default=128)
    parser.add_argument('--encoder_taint_batch_size', type=int, default=32)
    parser.add_argument('--taint_result_name', type=str, default='taint_result.json')

    parser.add_argument('--gru_hidden_dim', type=int, default=32)
    parser.add_argument('--gru_num_layers', type=int, default=1)
    parser.add_argument('--bafn_hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--norm_type', type=str, default='layernorm', choices=['layernorm', 'batchnorm', 'none'])

    # BiGRU 污点路径分支参数：必须和训练阶段一致
    parser.add_argument('--taint_gru_hidden_dim', type=int, default=32)
    parser.add_argument('--taint_gru_num_layers', type=int, default=1)

    parser.add_argument('--drop_blank', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--drop_comment', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fine_tune_encoder', action='store_true', default=False)
    parser.add_argument('--threshold', type=float, default=0.5)

    parser.add_argument(
        '--ignore_checkpoint_args',
        action='store_true',
        default=False,
        help='默认使用 checkpoint 中保存的训练参数构建测试模型。设置该参数后优先使用命令行参数。',
    )

    return parser.parse_args()


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def merge_checkpoint_args(args, checkpoint_args: Dict):
    """
    使用 checkpoint 中的训练参数恢复模型结构和数据处理参数。
    注意：checkpoint_path / test_output_dir 仍保留测试脚本指定值。
    """
    if args.ignore_checkpoint_args or not checkpoint_args:
        return args

    keep_current_fields = {
        'output_dir',
        'checkpoint_path',
        'split_csv',
        'test_output_dir',
        'ignore_checkpoint_args',
    }

    for key, value in checkpoint_args.items():
        if key in keep_current_fields:
            continue
        if hasattr(args, key):
            setattr(args, key, value)

    return args


def get_pad_token_id(tokenizer) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 1


def encode_one_matrix(encoder, input_ids: torch.Tensor, pad_token_id: int, device, mini_batch_size: int):
    """编码二维 input_ids 矩阵，返回 CLS 向量矩阵。"""
    if input_ids.size(0) == 0:
        raise ValueError('input_ids has zero rows')

    if mini_batch_size <= 0:
        mini_batch_size = input_ids.size(0)

    all_embeds = []

    for start in range(0, input_ids.size(0), mini_batch_size):
        chunk = input_ids[start:start + mini_batch_size].to(device)
        attention_mask = chunk.ne(pad_token_id).long()
        outputs = encoder(input_ids=chunk, attention_mask=attention_mask)
        all_embeds.append(outputs.last_hidden_state[:, 0, :])

    return torch.cat(all_embeds, dim=0)


def encode_batch_code_embeddings(
    encoder,
    code_input_ids_list: List[torch.Tensor],
    pad_token_id: int,
    device,
    line_batch_size: int,
):
    """
    源码路径：
    每个 Skill 的多行代码分别由 UniXCoder 编码。
    返回 List[Tensor[line_num, hidden]]。
    """
    embeddings = []

    with torch.no_grad():
        for input_ids in code_input_ids_list:
            embeddings.append(
                encode_one_matrix(
                    encoder=encoder,
                    input_ids=input_ids,
                    pad_token_id=pad_token_id,
                    device=device,
                    mini_batch_size=line_batch_size,
                )
            )

    return embeddings


def encode_batch_taint_path_embeddings(
    encoder,
    taint_input_ids_list: List[torch.Tensor],
    has_taint_list: List[bool],
    pad_token_id: int,
    device,
    taint_batch_size: int,
    hidden_size: int,
) -> List[torch.Tensor]:
    """
    BiGRU 污点路径测试逻辑：
    - 有污点路径：返回 Tensor[path_num, hidden_size]
    - 无污点路径：返回空 Tensor[0, hidden_size]

    注意：
    这里不能 mean pooling。
    多条污点路径之间的前后传播关系由模型内部 taint_gru 建模。
    """
    taint_path_embeds: List[torch.Tensor] = []

    with torch.no_grad():
        for input_ids, has_taint in zip(taint_input_ids_list, has_taint_list):
            if not has_taint:
                taint_path_embeds.append(
                    torch.empty(0, hidden_size, dtype=torch.float, device=device)
                )
                continue

            path_embeds = encode_one_matrix(
                encoder=encoder,
                input_ids=input_ids,
                pad_token_id=pad_token_id,
                device=device,
                mini_batch_size=taint_batch_size,
            )
            taint_path_embeds.append(path_embeds)

    return taint_path_embeds


def build_dataloader(examples, tokenizer, args, shuffle: bool = False):
    dataset = SkillCodeTaintDataset(
        examples=examples,
        tokenizer=tokenizer,
        repo_root=args.repo_root,
        code_block_size=args.code_block_size,
        taint_block_size=args.taint_block_size,
        max_lines=args.max_lines,
        max_taint_paths=args.max_taint_paths,
        taint_result_name=args.taint_result_name,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        collate_fn=collate_skill_taint_batch,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    if not y_true:
        return {
            'recall': 0.0,
            'f1': 0.0,
            'gmean': 0.0,
            'mcc': 0.0,
            'auc': None,
        }

    # 如果只有单类，confusion_matrix(...).ravel() 会报错，因此显式固定 labels
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    r = metrics.recall_score(y_true, y_pred, zero_division=0)
    F_measure = metrics.f1_score(y_true, y_pred, zero_division=0)

    a = TP + FP
    b = TP + FN
    c = TN + FP
    d = TN + FN

    if a * b * c * d != 0:
        MCC = (TP * TN - FP * FN) / sqrt(a * b * c * d)
    else:
        MCC = 0.0

    TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    TNR = TN / (FP + TN) if (FP + TN) > 0 else 0.0
    G_mean = sqrt(TPR * TNR)

    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) >= 2 else None

    return {
        'recall': float(r),
        'f1': float(F_measure),
        'gmean': float(G_mean),
        'mcc': float(MCC),
        'auc': None if auc is None else float(auc),
    }


def evaluate(
    model,
    encoder,
    dataloader,
    criterion,
    pad_token_id: int,
    device,
    hidden_size: int,
    threshold: float,
    args,
    desc: str = 'Test',
):
    model.eval()
    encoder.eval()

    total_loss = []
    all_probs, all_labels, all_metas = [], [], []

    with torch.no_grad():
        for code_ids_list, taint_ids_list, has_taint_list, labels, metas in tqdm(dataloader, desc=desc):
            labels = labels.to(device).view(-1, 1)

            code_embeds = encode_batch_code_embeddings(
                encoder=encoder,
                code_input_ids_list=code_ids_list,
                pad_token_id=pad_token_id,
                device=device,
                line_batch_size=args.encoder_line_batch_size,
            )

            # 关键：这里返回 List[Tensor[path_num, hidden]]，不是 Tensor[batch, hidden]
            taint_path_embeds = encode_batch_taint_path_embeddings(
                encoder=encoder,
                taint_input_ids_list=taint_ids_list,
                has_taint_list=has_taint_list,
                pad_token_id=pad_token_id,
                device=device,
                taint_batch_size=args.encoder_taint_batch_size,
                hidden_size=hidden_size,
            )

            logits, _ = model(code_embeds, taint_path_embeds)

            loss = criterion(logits, labels)
            total_loss.append(loss.item())

            probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            labs = labels.detach().cpu().numpy().reshape(-1).astype(int)

            all_probs.extend(probs.tolist())
            all_labels.extend(labs.tolist())
            all_metas.extend(metas)

    pred_labels = [1 if p >= threshold else 0 for p in all_probs]
    metric_values = compute_metrics(all_labels, pred_labels, all_probs)
    metric_values['loss'] = float(np.mean(total_loss)) if total_loss else 0.0

    return metric_values, all_labels, pred_labels, all_probs, all_metas


def save_predictions(output_dir, y_true, y_pred, y_prob, metas):
    rows = []

    for true_label, pred_label, prob, meta in zip(y_true, y_pred, y_prob, metas):
        rows.append({
            'url': meta.get('url', ''),
            'skill_name': meta.get('skill_name', ''),
            'true_label': bool_str(true_label),
            'pred_label': bool_str(pred_label),
            'pred_prob': float(prob),
            'is_correct': bool_str(int(true_label) == int(pred_label)),
            'has_taint': bool_str(1 if meta.get('has_taint', False) else 0),
            'taint_path_count': int(meta.get('taint_path_count', 0)),
        })

    pred_path = os.path.join(output_dir, 'test_predictions_taint_fusion_bigru.csv')
    pd.DataFrame(rows).to_csv(pred_path, index=False, encoding='utf-8-sig')

    return pred_path


def _normalize_split_value(x) -> str:
    return str(x).strip().lower()


def _select_split_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ['split', 'subset', 'set', 'stage', 'data_split']

    for col in candidates:
        if col in df.columns:
            return col

    return None


def _select_key_column(df: pd.DataFrame, preferred: Sequence[str]) -> Optional[str]:
    for col in preferred:
        if col in df.columns:
            return col

    return None


def _example_attr(ex, name: str):
    return getattr(ex, name, None)


def load_test_examples(args):
    """
    优先根据训练阶段保存的 skill_split.csv 读取测试集；
    失败时用相同随机种子重新划分。
    """
    examples = load_skill_examples(
        csv_path=args.data_csv,
        drop_blank=args.drop_blank,
        drop_comment=args.drop_comment,
        min_lines=1,
    )

    split_csv = args.split_csv or os.path.join(args.output_dir, 'skill_split.csv')

    if os.path.exists(split_csv):
        df = pd.read_csv(split_csv)
        split_col = _select_split_column(df)

        if split_col is not None:
            test_df = df[df[split_col].map(_normalize_split_value).isin({'test', '测试'})].copy()

            if len(test_df) > 0:
                key_col = _select_key_column(test_df, ['skill_name', 'url', 'id', 'repo', 'path'])

                if key_col is not None:
                    split_keys = set(test_df[key_col].astype(str).tolist())
                    selected = []

                    for ex in examples:
                        ex_key = _example_attr(ex, key_col)
                        if ex_key is not None and str(ex_key) in split_keys:
                            selected.append(ex)

                    if selected:
                        print(f'Loaded test examples from split CSV: {split_csv}')
                        return selected, examples, split_csv

                print('[WARN] skill_split.csv 存在，但无法根据 key 列匹配样本，将使用相同 seed 重新划分。')
            else:
                print('[WARN] skill_split.csv 中未找到 split == test 的样本，将使用相同 seed 重新划分。')
        else:
            print('[WARN] skill_split.csv 中没有 split/subset/set/stage 列，将使用相同 seed 重新划分。')

    _, _, test_examples = stratified_split_examples(
        examples,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
        args.seed,
    )

    print('[WARN] Using fallback stratified split to construct test examples.')
    return test_examples, examples, split_csv


def inspect_checkpoint_state(checkpoint: dict):
    """
    检查 checkpoint 是否为 BiGRU 污点路径版模型。
    """
    state = checkpoint.get('model_state_dict', {})

    has_taint_gru = any(k.startswith('taint_gru') for k in state.keys())
    has_taint_norm = any(k.startswith('taint_norm') for k in state.keys())

    fusion_shapes = {
        k: tuple(v.shape)
        for k, v in state.items()
        if 'fusion_fc' in k and hasattr(v, 'shape')
    }

    return {
        'has_taint_gru': has_taint_gru,
        'has_taint_norm': has_taint_norm,
        'fusion_shapes': fusion_shapes,
    }


def assert_bigru_checkpoint(checkpoint: dict, checkpoint_path: str):
    """
    如果误加载旧版 mean-pooling 融合模型 checkpoint，直接给出明确错误。
    """
    info = inspect_checkpoint_state(checkpoint)

    print('Checkpoint structure check:', info)

    if not info['has_taint_gru']:
        raise RuntimeError(
            '\n当前 checkpoint 不是 BiGRU 污点路径版模型。\n'
            '原因：checkpoint 中没有 taint_gru.* 参数，但当前 BAFLineTaintFusion 需要 taint_gru。\n\n'
            f'当前加载的 checkpoint: {checkpoint_path}\n\n'
            '请检查是否加载错目录。BiGRU 训练脚本默认保存路径通常是：\n'
            '/root/BATaint/output/bafline_taint_fusion_bigru/best_model.pt\n\n'
            '建议运行：\n'
            'python test_taint_fusion_bafline_bigru.py \\\n'
            '  --output_dir /root/BATaint/output/bafline_taint_fusion_bigru \\\n'
            '  --checkpoint_path /root/BATaint/output/bafline_taint_fusion_bigru/best_model.pt\n\n'
            '如果该文件不存在，请先重新训练 BiGRU 版本模型。\n'
        )


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = str(device)

    if not args.checkpoint_path:
        args.checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')

    if not args.test_output_dir:
        args.test_output_dir = args.output_dir

    os.makedirs(args.test_output_dir, exist_ok=True)

    print('===== Load Checkpoint =====')
    print('Checkpoint:', args.checkpoint_path)

    checkpoint = safe_torch_load(args.checkpoint_path, device)

    # 先检查 checkpoint 是否真的是 BiGRU 版，避免出现 state_dict 维度错误
    assert_bigru_checkpoint(checkpoint, args.checkpoint_path)

    checkpoint_args = checkpoint.get('args', {})
    args = merge_checkpoint_args(args, checkpoint_args)

    if not getattr(args, 'test_output_dir', ''):
        args.test_output_dir = args.output_dir

    os.makedirs(args.test_output_dir, exist_ok=True)

    print('Test output dir:', args.test_output_dir)
    print('Device:', device)

    print('===== Load Test Dataset =====')
    test_examples, all_examples, split_csv = load_test_examples(args)

    print('All :', summarize_examples(all_examples))
    print('Test:', summarize_examples(test_examples))

    print('===== Load Encoder =====')
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name_or_path)
    encoder = AutoModel.from_pretrained(args.encoder_name_or_path).to(device)
    encoder.eval()

    if checkpoint.get('encoder_state_dict') is not None:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])

    for p in encoder.parameters():
        p.requires_grad = False

    hidden_size = getattr(encoder.config, 'hidden_size', 768)
    pad_token_id = get_pad_token_id(tokenizer)

    print('===== Build Model =====')
    model = BAFLineTaintFusion(
        code_embed_dim=hidden_size,
        taint_embed_dim=hidden_size,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_num_layers,
        bafn_output_dim=args.bafn_hidden_dim,
        dropout=args.dropout,
        device=device,
        norm_type=args.norm_type,
        taint_gru_hidden_dim=args.taint_gru_hidden_dim,
        taint_gru_num_layers=args.taint_gru_num_layers,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # 优先使用训练阶段在验证集上搜索得到的最佳阈值；没有则回退到命令行 --threshold。
    test_threshold = float(checkpoint.get('best_threshold', args.threshold))
    print(f'Test threshold: {test_threshold:.6f}')

    test_dl = build_dataloader(test_examples, tokenizer, args, shuffle=False)
    # 模型输出 raw logits，测试损失也必须使用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    print('===== Test =====')
    test_metrics, y_true, y_pred, y_prob, metas = evaluate(
        model=model,
        encoder=encoder,
        dataloader=test_dl,
        criterion=criterion,
        pad_token_id=pad_token_id,
        device=device,
        hidden_size=hidden_size,
        threshold=test_threshold,
        args=args,
        desc='Test',
    )

    pred_path = save_predictions(args.test_output_dir, y_true, y_pred, y_prob, metas)

    test_summary = {
        'checkpoint_path': args.checkpoint_path,
        'checkpoint_epoch': checkpoint.get('epoch'),
        'checkpoint_valid_metrics': checkpoint.get('valid_metrics'),
        'checkpoint_best_threshold': checkpoint.get('best_threshold'),
        'test_threshold': test_threshold,
        'test_metrics': test_metrics,
        'data_summary': {
            'all': summarize_examples(all_examples),
            'test': summarize_examples(test_examples),
        },
        'split_csv': split_csv,
        'test_predictions_csv': pred_path,
        'args': vars(args),
    }

    summary_path = os.path.join(args.test_output_dir, 'test_summary_taint_fusion_bigru.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(test_summary, f, ensure_ascii=False, indent=2)

    print('===== Test Done =====')
    print('Test metrics:', test_metrics)
    print('Predictions:', pred_path)
    print('Summary:', summary_path)


if __name__ == '__main__':
    main()
