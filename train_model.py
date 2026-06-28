# -*- coding: utf-8 -*-


import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# 如果你直接替换原 Skills_attention.py，请使用：
# from BAFLineDP import BAFLineTaintFusion
# 如果你保留本回答生成的新文件名，请使用下面这一行：
from SkillsPF import SkillsPF_model

from my_util import (
    SkillCodeTaintDataset,
    bool_str,
    collate_skill_taint_batch,
    load_skill_examples,
    save_split_summary,
    set_seed,
    stratified_split_examples,
    summarize_examples,
)
from taint_feature_extract import build_taint_text_summary


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_csv', type=str, default='/root/BATaint/datasets/skills_ground_truth_lines_dataset.csv')
    parser.add_argument('--repo_root', type=str, default='/root/BATaint/datasets/repo')
    parser.add_argument('--encoder_name_or_path', type=str, default='/root/BATaint/unixcoder-base-nine')
    parser.add_argument('--output_dir', type=str, default='/root/BATaint/output/bafline_taint_fusion_bigru')

    parser.add_argument('--train_ratio', type=float, default=0.6)
    parser.add_argument('--valid_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--encoder_lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--max_grad_norm', type=float, default=5.0)

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

    # =========================
    # 【修改位置 1】新增污点路径 BiGRU 超参数
    # =========================
    parser.add_argument('--taint_gru_hidden_dim', type=int, default=32)
    parser.add_argument('--taint_gru_num_layers', type=int, default=1)

    parser.add_argument('--drop_blank', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--drop_comment', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fine_tune_encoder', action='store_true', default=False)
    parser.add_argument('--threshold', type=float, default=0.5)

    # =========================
    # 【新增】解决类别不平衡下固定 0.5 阈值导致 recall=0 的问题
    # threshold_metric: 在验证集上搜索最佳阈值时优化的指标
    # selection_metric: 保存 best_model.pt 时使用的验证集指标
    # =========================
    parser.add_argument('--threshold_metric', type=str, default='gmean', choices=['f1', 'gmean', 'mcc', 'recall'])
    parser.add_argument('--selection_metric', type=str, default='auc', choices=['f1', 'gmean', 'mcc', 'auc', 'recall'])
    parser.add_argument('--min_threshold', type=float, default=0.05)
    parser.add_argument('--max_threshold', type=float, default=0.95)
    parser.add_argument('--threshold_steps', type=int, default=181)

    return parser.parse_args()


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
    freeze_encoder: bool = True,
):
    """源码路径：每个 Skill 的多行代码分别由 UniXCoder 编码，返回 List[[line_num, hidden]]。"""
    embeddings = []
    context = torch.no_grad() if freeze_encoder else torch.enable_grad()
    with context:
        for input_ids in code_input_ids_list:
            embeddings.append(encode_one_matrix(encoder, input_ids, pad_token_id, device, line_batch_size))
    return embeddings


# =========================
# 【修改位置 2】替换原 encode_batch_taint_embeddings
# 原函数会 path_embeds.mean(dim=0)，现在改为保留路径向量序列。
# =========================
def encode_batch_taint_path_embeddings(
    encoder,
    taint_input_ids_list: List[torch.Tensor],
    has_taint_list: List[bool],
    pad_token_id: int,
    device,
    taint_batch_size: int,
    hidden_size: int,
    freeze_encoder: bool = True,
) -> List[torch.Tensor]:
    """
    污点路径路径：
    - 有污点路径：返回 Tensor[path_num, hidden_size]
    - 无污点路径：返回空 Tensor[0, hidden_size]

    注意：这里不做 mean pooling，多条路径之间的上下文关系交给模型中的 taint_gru 建模。
    """
    taint_path_embeds: List[torch.Tensor] = []
    context = torch.no_grad() if freeze_encoder else torch.enable_grad()
    with context:
        for input_ids, has_taint in zip(taint_input_ids_list, has_taint_list):
            if not has_taint:
                taint_path_embeds.append(torch.empty(0, hidden_size, dtype=torch.float, device=device))
                continue
            path_embeds = encode_one_matrix(encoder, input_ids, pad_token_id, device, taint_batch_size)
            taint_path_embeds.append(path_embeds)
    return taint_path_embeds


def build_dataloader(examples, tokenizer, args, shuffle: bool):
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


def compute_pos_weight(train_examples) -> torch.Tensor:
    labels = np.array([ex.label for ex in train_examples])
    pos = float(np.sum(labels == 1))
    neg = float(np.sum(labels == 0))
    if pos == 0:
        return torch.tensor([1.0], dtype=torch.float)
    return torch.tensor([neg / pos], dtype=torch.float)

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from math import sqrt

def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    if not y_true:
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'gmean': 0.0,
            'mcc': 0.0,
            'auc': None,
            'tn': 0,
            'fp': 0,
            'fn': 0,
            'tp': 0,
        }

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, zero_division=0)

    a = TP + FP
    b = TP + FN
    c = TN + FP
    d = TN + FN
    if a * b * c * d != 0:
        mcc = (TP * TN - FP * FN) / sqrt(a * b * c * d)
    else:
        mcc = 0.0

    tpr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    tnr = TN / (FP + TN) if (FP + TN) > 0 else 0.0
    gmean = sqrt(tpr * tnr)

    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) >= 2 else None

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'gmean': float(gmean),
        'mcc': float(mcc),
        'auc': None if auc is None else float(auc),
        'tn': int(TN),
        'fp': int(FP),
        'fn': int(FN),
        'tp': int(TP),
    }


def search_best_threshold(
    y_true,
    y_prob,
    metric_name: str = 'f1',
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    steps: int = 181,
):
    """
    在验证集上搜索最佳分类阈值。

    作用：
    - 解决类别不平衡时固定 threshold=0.5 导致模型全部预测为 FALSE 的问题；
    - 对缺陷检测任务，建议优先使用 f1 或 gmean。
    """
    if not y_true or not y_prob:
        return 0.5, compute_metrics([], [], [])

    thresholds = np.linspace(min_threshold, max_threshold, max(2, int(steps)))
    best_threshold = 0.5
    best_score = -1.0
    best_metrics = None

    for th in thresholds:
        preds = [1 if p >= th else 0 for p in y_prob]
        cur_metrics = compute_metrics(y_true, preds, y_prob)
        score = cur_metrics.get(metric_name)

        if score is None:
            score = -1.0

        # 同分时选择更低阈值，提高缺陷召回率
        if score > best_score or (score == best_score and th < best_threshold):
            best_score = float(score)
            best_threshold = float(th)
            best_metrics = cur_metrics

    if best_metrics is None:
        preds = [1 if p >= 0.5 else 0 for p in y_prob]
        best_metrics = compute_metrics(y_true, preds, y_prob)
        best_threshold = 0.5

    best_metrics['threshold'] = float(best_threshold)
    best_metrics['threshold_metric'] = metric_name
    return best_threshold, best_metrics

def evaluate(model, encoder, dataloader, criterion, pad_token_id: int, device, hidden_size: int, threshold: float, args, desc: str = 'Eval'):
    model.eval()
    encoder.eval()
    total_loss = []
    all_probs, all_labels, all_metas = [], [], []

    with torch.no_grad():
        for code_ids_list, taint_ids_list, has_taint_list, labels, metas in tqdm(dataloader, desc=desc):
            labels = labels.to(device).view(-1, 1)
            code_embeds = encode_batch_code_embeddings(
                encoder,
                code_ids_list,
                pad_token_id,
                device,
                args.encoder_line_batch_size,
                freeze_encoder=True,
            )
            # =========================
            # 【修改位置 3】评估阶段也返回 List[[path_num, hidden]]，交给模型内部 taint_gru
            # =========================
            taint_path_embeds = encode_batch_taint_path_embeddings(
                encoder,
                taint_ids_list,
                has_taint_list,
                pad_token_id,
                device,
                args.encoder_taint_batch_size,
                hidden_size,
                freeze_encoder=True,
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
    metrics = compute_metrics(all_labels, pred_labels, all_probs)
    metrics['loss'] = float(np.mean(total_loss)) if total_loss else 0.0
    return metrics, all_labels, pred_labels, all_probs, all_metas


def save_predictions(output_dir, y_true, y_pred, y_prob, metas):
    rows = []
    for true_label, pred_label, prob, meta in zip(y_true, y_pred, y_prob, metas):
        rows.append({
            'url': meta['url'],
            'skill_name': meta['skill_name'],
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


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.device = str(device)

    print('===== Load Skill Dataset =====')
    examples = load_skill_examples(
        csv_path=args.data_csv,
        drop_blank=args.drop_blank,
        drop_comment=args.drop_comment,
        min_lines=1,
    )
    print('Total:', summarize_examples(examples))

    taint_summary = build_taint_text_summary(
        repo_root=args.repo_root,
        skill_names=[ex.skill_name for ex in examples],
        taint_result_name=args.taint_result_name,
    )
    taint_summary_path = os.path.join(args.output_dir, 'taint_text_summary.json')
    with open(taint_summary_path, 'w', encoding='utf-8') as f:
        json.dump(taint_summary, f, ensure_ascii=False, indent=2)
    print('Taint summary:', {k: taint_summary[k] for k in ['total_skill_count', 'skills_with_taint_paths', 'skills_without_taint_paths']})

    train_examples, valid_examples, test_examples = stratified_split_examples(
        examples,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
        args.seed,
    )
    save_split_summary(args.output_dir, train_examples, valid_examples, test_examples)
    print('Train:', summarize_examples(train_examples))
    print('Valid:', summarize_examples(valid_examples))
    print('Test :', summarize_examples(test_examples))

    print('===== Load Encoder =====')
    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name_or_path)
    encoder = AutoModel.from_pretrained(args.encoder_name_or_path).to(device)

    freeze_encoder = not args.fine_tune_encoder
    if freeze_encoder:
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    hidden_size = getattr(encoder.config, 'hidden_size', 768)
    pad_token_id = get_pad_token_id(tokenizer)

    model = BAFLineTaintFusion(
        code_embed_dim=hidden_size,
        taint_embed_dim=hidden_size,
        gru_hidden_dim=args.gru_hidden_dim,
        gru_num_layers=args.gru_num_layers,
        bafn_output_dim=args.bafn_hidden_dim,
        dropout=args.dropout,
        device=device,
        norm_type=args.norm_type,
        # =========================
        # 【修改位置 4】传入污点路径 BiGRU 参数
        # =========================
        taint_gru_hidden_dim=args.taint_gru_hidden_dim,
        taint_gru_num_layers=args.taint_gru_num_layers,
    ).to(device)

    train_dl = build_dataloader(train_examples, tokenizer, args, shuffle=True)
    valid_dl = build_dataloader(valid_examples, tokenizer, args, shuffle=False)
    test_dl = build_dataloader(test_examples, tokenizer, args, shuffle=False)

    if args.fine_tune_encoder:
        optimizer = optim.AdamW([
            {'params': model.parameters(), 'lr': args.lr},
            {'params': encoder.parameters(), 'lr': args.encoder_lr},
        ], weight_decay=args.weight_decay)
        trainable_params = list(model.parameters()) + list(encoder.parameters())
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        trainable_params = list(model.parameters())

    # =========================
    # 【关键修正】模型输出的是 raw logits，不是 sigmoid 后的概率。
    # 因此必须使用 BCEWithLogitsLoss；同时使用 pos_weight 处理类别不平衡。
    # 原来使用 BCELoss 会导致训练目标不匹配，容易出现全部预测为负类。
    # =========================
    pos_weight = compute_pos_weight(train_examples).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f'pos_weight = {pos_weight.detach().cpu().item():.4f}')

    best_score = -1.0
    best_epoch = 0
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    history = []

    print('===== Train =====')
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        encoder.train() if args.fine_tune_encoder else encoder.eval()
        train_losses = []

        for code_ids_list, taint_ids_list, has_taint_list, labels, metas in tqdm(train_dl, desc=f'Train Epoch {epoch}'):
            labels = labels.to(device).view(-1, 1)
            code_embeds = encode_batch_code_embeddings(
                encoder,
                code_ids_list,
                pad_token_id,
                device,
                args.encoder_line_batch_size,
                freeze_encoder=freeze_encoder,
            )
            # =========================
            # 【修改位置 5】训练阶段保留多条污点路径向量序列，不做均值池化
            # =========================
            taint_path_embeds = encode_batch_taint_path_embeddings(
                encoder,
                taint_ids_list,
                has_taint_list,
                pad_token_id,
                device,
                args.encoder_taint_batch_size,
                hidden_size,
                freeze_encoder=freeze_encoder,
            )
            logits, _ = model(code_embeds, taint_path_embeds)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            train_losses.append(loss.item())

        valid_metrics_raw, valid_y_true, _, valid_y_prob, _ = evaluate(
            model,
            encoder,
            valid_dl,
            criterion,
            pad_token_id,
            device,
            hidden_size,
            args.threshold,
            args,
            desc=f'Valid Epoch {epoch}',
        )

        # =========================
        # 【新增】在验证集上自动搜索最佳阈值，再用该阈值计算 F1/G-mean/MCC/Recall。
        # 注意：AUC 是排序指标，不能解决固定 0.5 阈值导致 recall=0 的问题。
        # =========================
        best_threshold, valid_metrics = search_best_threshold(
            y_true=valid_y_true,
            y_prob=valid_y_prob,
            metric_name=args.threshold_metric,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            steps=args.threshold_steps,
        )
        valid_metrics['loss'] = valid_metrics_raw.get('loss', 0.0)
        valid_metrics['auc'] = valid_metrics_raw.get('auc', valid_metrics.get('auc'))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0

        valid_score = valid_metrics.get(args.selection_metric)
        if valid_score is None:
            valid_score = valid_metrics.get('f1', 0.0)

        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'best_threshold': best_threshold,
            **{f'valid_{k}': v for k, v in valid_metrics.items()}
        })

        print(
            f"Epoch {epoch} | train_loss={train_loss:.4f} | "
            f"valid_loss={valid_metrics['loss']:.4f} | "
            f"valid_precision={valid_metrics['precision']:.4f} | "
            f"valid_recall={valid_metrics['recall']:.4f} | "
            f"valid_f1={valid_metrics['f1']:.4f} | "
            f"valid_gmean={valid_metrics['gmean']:.4f} | "
            f"valid_mcc={valid_metrics['mcc']:.4f} | "
            f"valid_auc={valid_metrics['auc']} | "
            f"best_threshold={best_threshold:.4f}"
        )

        if valid_score >= best_score:
            best_score = valid_score
            best_epoch = epoch
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': encoder.state_dict() if args.fine_tune_encoder else None,
                'args': vars(args),
                'valid_metrics': valid_metrics,
                'best_threshold': float(best_threshold),
                'threshold_metric': args.threshold_metric,
                'selection_metric': args.selection_metric,
            }, best_model_path)

    pd.DataFrame(history).to_csv(
        os.path.join(args.output_dir, 'training_history_taint_fusion_bigru.csv'),
        index=False,
        encoding='utf-8-sig',
    )

    # 当前版本仍保持你之前的设置：只训练和验证，不执行测试。
    # 如需恢复测试，取消下面代码注释即可。
    # print('===== Test =====')
    # checkpoint = safe_torch_load(best_model_path, device)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # if args.fine_tune_encoder and checkpoint.get('encoder_state_dict') is not None:
    #     encoder.load_state_dict(checkpoint['encoder_state_dict'])
    # test_metrics, y_true, y_pred, y_prob, metas = evaluate(
    #     model, encoder, test_dl, criterion, pad_token_id, device, hidden_size, args.threshold, args, desc='Test'
    # )
    # pred_path = save_predictions(args.output_dir, y_true, y_pred, y_prob, metas)
    # print('Test metrics:', test_metrics)
    # print('Predictions:', pred_path)

    print('===== Done =====')
    print('Best epoch:', best_epoch)
    print('Best valid score:', best_score)
    print('Best model:', best_model_path)


if __name__ == '__main__':
    main()
