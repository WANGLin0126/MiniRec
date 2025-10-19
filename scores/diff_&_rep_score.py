#!/usr/bin/env python3
import json
import numpy as np
import torch
import os
from collections import defaultdict

# ======================================================
# 1. Gaussian difficulty score
# ======================================================
def gaussian_scores(reward_pairs, mu=0.5, sigma=None):
    rewards = np.array([r for _, r in reward_pairs])
    
    if mu is None:
        mu = rewards.mean()
    if sigma is None or sigma == 0:
        sigma = rewards.std()
        if sigma == 0:
            sigma = 1e-6

    scores_raw = np.exp(- (rewards - mu) ** 2 / (2 * sigma ** 2))

    # min-max 归一化
    min_val, max_val = scores_raw.min(), scores_raw.max()
    if max_val == min_val:
        scores_norm = np.ones_like(scores_raw)
    else:
        scores_norm = (scores_raw - min_val) / (max_val - min_val)

    return [(sid, float(r), float(sc)) 
            for (sid, r), sc in zip(reward_pairs, scores_norm)]


def compute_difficulty_scores(jsonl_path):
    reward_dict = defaultdict(list)

    # 从 JSONL 中收集 rewards
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = int(rec["sample_id"])
            rewards = rec.get("reward", [])
            if rewards:
                mean_reward = sum(rewards) / len(rewards)
                reward_dict[sid].append(mean_reward)

    # 合并相同 sample_id 的 reward
    merged = []
    for sid, vals in reward_dict.items():
        merged_mean = sum(vals) / len(vals)
        merged.append((sid, merged_mean))

    # 计算 Gaussian 分数
    scores = gaussian_scores(merged, mu=0.5)

    # 转 dict {sid: score}
    scores_dict = {sid: score for sid, _, score in scores}
    return scores_dict


# ======================================================
# 2. Cosine representativeness score
# ======================================================
def compute_cosine_scores(pt_path):
    grads_1st, grads_2nd, ids = torch.load(pt_path)

    mean_vec = grads_2nd.mean(dim=0)
    mean_vec_norm = mean_vec / (mean_vec.norm() + 1e-12)

    grads_norm = grads_2nd / (grads_2nd.norm(dim=1, keepdim=True) + 1e-12)
    cos_sims = (grads_norm @ mean_vec_norm.unsqueeze(1)).squeeze()

    scores = (cos_sims + 1) / 2

    min_val, max_val = scores.min(), scores.max()
    if max_val == min_val:
        scores_norm = torch.ones_like(scores)
    else:
        scores_norm = (scores - min_val) / (max_val - min_val)

    # 转成 dict {sid: score}
    scores_dict = {int(i.item()): float(s) for i, s in zip(ids, scores_norm)}
    return scores_dict


# ======================================================
# 3. 整合并保存
# ======================================================
def main():
    reward_file = "/storage_fast/lwang/SeqRecDistill/RRec/train_diff/all_merged_sorted.jsonl"
    grads_file = "/storage_fast/lwang/SeqRecDistill/RRec/grads/grads_all_sorted_dedup.pt"
    output_file = "/storage_fast/lwang/SeqRecDistill/RRec/scores/sample_scores.jsonl"

    # 确保目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(">>> 计算 difficulty scores (Gaussian)...")
    diff_scores = compute_difficulty_scores(reward_file)

    print(">>> 计算 representativeness scores (Cosine)...")
    rep_scores = compute_cosine_scores(grads_file)

    # 合并结果
    all_ids = set(diff_scores.keys()) | set(rep_scores.keys())
    merged = []
    for sid in sorted(all_ids):
        merged.append({
            "sample_id": sid,
            "difficulty": diff_scores.get(sid, None),
            "representativeness": rep_scores.get(sid, None)
        })

    # 保存为 JSONL
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec) + "\n")

    print(f"✅ 已保存综合分数到 {output_file}")
    print("前 5 个样本:")
    for rec in merged[:5]:
        print(rec)


if __name__ == "__main__":
    main()