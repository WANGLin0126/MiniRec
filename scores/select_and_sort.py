#!/usr/bin/env python3
import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# ======================================================
# 1. 数据加载
# ======================================================
def load_embeddings(path, device="cuda"):
    embs = torch.load(path, map_location=device)  # [N, D]
    if isinstance(embs, (list, tuple)):  # 保守判断
        embs = embs[0]
    return embs.to(device)

def load_scores(path):
    """从 jsonl 加载 difficulty 和 representativeness"""
    diff_dict, rep_dict = {}, {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sid = int(rec["sample_id"])
            diff_dict[sid] = rec.get("difficulty", 0.0)
            rep_dict[sid] = rec.get("representativeness", 0.0)
    return diff_dict, rep_dict

# ======================================================
# 2. MaxMin 选择
# ======================================================
def maxmin_select(embs, scores_dict, select_k, init_id=None):
    device = embs.device
    N, D = embs.shape
    ids = torch.arange(N, device=device)

    # 归一化向量
    embs_norm = embs / (embs.norm(dim=1, keepdim=True) + 1e-12)

    # 初始点：如果没指定，就选 scores_dict 最大的
    if init_id is None:
        init_id = max(scores_dict.keys(), key=lambda x: scores_dict[x])
    selected_ids = [init_id]
    selected_vs = [1.0]

    # 初始化距离
    sim_init = (embs_norm @ embs_norm[init_id].unsqueeze(1)).squeeze(1)  # [N]
    dists = 1 - sim_init

    # 分数张量
    scores_arr = torch.tensor(
        [scores_dict.get(int(i.item()), 0.0) for i in ids],
        dtype=torch.float32,
        device=device,
    )

    for _ in tqdm(range(1, select_k), desc="maxmin selecting"):
        min_val, max_val = dists.min(), dists.max()
        v = (dists - min_val) / (max_val - min_val + 1e-12)

        final_scores = v * scores_arr
        final_scores[torch.tensor(selected_ids, device=device)] = -float("inf")

        new_id = int(torch.argmax(final_scores).item())
        selected_ids.append(new_id)
        selected_vs.append(float(v[new_id].item()))

        # 更新 dists
        sim_new = (embs_norm @ embs_norm[new_id].unsqueeze(1)).squeeze(1)
        dist_new = 1 - sim_new
        dists = torch.minimum(dists, dist_new)

    return selected_ids, selected_vs

# ======================================================
# 3. Fold 排序 & 可视化
# ======================================================
def random_fold_sort_with_visual(
    selected_ids,
    jsonl_path: str,
    output_path: str,
    n_folds: int = 5,
    plot_path: str = "reward_difficulty.png",
    seed: int = 42,
):
    selected_ids = set(selected_ids)

    # 读取 jsonl，计算平均 reward
    sample_rewards = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            data = json.loads(line)
            sid = int(data["sample_id"])
            if sid in selected_ids:
                avg_reward = np.mean(data["reward"])
                sample_rewards[sid] = avg_reward
    print(f"匹配到 {len(sample_rewards)} 个样本")

    # 随机划分 fold
    all_ids = list(sample_rewards.keys())
    random.seed(seed)
    random.shuffle(all_ids)
    N = len(all_ids)
    if N < n_folds:
        raise ValueError(f"样本数 {N} 小于 fold 数 {n_folds}")
    fold_size = N // n_folds
    folds = []

    # 每个 fold 内部排序（reward 降序）
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else N
        fold_ids = all_ids[start:end]
        fold_rewards = [sample_rewards[sid] for sid in fold_ids]

        sorted_fold = sorted(zip(fold_ids, fold_rewards), key=lambda x: x[1], reverse=True)
        fold_ids_sorted = [sid for sid, _ in sorted_fold]
        fold_rewards_sorted = [r for _, r in sorted_fold]
        folds.append((fold_ids_sorted, fold_rewards_sorted))

    # 拼接所有 fold
    final_sorted_ids, final_rewards = [], []
    for (fold_ids_sorted, fold_rewards_sorted) in folds:
        final_sorted_ids.extend(fold_ids_sorted)
        final_rewards.extend(fold_rewards_sorted)

    # 保存 ids
    output_path = output_path.replace(".npy", f"_fold_{str(n_folds)}.npy")
    np.save(output_path, np.array(final_sorted_ids, dtype=int))
    print(f"排序结果已保存到 {output_path}")

    # 作图
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10.colors
    offset, seq_rewards = 0, []
    for i, (_, fold_rewards_sorted) in enumerate(folds):
        x_range = range(offset, offset + len(fold_rewards_sorted))
        plt.plot(x_range, fold_rewards_sorted, color=colors[i % len(colors)], label=f"Fold {i+1}")
        plt.scatter(x_range, fold_rewards_sorted, s=10, color=colors[i % len(colors)])
        seq_rewards.extend(fold_rewards_sorted)
        offset += len(fold_rewards_sorted)

    plt.title(f"K-Fold Curriculum (random fold + intra-fold sort, n_folds={n_folds})")
    plt.xlabel("Sample Rank (after fold concat)")
    plt.ylabel("Average Reward")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    print(f"可视化结果已保存到 {plot_path}")

    # 一致性检查
    assert np.allclose(seq_rewards, final_rewards, atol=1e-8)

    return final_sorted_ids, folds

# ======================================================
# 4. 主函数
# ======================================================
def main():
    # 参数
    select_k = 256
    alpha = 1
    n_folds = 4

    embs_file = "/storage_fast/lwang/SeqRecDistill/RRec/embs/selected/all_unique_embs.pt"
    scores_file = "/storage_fast/lwang/SeqRecDistill/RRec/scores/sample_scores.jsonl"
    train_jsonl = "/storage_fast/lwang/SeqRecDistill/RRec/train_diff/all_merged_sorted.jsonl"

    save_pt = f"/storage_fast/lwang/SeqRecDistill/RRec/scores/selected_indices_{select_k}.npy"
    save_jsonl = "/storage_fast/lwang/SeqRecDistill/RRec/scores/selected_samples.jsonl"
    sorted_output = f"/storage_fast/lwang/SeqRecDistill/RRec/scores/sorted_{str(select_k)}.npy"
    plot_path = "curriculum_seq_random.png"

    os.makedirs(os.path.dirname(save_pt), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载数据
    embs = load_embeddings(embs_file, device=device)
    diff_dict, rep_dict = load_scores(scores_file)
    scores_dict = {k: alpha * diff_dict[k] + rep_dict[k] for k in diff_dict.keys()}

    # 2. MaxMin 选择
    selected_ids, selected_vs = maxmin_select(embs, scores_dict, select_k=select_k)

    # 保存挑选结果
    np.save(save_pt, selected_ids)
    print(f"✅ 已保存选择结果到 {save_pt}")

    with open(save_jsonl, "w", encoding="utf-8") as f:
        for sid, v in zip(selected_ids, selected_vs):
            rec = {
                "sample_id": sid,
                "v": v,
                "difficulty": diff_dict.get(sid, 0.0),
                "representativeness": rep_dict.get(sid, 0.0),
                "total": diff_dict.get(sid, 0.0) + rep_dict.get(sid, 0.0),
            }
            f.write(json.dumps(rec) + "\n")
    print(f"✅ 已保存详细结果到 {save_jsonl}")

    # 3. 排序 + 可视化
    sorted_ids, folds = random_fold_sort_with_visual(
        selected_ids,
        jsonl_path=train_jsonl,
        output_path=sorted_output,
        n_folds=n_folds,
        plot_path=plot_path,
        seed=42,
    )

    # 打印前几个样本
    print("\n示例选出的前 10 个样本:")
    for sid, v in zip(selected_ids[:10], selected_vs[:10]):
        d = diff_dict.get(sid, 0.0)
        r = rep_dict.get(sid, 0.0)
        print(f"id={sid}, v={v:.4f}, difficulty={d:.4f}, represent={r:.4f}, total={d+r:.4f}")

if __name__ == "__main__":
    main()