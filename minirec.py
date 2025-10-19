#!/usr/bin/env python3
import os
import json
import random
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl


# ======================================================
# 1. Gaussian difficulty score
# ======================================================
def gaussian_scores(reward_pairs, mu=0.5, sigma=None):
    rewards = np.array([r for _, r in reward_pairs])
    mu = rewards.mean() if mu is None else mu
    sigma = rewards.std() if (sigma is None or sigma == 0) else sigma
    sigma = sigma if sigma > 0 else 1e-6

    scores_raw = np.exp(- (rewards - mu) ** 2 / (2 * sigma ** 2))
    min_val, max_val = scores_raw.min(), scores_raw.max()
    scores_norm = (scores_raw - min_val) / (max_val - min_val + 1e-12) if max_val > min_val else np.ones_like(scores_raw)
    return {sid: float(sc) for (sid, _), sc in zip(reward_pairs, scores_norm)}


def compute_difficulty_scores(jsonl_path):
    reward_dict = defaultdict(list)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = int(rec["sample_id"])
            rewards = rec.get("reward", [])
            if rewards:
                reward_dict[sid].append(sum(rewards) / len(rewards))

    merged = [(sid, sum(vals) / len(vals)) for sid, vals in reward_dict.items()]
    return gaussian_scores(merged, mu=0.5)


# ======================================================
# 2. Cosine representativeness score
# ======================================================
def compute_cosine_scores(pt_path):
    _, grads, ids = torch.load(pt_path)
    mean_vec = grads.mean(dim=0)
    mean_vec_norm = mean_vec / (mean_vec.norm() + 1e-12)

    grads_norm = grads / (grads.norm(dim=1, keepdim=True) + 1e-12)
    cos_sims = (grads_norm @ mean_vec_norm.unsqueeze(1)).squeeze()
    scores = (cos_sims + 1) / 2

    min_val, max_val = scores.min(), scores.max()
    scores_norm = (scores - min_val) / (max_val - min_val + 1e-12) if max_val > min_val else torch.ones_like(scores)
    return {int(i.item()): float(s) for i, s in zip(ids, scores_norm)}


# ======================================================
# 3. Embeddings + Save JSONL
# ======================================================
def load_embeddings(path, device="cuda"):
    embs = torch.load(path, map_location=device)
    if isinstance(embs, (list, tuple)):
        embs = embs[0]
    return embs.to(device)


def save_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ======================================================
# 4. MaxMin selection
# ======================================================
def maxmin_select(embs, scores_dict, k, init_id=None):
    device = embs.device
    N = embs.shape[0]
    ids = torch.arange(N, device=device)
    embs_norm = embs / (embs.norm(dim=1, keepdim=True) + 1e-12)

    if init_id is None:
        init_id = max(scores_dict, key=scores_dict.get)
    selected, vs = [init_id], [1.0]

    dists = 1 - (embs_norm @ embs_norm[init_id].unsqueeze(1)).squeeze(1)
    scores_arr = torch.tensor([scores_dict.get(int(i), 0.0) for i in ids],
                              dtype=torch.float32, device=device)

    for _ in tqdm(range(1, k), desc="maxmin selecting"):
        v = (dists - dists.min()) / (dists.max() - dists.min() + 1e-12)
        final_scores = v * scores_arr
        final_scores[torch.tensor(selected, device=device)] = -1e9

        new_id = int(torch.argmax(final_scores))
        selected.append(new_id)
        vs.append(float(v[new_id]))
        dists = torch.minimum(dists, 1 - (embs_norm @ embs_norm[new_id].unsqueeze(1)).squeeze(1))

    return selected, vs


# ======================================================
# 5. Fold sorting & visualization
# ======================================================
def random_fold_sort_with_visual(selected_ids, reward_file, output_path, n_folds=5, plot_path="curriculum.pdf", seed=42):
    selected_ids = set(selected_ids)

    sample_rewards = {}
    with open(reward_file, "r") as f:
        for line in f:
            data = json.loads(line)
            sid = int(data["sample_id"])
            if sid in selected_ids:
                sample_rewards[sid] = np.mean(data["reward"])

    all_ids = list(sample_rewards.keys())
    random.seed(seed)
    random.shuffle(all_ids)
    N = len(all_ids)
    if N < n_folds:
        raise ValueError(f"Number of samples {N} is smaller than number of folds {n_folds}")
    fold_size = N // n_folds

    folds, sorted_ids = [], []
    for i in range(n_folds):
        fold_ids = all_ids[i * fold_size: (i + 1) * fold_size if i < n_folds - 1 else N]
        sorted_fold = sorted(((sid, sample_rewards[sid]) for sid in fold_ids),
                             key=lambda x: x[1], reverse=True)
        fi, fr = zip(*sorted_fold)
        folds.append((fi, fr))
        sorted_ids.extend(fi)

    plt.figure(figsize=(7, 3))

    # 1. Find overall reward range for normalization
    all_rewards = np.concatenate([r for _, r in folds])
    norm = mpl.colors.Normalize(vmin=all_rewards.min(), vmax=all_rewards.max())
    cmap = plt.cm.viridis  # Can change to plasma / jet / coolwarm etc.

    # 2. Draw scatter plot per fold
    offset = 0
    sc = None
    for i, (_, rewards) in enumerate(folds):
        x_range = np.arange(offset, offset + len(rewards))
        sc = plt.scatter(
            x_range,
            rewards,
            c=rewards,
            cmap=cmap,
            norm=norm,
            s=20,
            label=f"Fold {i+1}"
        )
        offset += len(rewards)

    # 3. Titles and axes
    plt.title(f"K-Group Curriculum (K = {n_folds})")
    plt.xlabel("Sample Rank", fontsize=12)
    plt.ylabel("Avg Reward", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    # 4. Add colorbar for all points (pass one scatter object)
    # plt.colorbar(sc, label="Reward (Difficulty)")

    # 5. Save the figure
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"✅ Plot saved -> {plot_path}")

    return sorted_ids, folds


# ======================================================
# 6. Main pipeline
# ======================================================
def main(cfg):
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Step1: Scores
    print(">>> Step1: Computing Difficulty and Representativeness scores...")
    diff_scores = compute_difficulty_scores(cfg["reward_file"])
    rep_scores = compute_cosine_scores(cfg["grads_file"])
    scores_path = os.path.join(cfg["output_dir"], "sample_scores.jsonl")
    all_ids = sorted(set(diff_scores) | set(rep_scores))
    save_jsonl(scores_path, [
        {"sample_id": sid,
         "difficulty": diff_scores.get(sid),
         "representativeness": rep_scores.get(sid)} for sid in all_ids
    ])
    print(f"✅ Scores saved -> {scores_path}")

    # Step2: Selection
    print(">>> Step2: MaxMin selection...")
    embs = load_embeddings(cfg["embs_file"], cfg["device"])
    scores_dict = {sid: cfg["alpha"] * diff_scores.get(sid, 0.0) + rep_scores.get(sid, 0.0) for sid in all_ids}
    selected_ids, vs = maxmin_select(embs, scores_dict, cfg["select_k"])

    save_jsonl(os.path.join(cfg["output_dir"], "selected_samples.jsonl"), [
        {"sample_id": sid,
         "v": v,
         "difficulty": diff_scores.get(sid, 0.0),
         "representativeness": rep_scores.get(sid, 0.0),
         "total": diff_scores.get(sid, 0.0) + rep_scores.get(sid, 0.0)}
        for sid, v in zip(selected_ids, vs)
    ])
    print(f"✅ Selected {len(selected_ids)} samples")

    # Step3: Curriculum
    print(">>> Step3: Curriculum sorting...")
    sorted_output = os.path.join(cfg["output_dir"], f"sorted_{cfg['select_k']}.npy")
    plot_path = os.path.join(cfg["output_dir"], "curriculum.pdf")
    random_fold_sort_with_visual(selected_ids, cfg["reward_file"], sorted_output, cfg["n_folds"], plot_path)
    print(f"✅ Curriculum saved -> {sorted_output}, {plot_path}")

    print("\nExample of first 10 samples:")
    for sid, v in zip(selected_ids[:10], vs[:10]):
        print(f"id={sid}, v={v:.4f}, diff={diff_scores.get(sid,0.0):.4f}, "
              f"rep={rep_scores.get(sid,0.0):.4f}, total={diff_scores.get(sid,0.0)+rep_scores.get(sid,0.0):.4f}")


# ======================================================
# 0. Global configuration (modify here)
# ======================================================
CONFIG = {
    "reward_file": "./train_diff/all_merged_sorted.jsonl",
    "grads_file": "./grads/grads_all_sorted_dedup.pt",
    "embs_file": "./embs/selected/all_unique_embs.pt",
    "output_dir": "./scores",

    "select_k": 1024,
    "alpha": 1,
    "n_folds": 16,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


if __name__ == "__main__":
    main(CONFIG)