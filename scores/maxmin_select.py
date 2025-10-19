#!/usr/bin/env python3
import torch
import json
import os
from tqdm import tqdm
import numpy as np

# ======================================================
# 1. 数据加载
# ======================================================
def load_embeddings(path, device="cuda"):
    embs = torch.load(path, map_location=device)  # [N, D]
    if isinstance(embs, (list, tuple)):  # 保守判断
        embs = embs[0]
    return embs.to(device)

def load_scores(path):
    diff_dict = {}
    rep_dict = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sid = int(rec["sample_id"])
            diff_dict[sid] = rec.get("difficulty", 0.0)
            rep_dict[sid] = rec.get("representativeness", 0.0)
    return diff_dict, rep_dict

# ======================================================
# 2. maxmin 选择
# ======================================================
def maxmin_select(embs, scores_dict, select_k, init_id=None):
    """
    embs: torch.Tensor [N, D] (在 GPU 上)
    scores_dict: {sample_id: diff+rep}
    select_k: int, 子集大小
    init_id: 起始点 id (默认挑 diff+rep 最大的)
    返回:
        selected_ids: list of sample_id
        selected_vs:  list of v 值
    """
    device = embs.device
    N, D = embs.shape

    ids = torch.arange(N, device=device)
    embs_norm = embs / (embs.norm(dim=1, keepdim=True) + 1e-12)

    # --- 1. 初始点
    if init_id is None:
        init_id = max(scores_dict.keys(), key=lambda x: scores_dict[x])
    selected_ids = [init_id]
    selected_vs = [1.0]  # 初始点 v 设为 1

    # --- 2. 初始化最近距离
    sim_init = (embs_norm @ embs_norm[init_id].unsqueeze(1)).squeeze(1)  # [N]
    dists = 1 - sim_init

    # --- 3. 迭代选择
    scores_arr = torch.tensor(
        [scores_dict.get(int(i.item()), 0.0) for i in ids],
        dtype=torch.float32,
        device=device,
    )

    for _ in tqdm(range(1, select_k), desc="maxmin selecting"):
        # 归一化距离 v
        min_val, max_val = dists.min(), dists.max()
        v = (dists - min_val) / (max_val - min_val + 1e-12)

        # 计算最终分数
        final_scores = v * scores_arr

        # 排除已选
        final_scores[torch.tensor(selected_ids, device=device)] = -float("inf")

        # 挑下一个
        new_id = int(torch.argmax(final_scores).item())
        selected_ids.append(new_id)
        selected_vs.append(float(v[new_id].item()))

        # 更新最近距离
        sim_new = (embs_norm @ embs_norm[new_id].unsqueeze(1)).squeeze(1)
        dist_new = 1 - sim_new
        dists = torch.minimum(dists, dist_new)

    return selected_ids, selected_vs

# ======================================================
# 3. 主函数
# ======================================================
def main():
    
    select_k = 256

    embs_file = "/storage_fast/lwang/SeqRecDistill/RRec/embs/selected/all_unique_embs.pt"
    scores_file = "/storage_fast/lwang/SeqRecDistill/RRec/scores/sample_scores.jsonl"
    save_pt = f"/storage_fast/lwang/SeqRecDistill/RRec/scores/selected_indices_{select_k}.npy"
    save_jsonl = "/storage_fast/lwang/SeqRecDistill/RRec/scores/selected_samples.jsonl"

    alpha = 2.0
    

    os.makedirs(os.path.dirname(save_pt), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载数据
    embs = load_embeddings(embs_file, device=device)
    diff_dict, rep_dict = load_scores(scores_file)
    scores_dict = {k: alpha * diff_dict[k] + rep_dict[k] for k in diff_dict.keys()}

    # 运行 maxmin
    selected_ids, selected_vs = maxmin_select(embs, scores_dict, select_k=select_k)

    # 保存 torch 格式
    np.save(save_pt, selected_ids)
    print(f"✅ 已保存选择结果到 {save_pt}")

    # 保存 JSONL (详细信息)
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

    # 打印前 10 个样本
    print("\n示例选出的前 10 个样本:")
    for sid, v in zip(selected_ids[:10], selected_vs[:10]):
        d = diff_dict.get(sid, 0.0)
        r = rep_dict.get(sid, 0.0)
        print(f"id={sid}, v={v:.4f}, difficulty={d:.4f}, represent={r:.4f}, total={d+r:.4f}")


if __name__ == "__main__":
    main()