import os
import datasets
import rich
import torch
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import GenerationConfig, EarlyStoppingCallback
from data_collators.data_collator import RRecDataCollator as DataCollator
from paths import model_names
from trainers.utils import get_tokenizer, calculate_metrics, MetricUpdater, get_compute_metrics
from trainers.RecPOTrainer import RecPOTrainer, RecPOTrainingArguments
import numpy as np
import random
import json
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import glob
from tqdm import tqdm
from torch import distributed as dist

class EvalRecPOTrainer(RecPOTrainer):
    def compute_rec_score(self, model, inputs, similarity=None, train_eval='eval'):

        if train_eval == 'train':
            num_samples = self.args.generation_config.num_return_sequences
            user_input_prefix = "multi_user"
        else:
            num_samples = 1
            user_input_prefix = "user"

        seq_labels = inputs['seq_labels']
        seq_input_ids = inputs['seq_input_ids']
        batch_size = seq_labels.shape[0]
        seq_labels = seq_labels.view(batch_size, 1, 1).expand(-1, num_samples, -1)
        seq_input_ids = seq_input_ids.view(
            batch_size, 1, -1
        ).expand(-1, num_samples, -1).reshape(batch_size * num_samples, -1)

        if similarity is None:
            similarity, _ = self.compute_sim_val(
                model,
                inputs | {'seq_input_ids': seq_input_ids},
                user_input_prefix=user_input_prefix,
            )

        shaped_sim = similarity.view(batch_size, num_samples, -1).float()
        cutoff = self.args.reward_ndcg_k
        dcg_k, _ = calculate_metrics(shaped_sim, seq_labels, cutoff)

        ndcg = dcg_k.sum(dim=-1)  # (B, num_samples)

        sim_softmax = shaped_sim.softmax(dim=-1)
        sim_softmax = sim_softmax.gather(2, seq_labels)
        sim_softmax = sim_softmax.squeeze(2)  # (B, num_samples)

        rewards = (1 - self.args.reward_softmax_weight) * ndcg + \
                  self.args.reward_softmax_weight * sim_softmax

        result = {
            'rewards_per_sample': rewards.detach().cpu().tolist(),
            'ndcg_per_sample': ndcg.detach().cpu().tolist(),
            'hit_ratio_per_sample': (ndcg > 0).int().cpu().tolist(),
        }

        return result



def evaluate(
        n_train = 2048,
        dataset_category: str = "CDs_and_Vinyl",
        dataset_dir="data/CDs_and_Vinyl_0_2022-10-2023-10",
        run_name: str = "difficulty-debug",
        batch_size: int = 16,
        window_size: int = 5,
        model: str = 'gemma',
        output_dir = "./checkpoints",
        max_new_tokens=256,
        group_size=4,
        use_lora=True,
        seed=42,
        use_vllm=False,
        early_stopping_patience=3,
        checkpoint_path = "/storage_fast/lwang/SeqRecDistill/RRec/checkpoints/checkpoint-256-gemma-CDs_and_Vinyl-1/checkpoint-28",
        **kwargs,
):
    datasets.disable_progress_bars()
    if checkpoint_path is not None:
        result = checkpoint_path.split("/")[-2] + "-" + checkpoint_path.split("/")[-1].split("-")[-1]
        output_file=f"./train_diff/{n_train}_{result}_difficulty.jsonl"
    else:
        output_file=f"./train_diff/{n_train}_difficulty.jsonl"
    # 选择模型
    if model == 'gemma':
        model_name = model_names["Gemma-2-2b-it"]
        from models.gemma_models import Gemma2RRecCasualLM as ModelClass, Gemma2RRecConfig as ConfigClass
    elif model == 'qwen3b':
        model_name = model_names["Qwen2.5-3B-Instruct"]
        from models.qwen_models import Qwen2RRecCasualLM as ModelClass, Qwen2RRecConfig as ConfigClass
    elif model == 'qwen1.5b':
        model_name = model_names["Qwen2.5-1.5B-Instruct"]
        from models.qwen_models import Qwen2RRecCasualLM as ModelClass, Qwen2RRecConfig as ConfigClass
    elif model == 'qwen4b':
        model_name = model_names["Qwen3-4B"]
        from models.qwen_models import Qwen2RRecCasualLM as ModelClass, Qwen2RRecConfig as ConfigClass
    elif model == 'deepseek':
        model_name = model_names["DeepSeek-R1-Distill-Qwen-1.5B"]
        from models.qwen_models import Qwen2RRecCasualLM as ModelClass, Qwen2RRecConfig as ConfigClass
    else:
        raise NotImplementedError


    output_dir = os.path.join(output_dir, run_name)

    accelerator = Accelerator()
    if accelerator.is_main_process:
        rich.print("Arguments: ", locals())


    dset = datasets.load_from_disk(dataset_dir)
    
    all_inds = np.arange(10748)
    

    
    
    
    
    selected_inds = np.arange(10748)

    id_map = {i: int(selected_inds[i]) for i in range(len(selected_inds))}


    def add_sample_id(example, idx):
        return {"sample_id": str(id_map[idx])}

    dset['train'] = dset['train'].map(add_sample_id, with_indices=True)

    path = f"/storage_fast/lwang/SeqRecDistill/RRec/train_diff/remian_inds.npy"
    remain_inds = np.load(path)
    dset['train'] = dset['train'].select(remain_inds)

    print(dset['train'][4])
    print(dset['train'][50])

    tokenizer = get_tokenizer(model_name)
    emb_token, emb_end_token = '<answer>', '</answer>'

    config = ConfigClass.from_pretrained(model_name)
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    tokenizer.save_pretrained(output_dir)

    
    base_model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": accelerator.process_index},
        config=config
    )

    # generation config
    gen_config = GenerationConfig.from_pretrained(model_name)
    gen_config.max_new_tokens = max_new_tokens
    gen_config.num_return_sequences = group_size
    gen_config.top_k = 200
    gen_config.top_p = 1.0
    gen_config.temperature = 1.5

    # LoRA
    if use_lora:
    
        if checkpoint_path is not None:
            base_model = PeftModel.from_pretrained(base_model, checkpoint_path)
            print("="*100)
            print(f"Loading LoRA weights from: {checkpoint_path}")
            print("="*100)


    training_args = RecPOTrainingArguments(
        seed=seed,

        item_emb_batch_size=64,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        max_grad_norm=1.,
        num_train_epochs=1,
        learning_rate=1e-5,
        fp16=True,

        save_strategy="steps",
        save_steps=1,
        save_only_model=False,
        save_total_limit=1,
        load_best_model_at_end=True,

        eval_strategy="steps",
        eval_steps=1,

        fp16_full_eval=True,
        per_device_eval_batch_size=batch_size,
        metric_for_best_model='eval_valid_ndcg@10',
        eval_on_start=False,
        batch_eval_metrics=True,

        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        lr_scheduler_type="constant",
        warmup_steps=16,
        report_to='none',
        run_name=run_name,
        logging_dir="./logs",
        gradient_checkpointing_kwargs={'use_reentrant': False},

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,

        gather_negs_across_processes=True,
        generation_config=gen_config,

        train_generation_batch_size=batch_size,
        test_generation_batch_size=batch_size,

        dataset_window_size=5,
        dataset_category=dataset_category,
        emb_token=emb_token,
        emb_end_token=emb_end_token,
        use_vllm=use_vllm,

    )
    metric_updater = MetricUpdater(ks=[5, 10, 20, 50])

    trainer = EvalRecPOTrainer(
        model=base_model,
        compute_metrics=get_compute_metrics(metric_updater, ),
        data_collator=DataCollator(tokenizer=tokenizer, return_tensors="pt"),
        full_dataset=dset,
        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience)],
        processing_class=tokenizer,
        args=training_args,
    )



    is_distributed = torch.distributed.is_initialized() and accelerator.num_processes > 1
    rank = accelerator.process_index
    world_size = accelerator.num_processes

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(
            trainer.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

    dataloader = DataLoader(
        trainer.train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=trainer.data_collator,
    )
     
    print("="*100)
    print(f"batch_size      : {batch_size}")
    print(f"len(dataloader) : {len(dataloader)}")
    print("="*100)

    if is_distributed:
        rank_file = f"{output_file}.rank{rank}"
    else:
        rank_file = output_file

    with open(rank_file, "w", encoding="utf-8") as f:
        model = trainer.get_model_for_eval()
        model.eval()
        for step, batch in enumerate(tqdm(dataloader, desc="Evaluating on Train set", total=len(dataloader))):
            print(f"step        : {step}")
            print(f"batch.keys(): {batch.keys()}")
            print(f"batch['sample_id']: {batch['sample_id']}")
            batch = trainer._prepare_inputs(batch)
            print(f"generate in train....")
            batch |= trainer._generate_in_train(model, batch)
            print(f"compute rec score....")
            with torch.no_grad():
                result = trainer.compute_rec_score(model, batch, train_eval="train")

            rewards = result.get("rewards_per_sample", [])
            ndcgs = result.get("ndcg_per_sample", [])
            hit_ratios = result.get("hit_ratio_per_sample", [])
            sample_ids = batch['sample_id']

            for i, sid in enumerate(sample_ids):
                rec = {
                    "sample_id": str(sid),
                    "reward": rewards[i],
                    "ndcg": ndcgs[i],
                    "hit_ratio": hit_ratios[i],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if accelerator.is_main_process and step % 10 == 0:
                print(f"[Eval][rank {rank}] processed {step} batches")

    if is_distributed:
        dist.barrier()   
    if is_distributed and accelerator.is_main_process:
        with open(output_file, "w", encoding="utf-8") as fout:
            for rf in sorted(glob.glob(f"{output_file}.rank*")):
                with open(rf, "r", encoding="utf-8") as fin:
                    for line in fin:
                        fout.write(line)
                os.remove(rf)  
        print(f"[Eval] merged {world_size} rank files into {output_file}")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    import fire
    fire.Fire(evaluate)