import os
import json
import datasets
import rich
import torch
import os
import glob
import json
import numpy as np
from tqdm import tqdm
from accelerate import Accelerator
from peft import PeftModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.trainer_callback import EarlyStoppingCallback

from data_collators.data_collator import RRecDataCollator as DataCollator
from paths import model_names
from trainers.utils import (
    get_compute_metrics,
    get_tokenizer,
    MetricUpdater,
    calculate_metrics,
)
from trainers.RecPOTrainer import RecPOTrainer, RecPOTrainingArguments


class CombinedTrainer(RecPOTrainer):

    def compute_gradients_info(self, dataloader, model, prefix="multi_user"):

        grad_1st_list = []
        hvp_list = []
        user_ids_list = []

        model.train()
        model.requires_grad_(True)

        for batch in tqdm(dataloader, desc="Compute gradients info"):
            batch = self._prepare_inputs(batch)
            batch |= self._generate_in_train(model, batch)
            batch |= self.compute_rec_score(model, batch)
            user_ids = batch["train_data_id"]


            loss = self.batch_forward(model, batch, prefix=prefix)
            last_hs = self.last_token_hidden_states
            last_hs.retain_grad()

            grad_1st = torch.autograd.grad(
                loss, last_hs, create_graph=True
            )[0]  # (B, hidden_dim)


            v = grad_1st

            # Hessian-Vector Product: Hv = H @ v
            Hv = torch.autograd.grad(
                (grad_1st * v).sum(),
                last_hs,
                retain_graph=False
            )[0]  # (B, hidden_dim)

            grad_1st_list.append(grad_1st.detach())
            hvp_list.append(Hv.detach())
            user_ids_list.append(torch.tensor(user_ids).to(grad_1st.device))

        grad_1st_list = torch.cat(grad_1st_list, dim=0)   # (N, hidden_dim)
        hvp_list = torch.cat(hvp_list, dim=0)             # (N, hidden_dim)
        user_ids_list = torch.cat(user_ids_list, dim=0)   # (N,)

        return grad_1st_list, hvp_list, user_ids_list



def run_once(
    dataset_category,
    dset,
    run_name,
    batch_size,
    model,
    output_dir,
    max_new_tokens,
    group_size,
    use_lora,
    checkpoint_path,
    a,
    b,
    use_reasoning,
    use_vllm,
):
    accelerator = Accelerator()

    # === 模型选择 ===
    if model == "gemma":
        model_name = model_names["Gemma-2-2b-it"]
        from models.gemma_models import Gemma2RRecCasualLM as ModelClass, Gemma2RRecConfig as ConfigClass
    elif model == "qwen3b":
        model_name = model_names["Qwen2.5-3B-Instruct"]
        from models.qwen_models import Qwen2RRecCasualLM as ModelClass, Qwen2RRecConfig as ConfigClass
    elif model == "qwen1.5b":
        model_name = model_names["Qwen2.5-1.5B-Instruct"]
        from models.qwen_models import Qwen2RRecCasualLM as ModelClass, Qwen2RRecConfig as ConfigClass
    elif model == "qwen4b":
        model_name = model_names["Qwen3-4B"]
        from models.qwen_models import Qwen2RRecCasualLM as ModelClass, Qwen2RRecConfig as ConfigClass
    elif model == "deepseek":
        model_name = model_names["DeepSeek-R1-Distill-Qwen-1.5B"]
        from models.qwen_models import Qwen2RRecCasualLM as ModelClass, Qwen2RRecConfig as ConfigClass
    else:
        raise NotImplementedError



    tokenizer = get_tokenizer(model_name)
    emb_token, emb_end_token = "<answer>", "</answer>"

    config = ConfigClass.from_pretrained(model_name)
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    tokenizer.save_pretrained(output_dir)

    base_model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": accelerator.process_index},
        config=config,
    )

    gen_config = GenerationConfig.from_pretrained(model_name)
    gen_config.max_new_tokens = max_new_tokens
    gen_config.num_return_sequences = group_size
    gen_config.top_k = 200
    gen_config.top_p = 1.0
    gen_config.temperature = 1.5

    if use_lora and checkpoint_path is not None:
        base_model = PeftModel.from_pretrained(base_model, checkpoint_path)

    training_args = RecPOTrainingArguments(
        seed=42,
        item_emb_batch_size=64,
        per_device_train_batch_size=batch_size,
        eval_steps=1,
        output_dir=output_dir,
        generation_config=gen_config,
        dataset_category=dataset_category,
        emb_token=emb_token,
        emb_end_token=emb_end_token,
        use_vllm=use_vllm,
    )

    trainer = CombinedTrainer(
        model=base_model,
        compute_metrics=get_compute_metrics(MetricUpdater(ks=[5, 10, 20, 50])),
        data_collator=DataCollator(tokenizer=tokenizer, return_tensors="pt"),
        
        full_dataset=dset,

        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        processing_class=tokenizer,
        args=training_args,
    )

    from torch.utils.data import DistributedSampler
    sampler = DistributedSampler(
        dset["train"].select(range(a, b)),
        shuffle=False,
        drop_last=False
    )

    dataloader = torch.utils.data.DataLoader(
        dset["train"].select(range(a, b)),
        batch_size=batch_size,
        collate_fn=trainer.data_collator,
        shuffle=False,
        sampler=sampler,
    )

    model_eval = trainer.get_model_for_eval()
    grad_1st, hessian_trace, user_ids = trainer.compute_gradients_info(
        dataloader, model_eval, prefix="multi_user")


    gathered_grad_1st = accelerator.gather_for_metrics(grad_1st)
    gathered_hessian_trace = accelerator.gather_for_metrics(hessian_trace)
    gathered_user_ids = accelerator.gather_for_metrics(user_ids)



    if accelerator.is_main_process:
        os.makedirs("./grads", exist_ok=True)
        torch.save((gathered_grad_1st, gathered_hessian_trace, gathered_user_ids), f"./grads/{run_name}_{a}_{b}.pt")


def main(
    dataset_category="CDs_and_Vinyl",
    dataset_dir="data/CDs_and_Vinyl_0_2022-10-2023-10",
    run_name="merge-debug-v1",
    batch_size=2,
    model="gemma",
    output_dir="./checkpoints",
    max_new_tokens=256,
    group_size=1,
    use_lora=True,
    checkpoint_path="/path/to/your/checkpoint",
    use_reasoning=True,
    use_vllm=False,
):


    n_splits = 4

    for i in range(n_splits):
        if i != 3:
            continue

        
        dset = datasets.load_from_disk(dataset_dir)
        
        def add_sample_id(example, idx):
            return {"sample_id": str(idx)}

        dset["train"] = dset["train"].map(add_sample_id, with_indices=True)
        # dset["train"] = dset["train"].select(range(0, 16))
        
        # print(dset["train"][0].keys())
        total_samples = len(dset["train"])
        split_size = total_samples // n_splits

        a = i * split_size
        b = (i + 1) * split_size if i < n_splits - 1 else total_samples
        print(f"=== Processing split {i+1}/{n_splits}: a={a}, b={b} ===")
        run_once(
            dataset_category,
            dset,
            run_name,
            batch_size,
            model,
            output_dir,
            max_new_tokens,
            group_size,
            use_lora,
            checkpoint_path,
            a,
            b,
            use_reasoning,
            use_vllm,
        )


if __name__ == "__main__":
    import fire

    fire.Fire(main)