
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import datasets
import rich
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Dict, List, Any, Callable
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel
from transformers.trainer_callback import EarlyStoppingCallback
from transformers.generation.configuration_utils import GenerationConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from data_collators.data_collator import RRecDataCollator as DataCollator
from paths import model_names
from trainers.utils import get_compute_metrics, get_tokenizer, MetricUpdater
from trainers.RecPOTrainer import (
    RecPOTrainer,
    RecPOTrainingArguments)
from tqdm import tqdm
# from trainer.GRecTrainer import GRecTrainer
from kernels import NeuralTangentKernel
from DataDistill import DatasetDistillation
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'OFF'

class GenrateUserEmbeddings(RecPOTrainer):
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module],
                 args: RecPOTrainingArguments,
                 data_collator: Optional[DataCollator],
                 full_dataset: Optional["datasets.DatasetDict"],
                 processing_class: Optional[PreTrainedTokenizerBase] = None,
                 compute_metrics: Optional[Callable] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
    ):  

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            full_dataset=full_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    def genrate_user_embeddings(self,
                      num_items_in_batch = None,
                      a = 0,
                      b = 10,
                      use_reasoning = True,
                      **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate user embeddings in parallel across multiple GPUs or single GPU.
        Automatically detects the environment and adapts accordingly.
        
        Returns:
            tuple: (user_embeddings, item_embeddings, seq_labels)
        """
        # 检查train_dataset是否存在
        if self.train_dataset is None:
            raise ValueError("train_dataset is None, cannot generate embeddings")
        
        # 检测是否为多卡环境
        is_multi_gpu = self.accelerator.num_processes > 1
        


        train_dataset = self.train_dataset.select(range(a, min(b, len(self.train_dataset))))
        # train_dataset = self.train_dataset

        if is_multi_gpu:
            from torch.utils.data import DistributedSampler
            
            sampler = DistributedSampler(
                train_dataset,  # type: ignore
                shuffle=False,
                drop_last=False
            )
            
            dataloader = torch.utils.data.DataLoader(
                train_dataset,  # type: ignore
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=False,
                prefetch_factor=self.args.dataloader_prefetch_factor,
                sampler=sampler,  
            )
        else:


            dataloader = torch.utils.data.DataLoader(
                train_dataset,  # type: ignore
                batch_size=self.args.per_device_eval_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                persistent_workers=False,
                prefetch_factor=self.args.dataloader_prefetch_factor,
                shuffle=False,  
            )
        
        # dataloader = self.accelerator.prepare(dataloader)

        eval_model = self.get_model_for_eval()
        all_embeddings = []
        all_seq_labels = []
        all_user_ids = []
        with torch.no_grad():

            # self._generate_item_embeddings(eval_model)
            # torch.save(self.item_hs, f"./item_hs.pt")


            bar = tqdm(dataloader, desc="Generating user embeddings")

            for batch in bar:
                batch = self._prepare_inputs(batch)
                batch_seq_labels = batch['seq_labels']
                batch_user_ids = batch['train_data_id']
                
                
                if use_reasoning:
                    batch_input = self._generate_in_train(eval_model, batch)
                    
                    
                    if 'multi_user_input_ids' in batch_input:
                        input_ids = batch_input['multi_user_input_ids']
                        attention_mask = batch_input['multi_user_attention_mask']
                    elif 'user_input_ids' in batch_input:
                        input_ids = batch_input['user_input_ids']
                        attention_mask = batch_input['user_attention_mask']
                    else:
                      
                        print(f"Available keys in batch_input: {list(batch_input.keys())}")
                        raise KeyError("Neither 'multi_user_input_ids' nor 'user_input_ids' found in batch_input")
                else:
                    input_ids       = batch['user_gen_input_ids']
                    attention_mask  = batch['user_gen_attention_mask']
                
                
                _, batch_emb = self.model(
                    input_ids, 
                    attention_mask,
                    return_with_last_hidden_states=True  
                )
                
                
                
                all_user_ids.append(torch.tensor(batch_user_ids).to(input_ids.device))
                all_seq_labels.append(batch_seq_labels) 
                all_embeddings.append(batch_emb)
            
            all_user_ids = torch.cat(all_user_ids, dim=0)
            all_seq_labels = torch.cat(all_seq_labels, dim=0)
            all_embeddings = torch.cat(all_embeddings, dim=0)

        del eval_model
        
        if is_multi_gpu:

            gathered_embeddings = self.accelerator.gather_for_metrics(all_embeddings)
            gathered_seq_labels = self.accelerator.gather_for_metrics(all_seq_labels)
            gathered_user_ids = self.accelerator.gather_for_metrics(all_user_ids)

            # print('-'*100)
            # print(f"all_user_ids: {all_user_ids}")
            # print(f"gathered_user_ids: {gathered_user_ids}")
            # print('-'*100)

    
            if self.accelerator.is_main_process:

                return gathered_embeddings, self.item_hs, gathered_seq_labels, gathered_user_ids
            else:
 
                return torch.empty(0), torch.empty(0), [], []
        else:

            return all_embeddings, self.item_hs, all_seq_labels, all_user_ids


def train(
        output_dir="./checkpoints",

        run_name: str = "debug-v1",

        train_batch_size: int = 4,
        eval_batch_size: int = 1,

        train_generation_batch_size=16,
        test_generation_batch_size=32,

        item_emb_batch_size: int = 128,
        warmup_steps: int = 32,

        eval_freq=8,
        early_stopping_patience=8,
        eval_on_start: bool = True,
        logging_dir="./logs",
        gradient_accumulation_steps: int = 8,

        num_train_epochs: int = 3,
        learning_rate: float = 1e-5,

        cleanup_previous_checkpoints=False,
        
        dataset_category: str = "CDs_and_Vinyl",
        dataset_dir="data/CDs_and_Vinyl_0_2022-10-2023-10",
        
        use_lora=True,
        seed=42,
        model = 'gemma',

        resume_from_checkpoint: bool = False,
        window_size: int = 20,
        gather_negs_across_processes=True,
        max_new_tokens=256,
        group_size=1,
        lr_scheduler_type='constant',
        use_vllm=False,  
        a=0,
        b=10,
        use_reasoning = True,
        **kwargs,

):
    """
    训练函数 - 支持多卡并行训练
    
    启动多卡训练的方法：
    1. 使用 accelerate launch:
       accelerate launch --multi_gpu --num_processes=4 dataset_distillation.py
    
    2. 使用 torchrun:
       torchrun --nproc_per_node=4 dataset_distillation.py
    
    3. 使用 python -m torch.distributed.launch:
       python -m torch.distributed.launch --nproc_per_node=4 dataset_distillation.py
    """
    trainer_extra_kwargs = dict()
    lora_kwargs = dict()
    for k in kwargs:
        if k.startswith('trainer'):
            trainer_extra_kwargs[k.replace('trainer_', '')] = kwargs[k]
        else:
            lora_kwargs[k] = kwargs[k]
    del kwargs

    datasets.disable_progress_bars()
    if model == 'gemma':
        model_name = model_names["Gemma-2-2b-it"]
        from models.gemma_models import (Gemma2RRecCasualLM as ModelClass,
                                         Gemma2RRecConfig as ConfigClass)
    elif model == 'qwen':
        model_name = model_names["Qwen2.5-3B-Instruct"]
        from models.qwen_models import (Qwen2RRecCasualLM as ModelClass,
                                        Qwen2RRecConfig as ConfigClass)
    else:
        raise NotImplementedError
    import os
    output_dir = os.path.join(output_dir, run_name)

    accelerator = Accelerator()
    rich.print(accelerator.deepspeed_plugin)


    ################## set dataset ##################

    dset = datasets.load_from_disk(dataset_dir)



    tokenizer = get_tokenizer(model_name)

    emb_token = '<answer>'
    emb_end_token = '</answer>'

    config = ConfigClass.from_pretrained(model_name)
    config.use_cache = False
    config.pad_token_id = tokenizer.pad_token_id
    tokenizer.save_pretrained(output_dir)

    ################### set model ###################

    # checkpoint_path = os.path.abspath("/checkpoints/debug-v1/checkpoint-234")
    base_model = ModelClass.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # attn_implementation="flash_attention_2",
        device_map={"": accelerator.process_index},
        config=config
    )


    ################### set generation ###################
    gen_config = GenerationConfig.from_pretrained(model_name)
    gen_config.max_new_tokens = max_new_tokens
    gen_config.num_return_sequences = group_size
    # gen_config.top_k = 200
    gen_config.top_p = 1.0
    # gen_config.temperature = 1.5

    ################################################################


    peft_config_dict = {
        "inference_mode": False,
        "target_modules": [
            'k_proj', 'v_proj', 'q_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ],
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    peft_config_dict.update(lora_kwargs)




    if use_lora:
        print("*"*100)
        checkpoint_path = "./checkpoints/debug-v1/checkpoint-234"
        print(f"Loading LoRA weights from: {checkpoint_path}")
        print("*"*100)
        
        base_model = PeftModel.from_pretrained(base_model, checkpoint_path)

        print("*"*100)
        print("LoRA Checkpoint loaded successfully!")
        print("*"*100)
            
    else:
        if accelerator.is_main_process:
            rich.print("No PEFT applied, training the base model")



    # base_model.enable_input_require_grads()
    ################### set trainer ###################
    # calculate steps required for half an epoch
    eval_steps = len(dset['train']) / (train_batch_size *
                                       gradient_accumulation_steps * 3)
    eval_steps = eval_steps // eval_freq

    training_args = RecPOTrainingArguments(
        seed=seed,

        item_emb_batch_size=item_emb_batch_size,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        max_grad_norm=1.,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        fp16=True,

        save_strategy="steps",
        save_steps=eval_steps,
        save_only_model=False,
        save_total_limit=10,
        load_best_model_at_end=True,

        eval_strategy="steps",
        eval_steps=eval_steps,

        fp16_full_eval=True,
        per_device_eval_batch_size=eval_batch_size,
        metric_for_best_model='eval_ndcg@10',
        eval_on_start=eval_on_start,
        batch_eval_metrics=True,

        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        report_to='none',
        run_name=run_name,
        logging_dir=logging_dir,
        gradient_checkpointing_kwargs={'use_reentrant': False},

        ddp_find_unused_parameters=False,
        remove_unused_columns=False,

        gather_negs_across_processes=gather_negs_across_processes,
        generation_config=gen_config,

        train_generation_batch_size=train_generation_batch_size,
        test_generation_batch_size=test_generation_batch_size,

        dataset_window_size=window_size,
        dataset_category=dataset_category,
        emb_token=emb_token,
        emb_end_token=emb_end_token,
        use_vllm=use_vllm,
        **trainer_extra_kwargs,

    )
    metric_updater = MetricUpdater(ks=[5, 10, 20, 50])

    gen_user_embeddings = GenrateUserEmbeddings(
        model=base_model,
        compute_metrics=get_compute_metrics(metric_updater, ),
        data_collator=DataCollator(tokenizer=tokenizer,
                                    return_tensors="pt"),

        full_dataset=dset,  # type: ignore

        callbacks=[EarlyStoppingCallback(
            early_stopping_patience=early_stopping_patience)],
        processing_class=tokenizer,
        args=training_args,
    )
    
    results = gen_user_embeddings.genrate_user_embeddings(a=a,b=b, use_reasoning = use_reasoning)



    if accelerator.is_main_process:
    # save results
        print('-'*100)
        print(f"results: {results}")
        print('-'*100)
        if use_reasoning:
            torch.save(results, f".embs/selected_checkpoint/CDs_and_Vinyl_results_{a}_{b}.pt")
            print(f"Results saved to ./embs/selected_checkpoint/CDs_and_Vinyl_results_{a}_{b}.pt")
        else:
            torch.save(results, f".embs/selected_checkpoint_no_reasoning/CDs_and_Vinyl_results_{a}_{b}.pt")
            print(f"Results saved to ./embs/selected_checkpoint_no_reasoning/CDs_and_Vinyl_results_{a}_{b}.pt")

    ################################################################

if __name__ == "__main__":

    # CDs_and_Vinyl 10748 
    total_samples = 10748
    n = 4  
    batch_size = total_samples // n  
    use_reasoning = True

    # for i in range(n):
    for i in [2]: 
        print('*'*100)
        print(f"Batch {i+1}: Begin: {i * batch_size}, End: {min(i * batch_size + batch_size, total_samples)}")
        print('*'*100)
        a = i * batch_size
        b = min(a + batch_size, total_samples) if i < n - 1 else total_samples  
        print(f"Batch {i+1}: Begin: {a}, End: {b}")
        train(a=a, b=b, use_reasoning = use_reasoning)

