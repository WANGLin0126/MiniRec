#!/usr/bin/env python3


import json
import os



import numpy as np
from typing import Dict, List, Any
import argparse

def load_trainer_state(checkpoint_path: str) -> Dict[str, Any]:

    trainer_state_path = os.path.join(checkpoint_path, "trainer_state.json")
    if not os.path.exists(trainer_state_path):
        raise FileNotFoundError(f"Not found trainer_state_path: {trainer_state_path}")
    
    with open(trainer_state_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_config(checkpoint_path: str) -> Dict[str, Any]:
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        print(f"Not found model config file: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_training_args(checkpoint_path: str) -> Dict[str, Any]:
    args_path = os.path.join(checkpoint_path, "training_args.bin")
    if not os.path.exists(args_path):
        args_json_path = os.path.join(checkpoint_path, "training_args.json")
        if os.path.exists(args_json_path):
            with open(args_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            print(f"Not found training args file: {args_path} or {args_json_path}")
            return {}
    
    try:
        import torch
        args_dict = torch.load(args_path, map_location='cpu')
        serializable_args = {}
        for key, value in args_dict.__dict__.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_args[key] = value
            else:
                serializable_args[key] = str(value)
        return serializable_args
    except Exception as e:
        print(f"Failed to load training args file: {e}")
        return {}

def extract_model_settings(checkpoint_path: str) -> Dict[str, Any]:
    settings = {
        'model_config': {},
        'training_args': {},
        'checkpoint_info': {}
    }

    settings['model_config'] = load_model_config(checkpoint_path)
    
    settings['training_args'] = load_training_args(checkpoint_path)
    
    checkpoint_name = os.path.basename(checkpoint_path)
    settings['checkpoint_info'] = {
        'checkpoint_name': checkpoint_name,
        'checkpoint_path': checkpoint_path,
        'step': int(checkpoint_name.split('-')[1]) if '-' in checkpoint_name else 0
    }
    
    return settings

def extract_eval_metrics(log_history: List[Dict]) -> List[Dict]:
    eval_test_metrics = []
    
    for log in log_history:
        eval_test_log = {}
        for key, value in log.items():
            if key.startswith('eval_test_'):
                eval_test_log[key] = value
        
        if eval_test_log:  
            eval_test_log['step'] = log.get('step', 0)
            eval_test_log['epoch'] = log.get('epoch', 0)
            eval_test_metrics.append(eval_test_log)
    
    return eval_test_metrics

def analyze_recommendation_metrics(eval_test_metrics: List[Dict]) -> Dict[str, Any]:
    if not eval_test_metrics:
        return {}
    
    ndcg_metrics = {
        'ndcg@5': [],
        'ndcg@10': [],
        'ndcg@20': [],
        'ndcg@1000': []
    }
    
    hit_rate_metrics = {
        'hit_rate@5': [],
        'hit_rate@10': [],
        'hit_rate@20': []
    }
    
    reward_metrics = {
        'reward': [],
        'reward_max': [],
        'reward_min': []
    }
    
    steps = []
    
    for metric in eval_test_metrics:
        step = metric.get('step', 0)
        steps.append(step)
        
        for k in [5, 10, 20, 1000]:
            key = f'eval_test_ndcg@{k}'
            if key in metric:
                ndcg_metrics[f'ndcg@{k}'].append(metric[key])
        
        for k in [5, 10, 20]:
            key = f'eval_test_hit_rate@{k}'
            if key in metric:
                hit_rate_metrics[f'hit_rate@{k}'].append(metric[key])
        
        for key in ['eval_test_reward', 'eval_test_reward_max', 'eval_test_reward_min']:
            if key in metric:
                reward_key = key.replace('eval_test_', '')
                reward_metrics[reward_key].append(metric[key])
    
    analysis = {
        'total_eval_steps': len(eval_test_metrics),
        'steps': steps,
        'ndcg_analysis': {},
        'hit_rate_analysis': {},
        'reward_analysis': {},
        'best_metrics': {},
        'final_metrics': {},
        'best_ndcg10_metrics': {}  
    }
    
    for metric_name, values in ndcg_metrics.items():
        if values:
            values = np.array(values)
            analysis['ndcg_analysis'][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'final': float(values[-1]),
                'best': float(np.max(values)),
                'best_step': int(steps[np.argmax(values)])
            }
    
    for metric_name, values in hit_rate_metrics.items():
        if values:
            values = np.array(values)
            analysis['hit_rate_analysis'][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'final': float(values[-1]),
                'best': float(np.max(values)),
                'best_step': int(steps[np.argmax(values)])
            }
    
    for metric_name, values in reward_metrics.items():
        if values:
            values = np.array(values)
            analysis['reward_analysis'][metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'final': float(values[-1])
            }
    
    if 'ndcg@10' in analysis['ndcg_analysis']:
        best_ndcg10_step = analysis['ndcg_analysis']['ndcg@10']['best_step']
        best_ndcg10_epoch = None
        
        for metric in eval_test_metrics:
            if metric.get('step', 0) == best_ndcg10_step:
                analysis['best_ndcg10_metrics'] = {
                    'step': best_ndcg10_step,
                    'epoch': metric.get('epoch', 0),
                    'ndcg@5': metric.get('eval_test_ndcg@5', None),
                    'ndcg@10': metric.get('eval_test_ndcg@10', None),
                    'ndcg@20': metric.get('eval_test_ndcg@20', None),
                    'ndcg@1000': metric.get('eval_test_ndcg@1000', None),
                    'hit_rate@5': metric.get('eval_test_hit_rate@5', None),
                    'hit_rate@10': metric.get('eval_test_hit_rate@10', None),
                    'hit_rate@20': metric.get('eval_test_hit_rate@20', None),
                    'reward': metric.get('eval_test_reward', None),
                    'reward_max': metric.get('eval_test_reward_max', None),
                    'reward_min': metric.get('eval_test_reward_min', None),
                    'softmax': metric.get('eval_test_softmax', None),
                    'advantage': metric.get('eval_test_advantage', None),
                    'advantage_max': metric.get('eval_test_advantage_max', None),
                    'advantage_min': metric.get('eval_test_advantage_min', None),
                    'loss': metric.get('eval_test_loss', None),
                    'runtime': metric.get('eval_test_runtime', None),
                    'samples_per_second': metric.get('eval_test_samples_per_second', None),
                    'output_length': metric.get('eval_test_output_length', None)
                }
                break
    
    if eval_test_metrics:
        final_metric = eval_test_metrics[-1]
        analysis['final_metrics'] = {
            'step': final_metric.get('step', 0),
            'epoch': final_metric.get('epoch', 0)
        }
        
        for key, value in final_metric.items():
            if key.startswith('eval_test_'):
                analysis['final_metrics'][key] = value
    
    return analysis

def print_metrics_summary(analysis: Dict[str, Any]):
    print("=" * 80)
    print("Recommendation Metrics Analysis")
    print("=" * 80)
    
    print(f"\nTotal Evaluation Steps: {analysis['total_eval_steps']}")
    
    print("\n" + "=" * 50)
    print("NDCG Metrics Analysis")
    print("=" * 50)
    for metric_name, stats in analysis['ndcg_analysis'].items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Final: {stats['final']:.6f}")
        print(f"  Best: {stats['best']:.6f} (Step {stats['best_step']})")
    
    print("\n" + "=" * 50)
    print("Hit Rate Metrics Analysis")
    print("=" * 50)
    for metric_name, stats in analysis['hit_rate_analysis'].items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Final: {stats['final']:.6f}")
        print(f"  Best: {stats['best']:.6f} (Step {stats['best_step']})")
    
    print("\n" + "=" * 50)
    print("Reward Metrics Analysis")
    print("=" * 50)
    for metric_name, stats in analysis['reward_analysis'].items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  Min: {stats['min']:.6f}")
        print(f"  Max: {stats['max']:.6f}")
        print(f"  Final: {stats['final']:.6f}")
    
    if analysis['best_ndcg10_metrics']:
        print("\n" + "=" * 50)
        print("All Metrics of NDCG@10 at Best Step (Step {}, Epoch {:.2f})".format(
            analysis['best_ndcg10_metrics']['step'],
            analysis['best_ndcg10_metrics']['epoch']
        ))
        print("=" * 50)
        
        print("\nNDCG Metrics:")
        for metric in ['ndcg@5', 'ndcg@10', 'ndcg@20', 'ndcg@1000']:
            value = analysis['best_ndcg10_metrics'].get(metric)
            if value is not None:
                print(f"  {metric}: {value:.6f}")
        
        print("\nHit Rate Metrics:")
        for metric in ['hit_rate@5', 'hit_rate@10', 'hit_rate@20']:
            value = analysis['best_ndcg10_metrics'].get(metric)
            if value is not None:
                print(f"  {metric}: {value:.6f}")
        
        print("\nReward Metrics:")
        for metric in ['reward', 'reward_max', 'reward_min']:
            value = analysis['best_ndcg10_metrics'].get(metric)
            if value is not None:
                print(f"  {metric}: {value:.6f}")
        
        print("\nOther Metrics:")
        other_metrics = ['softmax', 'advantage', 'advantage_max', 'advantage_min', 'loss']
        for metric in other_metrics:
            value = analysis['best_ndcg10_metrics'].get(metric)
            if value is not None:
                print(f"  {metric}: {value}")
        
        print("\nPerformance Metrics:")
        perf_metrics = ['runtime', 'samples_per_second', 'output_length']
        for metric in perf_metrics:
            value = analysis['best_ndcg10_metrics'].get(metric)
            if value is not None:
                if metric == 'runtime':
                    print(f"  {metric}: {value:.2f}s")
                elif metric == 'samples_per_second':
                    print(f"  {metric}: {value:.2f}")
                else:
                    print(f"  {metric}: {value:.2f}")
    
    if analysis['final_metrics']:
        print("\n" + "=" * 50)
        print("Final Evaluation Metrics (Step {}, Epoch {:.2f})".format(
            analysis['final_metrics']['step'],
            analysis['final_metrics']['epoch']
        ))
        print("=" * 50)
        for key, value in analysis['final_metrics'].items():
            if key not in ['step', 'epoch']:
                print(f"{key}: {value}")

def print_model_settings_summary(model_settings: Dict[str, Any]):
    print("\n" + "=" * 50)
    print("Model Settings Summary")
    print("=" * 50)
    
    print("\nCheckpoint Information:")
    checkpoint_info = model_settings['checkpoint_info']
    print(f"  Checkpoint Name: {checkpoint_info['checkpoint_name']}")
    print(f"  Checkpoint Path: {checkpoint_info['checkpoint_path']}")
    print(f"  Training Steps: {checkpoint_info['step']}")
    
    print("\nImportant Model Configurations:")
    model_config = model_settings['model_config']
    important_model_keys = [
        'model_type', 'architectures', 'vocab_size', 'hidden_size', 
        'num_hidden_layers', 'num_attention_heads', 'intermediate_size',
        'max_position_embeddings', 'pad_token_id', 'bos_token_id', 'eos_token_id'
    ]
    
    for key in important_model_keys:
        if key in model_config:
            value = model_config[key]
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            elif isinstance(value, list) and len(value) <= 3:
                print(f"  {key}: {value}")
            elif isinstance(value, str):
                print(f"  {key}: {value}")
    
    print("\nImportant Training Parameters:")
    training_args = model_settings['training_args']
    important_training_keys = [
        'learning_rate', 'num_train_epochs', 'per_device_train_batch_size',
        'per_device_eval_batch_size', 'gradient_accumulation_steps',
        'warmup_steps', 'max_grad_norm', 'lr_scheduler_type',
        'metric_for_best_model', 'save_strategy', 'eval_strategy',
        'save_steps', 'eval_steps', 'fp16', 'gradient_checkpointing',
        'dataset_category', 'dataset_window_size', 'emb_token', 'emb_end_token',
        'use_vllm', 'generation_config', 'train_generation_batch_size',
        'test_generation_batch_size', 'item_emb_batch_size'
    ]
    
    for key in important_training_keys:
        if key in training_args:
            value = training_args[key]
            if isinstance(value, (int, float)):
                print(f"  {key}: {value}")
            elif isinstance(value, bool):
                print(f"  {key}: {value}")
            elif isinstance(value, str):
                print(f"  {key}: {value}")
            elif isinstance(value, dict):
                print(f"  {key}: {value}")
    
    print("\nGeneration Configuration:")
    if 'generation_config' in training_args:
        gen_config = training_args['generation_config']
        if isinstance(gen_config, dict):
            print("\nGeneration Configuration:")
            for key, value in gen_config.items():
                if isinstance(value, (int, float, str, bool)):
                    print(f"  {key}: {value}")
    
    model_config_count = len(model_settings['model_config'])
    training_args_count = len(model_settings['training_args'])
    print(f"\nConfiguration Statistics:")
    print(f"  Total Number of Model Configuration Items: {model_config_count}")
    print(f"  Total Number of Training Parameters: {training_args_count}")
    print(f"  (Full Configuration Please See Saved JSON File)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="weighted-loss-256-no-reasoning-gemma-CDs_and_Vinyl-2")
    args = parser.parse_args()
    
    run_name = args.run_name
    checkpoint_dir = f"checkpoints/{run_name}"
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith('checkpoint-')]
    if not checkpoints:
        print("Error: Checkpoint not found")
        return
    
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Analyzing checkpoint: {checkpoint_path}")
    
    try:
        trainer_state = load_trainer_state(checkpoint_path)
        
        model_settings = extract_model_settings(checkpoint_path)
        
        eval_metrics = extract_eval_metrics(trainer_state['log_history'])
        
        if not eval_metrics:
            print("Warning: No evaluation metrics found")
            return
        
        metrics_analysis = analyze_recommendation_metrics(eval_metrics)
        
        analysis = {
            'model_settings': model_settings,
            'metrics_analysis': metrics_analysis
        }
        
        print_metrics_summary(metrics_analysis)
        
        print_model_settings_summary(model_settings)
        
        output_file = f"./outputs/rec_metrics_analysis_with_settings_{run_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main() 