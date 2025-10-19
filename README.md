# MiniRec: Data-Efficient Reinforcement Learning for LLM-based Recommendation

# Abstract: 

The integration of reinforcement learning (RL) into large language models (LLMs) has opened new opportunities for recommender systems by eliciting reasoning and improving user preference modeling. However, RL-based LLM recommendation faces significant efficiency challenges, making full-data training costly. Existing data selection methods define sample value based on learnability or representativeness, yet their loss/gradient-driven or dataset coverage-driven criteria often misalign with RL learning dynamics, resulting in suboptimal performance.

To address this, we propose MiniRec, a data selection framework tailored for RL-based LLM recommendation. MiniRec evaluates sample learnability using key RL signals—rewards—pruning samples that are too easy (too high reward) or too difficult (consistently low reward). It assesses representativeness by aligning sample gradients with the approximated  ``ideal'' global RL optimization trajectory, selecting samples that mainly drive model updates, and it also enforces diversity to reduce redundancy. Combined with a curriculum learning strategy from easy to hard samples, MiniRec significantly reduces training cost while largely preserving performance. Extensive experiments demonstrate MiniRec’s effectiveness, highlighting the importance of reward-aligned, trajectory-informed data selection in RL-based LLM recommendation.





## 1. Preparations

### Requirements

The project requires Python 3.11 and the following dependencies:

```bash
pip install -r requirements.txt
```

### Path Configuration

Before running the code, you need to configure the model paths in `paths.py`. The file contains paths to the pretrained models:

```python
model_names = {
    "Gemma-2-2b-it": "/path/to/your/gemma-2-2b-it",
    "Qwen2.5-3B-Instruct": "/path/to/your/Qwen2.5-3B-Instruct",
}
```

You need to:
1. Download the pretrained models:
   - Gemma-2-2b-it: [Download from Hugging Face](https://huggingface.co/google/gemma-2-2b-it)
   - Qwen2.5-3B-Instruct: [Download from Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

2. Update the paths in `paths.py` to point to your local model directories:
   - Replace `/path/to/your/gemma-2-2b-it` with the actual path to your Gemma model
   - Replace `/path/to/your/Qwen2.5-3B-Instruct` with the actual path to your Qwen model

Make sure the paths are absolute paths and the directories contain the complete model files.

### Data Preparation

The project uses Amazon review data. To prepare the data:

1. Run the preprocessing script:
```bash
python preprocess.py \
    --category "CDs_and_Vinyl" \  # or other categories like "Musical_Instruments"
    --K 0 \                       # minimum number of interactions per user/item
    --st_year 2022 \
    --st_month 10 \
    --ed_year 2023 \
    --ed_month 10 \
    --window_size 20 \           # context window size
    --data_root_dir "data"       # output directory
```

The script will:
- Download the raw Amazon review data
- Filter users and items based on minimum interaction count (K)
- Process and clean the text data
- Split the data into train/validation/test sets
- Save the processed dataset in the specified directory


### Pretrained Model Preparations


## Selction


To selcet the data by curriculum learning strategy from easy to hard samples.








## 2. Training

To train the model, simply run:

```bash
bash launch_train.sh
```

The script uses the following default configuration:
- Uses 4 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3)
- Runs 3 processes for distributed training
- Uses DeepSpeed for optimization
- Default dataset: Musical_Instruments
- Default model: Gemma-2-2b-it

Key parameters in the script:
- `NUM_PROCESSES`: Number of processes for distributed training
- `MAIN_PROCESS_PORT`: Port for the main process
- `DATASET_CAT`: Dataset category (e.g., "Musical_Instruments", "CDs_and_Vinyl")
- `DATASET_DIR`: Path to the processed dataset
- `MODEL`: Base model to use ("gemma" or "qwen")

Training hyperparameters:
- `train_batch_size`: 4
- `eval_batch_size`: 32
- `max_new_tokens`: 640
- `warmup_steps`: 32
- `num_train_epochs`: 3
- `group_size`: 4

You can modify these parameters in `launch_train.sh` according to your needs.
