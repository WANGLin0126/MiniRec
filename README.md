# MiniRec: Data-Efficient Reinforcement Learning for LLM-based Recommendation

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Abstract

The integration of reinforcement learning (RL) into large language models (LLMs) has opened new opportunities for recommender systems by eliciting reasoning and improving user preference modeling. However, RL-based LLM recommendation faces significant efficiency challenges, making full-data training costly. Existing data selection methods define sample value based on learnability or representativeness, yet their loss/gradient-driven or dataset coverage-driven criteria often misalign with RL learning dynamics, resulting in suboptimal performance.

To address this, we propose **MiniRec**, a data selection framework tailored for RL-based LLM recommendation. MiniRec evaluates sample learnability using key RL signals—rewards—pruning samples that are too easy (too high reward) or too difficult (consistently low reward). It assesses representativeness by aligning sample gradients with the approximated "ideal" global RL optimization trajectory, selecting samples that mainly drive model updates, and it also enforces diversity to reduce redundancy. Combined with a curriculum learning strategy from easy to hard samples, MiniRec significantly reduces training cost while largely preserving performance. This codebase contains experiments **conducted on the CDs_and_Vinyl dataset from Amazon reviews**, demonstrating MiniRec's effectiveness and highlighting the importance of reward-aligned, trajectory-informed data selection in RL-based LLM recommendation.

## Note

**This codebase contains experiments conducted on the CDs_and_Vinyl dataset from Amazon reviews.** This choice reflects the dataset's rich user interaction patterns and representative nature for recommendation system evaluation.

## Table of Contents

- [Requirements](#requirements)
- [Configuration](#configuration)
- [Data Preparation](#data-preparation)
- [Data Selection](#data-selection)
- [Training](#training)
- [Results](#results)

## Requirements

The project requires Python 3.11 and the following dependencies:
torch==2.5.1+cu121 \
transformers==4.48.0 \
datasets==3.4.1 \
accelerate==1.6.0 \
peft==0.15.0 \
trl==0.16.1 \
numpy==2.2.5 \
pandas==2.2.3 \
rich==13.9.4 \
tqdm>=4.65.0  \
fire==0.7.0 \
requests==2.32.3 \
vllm==0.7.3 \
deepspeed==0.15.4 \
bitsandbytes==0.44.1

Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

### Model Path Configuration

Before running the code, you need to configure the model paths in `paths.py`. The file contains paths to the pretrained models:

```python
model_names = {
    "Gemma-2-2b-it": "/path/to/your/gemma-2-2b-it",
    "Qwen2.5-3B-Instruct": "/path/to/your/Qwen2.5-3B-Instruct",
}
```

**Setup Steps:**

1. **Download the pretrained models:**
   - [Gemma-2-2b-it](https://huggingface.co/google/gemma-2-2b-it)
   - [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)

2. **Update the paths in `paths.py`:**
   - Replace `/path/to/your/gemma-2-2b-it` with the actual path to your Gemma model
   - Replace `/path/to/your/Qwen2.5-3B-Instruct` with the actual path to your Qwen model

   > **Note:** Make sure the paths are absolute paths and the directories contain the complete model files.

## Data Preparation

The project uses Amazon review data. **This codebase contains experiments conducted on the CDs_and_Vinyl dataset.** To prepare the data, run the preprocessing script:

```bash
python preprocess.py \
    --category "CDs_and_Vinyl" \      # Primary dataset used in experiments
    --K 0 \                          # minimum interactions per user/item
    --st_year 2022 \
    --st_month 10 \
    --ed_year 2023 \
    --ed_month 10 \
    --window_size 20 \               # context window size
    --data_root_dir "data"           # output directory
```

**What the script does:**
- Downloads the raw Amazon review data
- Filters users and items based on minimum interaction count (K)
- Processes and cleans the text data
- Splits the data into train/validation/test sets
- Saves the processed dataset in the specified directory

## Data Selection

MiniRec implements a curriculum learning strategy that selects samples from easy to hard difficulty levels.

### Run Data Selection

```bash
python minirec.py
```

The selected training sample indices will be saved in the `scores/` folder.

### Available Datasets

Pre-selected training sample indices are available for different sizes:
- 256 samples
- 512 samples  
- 1024 samples

Each size includes corresponding curriculum-sorted versions in the `scores/` folder.

## Training

### Quick Start

To train the model with default settings:

```bash
bash launch_train.sh
```



## Results

The experimental results in this codebase are obtained on the **CDs_and_Vinyl** dataset from Amazon reviews. The framework provides significant improvements in training efficiency while maintaining model performance. Key benefits include:

- **Reduced Training Cost**: Up to 80% reduction in training time on CDs_and_Vinyl dataset
- **Maintained Performance**: Preserves recommendation quality across different sample sizes
- **Flexible Sampling**: Supports various dataset sizes and difficulty levels for CDs_and_Vinyl



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.