# Encore: Enhancing Numerical Reasoning with the Guidance of Reliable Reasoning Processes [ACL2024]

This repository contains code for the ACL2024 paper ["Enhancing Numerical Reasoning with the Guidance of Reliable Reasoning Processes"](https://arxiv.org/abs/2402.10654).

If you use Encore in your work, please cite it as follows:
```
@article{wang2024enhancing,
  title={Enhancing Numerical Reasoning with the Guidance of Reliable Reasoning Processes},
  author={Wang, Dingzirui and Dou, Longxu and Zhang, Xuanliang and Zhu, Qingfu and Che, Wanxiang},
  journal={arXiv preprint arXiv:2402.10654},
  year={2024}
}
```

## Build Environment
```
conda create -n encore python=3.9
conda activate encore
pip install -r requirements.txt
```

## Pre-Process Data
Download and put each dataset in ./dataset, and run the preprocess.py in each dataset file.

## Retrieve
This step is to train the retrieval model to retrieve the question-related textual sentences and tabular columns.

### Pre-Process
Pre-process the retrieval training data with [retrieve/scripts/preprocess.bash](./retrieve/scripts/preprocess.bash).

### Train
Run the retrieval model training with [retrieve/scripts/train.bash](./retrieve/scripts/train.bash).

### Evaluate
Evaluate the retrieval model with [retrieve/scripts/evaluate.bash](./retrieve/scripts/evaluate.bash), which also generates the retrieval results.

## Pre-Train
This step is to build the pre-training data and use the data built to pre-train the model, enhancing the table understanding ability of the model.

### Download Model
Download the [fairseq BART-large](https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz) and put it in generate/checkpoint/BART-large.

### Build Pre-Training Data
Build the pre-training data with [generate/scripts/pretrain/preprocess.bash](./generate/scripts/pretrain/preprocess.bash).

### Pre-Train
Pre-train the model with [generate/scripts/pretrain/train.bash](./generate/scripts/pretrain/train.bash).

## Fine-Tune
This step is to train the model with or without pre-training to solve the numerical reasoning using textual and tabular evidence.

### Pre-Process
Pre-process the retrieval training data with [generate/scripts/finetune/preprocess.bash](./retrieve/scripts/preprocess.bash).

### Train
Run the retrieval model training with [rgenerate/scripts/finetune/train.bash](./retrieve/scripts/train.bash).

### Evaluate
Evaluate the retrieval model with [generate/scripts/finetune/evaluate.bash](./retrieve/scripts/evaluate.bash), which also generates the retrieval results.
