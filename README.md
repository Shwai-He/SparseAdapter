<h1 align="center">[EMNLP Findings 2022] SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2210.04284"><img src="https://img.shields.io/badge/arXiv-2210.04284-b31b1b.svg" alt="arXiv"></a>
  <a href="https://aclanthology.org/2022.findings-emnlp.160/"><img src="https://img.shields.io/badge/EMNLP%20Findings-2022-4b8bbe.svg" alt="EMNLP Findings 2022"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-1.13.1-ee4c2c.svg" alt="PyTorch 1.13.1"></a>
  <a href="https://github.com/huggingface/transformers"><img src="https://img.shields.io/badge/Transformers-4.17.0-ffcc4d.svg" alt="Transformers 4.17.0"></a>
</p>

<p align="center">
  <a href="https://shwai-he.github.io/">Shwai He</a>, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao
</p>

<p align="center">
  <a href="#-introduction">📖 Introduction</a> •
  <a href="#-news">📰 News</a> •
  <a href="#-why-sparseadapter">✨ Why</a> •
  <a href="#-core-setting">🔍 Setting</a> •
  <a href="#-quick-start">🚀 Quick Start</a> •
  <a href="#%EF%B8%8F-installation">⚙️ Installation</a> •
  <a href="#-citation">📄 Citation</a>
</p>

## 📖 Introduction

This is the official implementation of the paper
[**SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters**](https://arxiv.org/abs/2210.04284),
published in **Findings of EMNLP 2022**.

<p align="center">
  <img src="Figures/SparseAdapter.png" width="1000" alt="SparseAdapter overview">
</p>

## 📰 News
- Oct 2022: SparseAdapter accepted by **Findings of EMNLP 2022**.
- Oct 2022: Paper and official implementation released.

## ✨ Why SparseAdapter
Adapter tuning is efficient because it freezes most PLM parameters, but strong adapter performance often requires larger adapter modules.
SparseAdapter revisits this tradeoff with pruning:

- Keep adapter-style tuning workflow.
- Introduce sparsity into adapter parameters.
- Improve parameter-efficiency while staying competitive with dense adapters.

The paper also introduces **Large-Sparse**, which increases adapter capacity under the same parameter budget by combining larger adapters with sparsity.

## 🔍 Core Setting
SparseAdapter training in this repo is mainly controlled by:

- `--pruner`: pruning strategy (e.g., `rand`, `snip`).
- `--sparsity`: sparsity ratio.
- `--attn_bn`, `--ffn_bn`: adapter bottleneck sizes.
- `--attn_mode`, `--ffn_mode`: adapter insertion modes.

These options are exposed in task scripts under `examples/pytorch/`.

## 📦 Repository Structure
- `src/`: modified Transformers source code and sparse trainer logic.
- `petl/`: adapter-related modules and utilities.
- `examples/pytorch/`: runnable scripts for GLUE, SQuAD, and summarization.
- `Figures/`: figures used in the paper and README.
- `utils/`: project utility scripts.

## ⚙️ Installation

- Python 3.8+
- torch==1.13.1
- transformers==4.17.0
- tokenizers==0.10.1
- nltk==3.5

```bash
conda create -n sparseadapter python=3.8 -y
conda activate sparseadapter

pip install -r requirements.txt
```

## 🚀 Quick Start
### 1) Text Classification (GLUE)

```bash
cd examples/pytorch/text-classification
bash run_glue.sh
```

Main script: `examples/pytorch/text-classification/run_glue_sparse.py`

### 2) Question Answering (SQuAD)

```bash
cd examples/pytorch/question-answering
bash run_qa.sh
```

Main script: `examples/pytorch/question-answering/run_qa_sparse.py`

### 3) Summarization (XSum/CNN-DM)

```bash
cd examples/pytorch/summarization
bash run_summarization.sh
```

Main script: `examples/pytorch/summarization/run_summarization_sparse.py`

## 🧪 Repro Tips
- Default `.sh` scripts are configured for multi-GPU runs (`device_ids="0 1 2 3 4 5 6 7"`). Adjust to your hardware.
- Outputs are written under `checkpoints/`.
- Start from each task script first, then tune pruning and sparsity knobs for your setup.

## 📄 Citation

```bibtex
@inproceedings{he2022sparseadapter,
  title = {SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters},
  author = {He, Shwai and Ding, Liang and Dong, Daize and Zhang, Miao and Tao, Dacheng},
  booktitle = {Findings of EMNLP},
  year = {2022},
  url = {https://aclanthology.org/2022.findings-emnlp.160}
}
```
