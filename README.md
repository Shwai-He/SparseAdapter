<div align="center">

# SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters

[![EMNLP Findings 2022](https://img.shields.io/badge/EMNLP%20Findings-2022-4b8bbe.svg?style=for-the-badge&logo=acl&logoColor=white)](https://aclanthology.org/2022.findings-emnlp.160/)
[![arXiv](https://img.shields.io/badge/arXiv-2210.04284-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2210.04284)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-24292e.svg?style=for-the-badge&logo=github&logoColor=white)](https://shwai-he.github.io/SparseAdapter/)
[![PyTorch 1.13.1](https://img.shields.io/badge/PyTorch-1.13.1-ee4c2c.svg?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers 4.17.0](https://img.shields.io/badge/Transformers-4.17.0-ffcc4d.svg?style=for-the-badge&logo=huggingface&logoColor=white)](https://github.com/huggingface/transformers)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<p align="center">
  <b><a href="https://shwai-he.github.io/">Shwai He</a><sup>1</sup></b> •
  <b><a href="https://scholar.google.com/citations?user=G_m9a7kAAAAJ">Liang Ding</a><sup>2</sup></b> •
  <b><a href="https://scholar.google.com/citations?user=jTjU9QcAAAAJ">Daize Dong</a><sup>3</sup></b> •
  <b><a href="https://scholar.google.com/citations?user=Y4cQ8uUAAAAJ">Miao Zhang</a><sup>4,5</sup></b> •
  <b><a href="https://scholar.google.com/citations?user=zkVOe18AAAAJ">Dacheng Tao</a><sup>3,2</sup></b>
</p>

<p align="center">
  <sup>1</sup>University of Maryland, College Park &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>2</sup>JD Explore Academy &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>3</sup>The University of Sydney &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>4</sup>King's College London &nbsp;&nbsp;|&nbsp;&nbsp;
  <sup>5</sup>Harbin Institute of Technology
</p>

---

[📖 Overview](#-overview) •
[✨ Key Highlights](#-key-highlights) •
[🧠 Methodology](#-methodology) •
[📊 GLUE Benchmark Results](#-glue-benchmark-results) •
[⚙️ Installation](#%EF%B8%8F-installation) •
[🚀 Quick Start](#-quick-start) •
[🔍 Pruning Strategies](#-pruning-strategies) •
[📦 Repository Structure](#-repository-structure) •
[📄 Citation](#-citation)

---

</div>

## 📖 Overview

Adapter tuning has emerged as an effective and parameter-efficient fine-tuning (PEFT) paradigm for large pre-trained language models (PLMs), freezing the original weights and updating only lightweight bottleneck modules inserted into the transformer layers. However, strong performance often requires scaling adapter bottleneck dimensions, thereby inflating trainable parameters.

**SparseAdapter** systematically investigates **parameter redundancy within adapter modules**. We show that:
1. Dense adapters contain substantial parameter redundancy during downstream adaptation.
2. Incorporating sparsity via gradient/magnitude-based pruning retains representation quality while pruning up to **50%–80%** of adapter parameters.
3. **Large-Sparse Adapter (SparseAdapter-L)**: By scaling the adapter bottleneck size while simultaneously pruning a high ratio of weights, SparseAdapter achieves **superior accuracy over dense adapters and full fine-tuning** under identical or smaller parameter budgets.

<p align="center">
  <img src="Figures/SparseAdapter.png" width="95%" alt="SparseAdapter Overview Diagram">
</p>

---

## ✨ Key Highlights

- ⚡ **Extreme Parameter Efficiency**: Operates with only **0.1% – 0.5%** trainable parameters relative to full fine-tuning.
- 🎯 **Competitive & Superior Performance**: Outperforms standard Houlsby and Pfeiffer adapters across GLUE, SQuAD, and text summarization benchmarks.
- 📈 **Large-Sparse Paradigm**: Expands bottleneck width while pruning inactive connections, breaking the capacity bottleneck of tiny dense adapters.
- 🛠️ **Drop-in Flexibility**: Supports multiple pruning schedules (**Magnitude, SNIP, GraSP, SynFlow, Random**) with standard HuggingFace Transformers.

---

## 🧠 Methodology

### 1. Adapter Architecture & Redundancy
A standard adapter module consists of a down-projection matrix $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d \times m}$, a non-linear activation $\sigma(\cdot)$, and an up-projection matrix $\mathbf{W}_{\text{up}} \in \mathbb{R}^{m \times d}$, with a residual skip connection:

$$\mathbf{h}' = \mathbf{h} + \sigma(\mathbf{h} \mathbf{W}_{\text{down}}) \mathbf{W}_{\text{up}}$$

where $d$ is the hidden dimension and $m \ll d$ is the bottleneck dimension.

### 2. SparseAdapter Formulation
SparseAdapter applies a binary pruning mask $\mathbf{M} \in \{0, 1\}^{|\mathbf{W}|}$ to adapter parameters $\mathbf{W} \in \{\mathbf{W}_{\text{down}}, \mathbf{W}_{\text{up}}\}$:

$$\mathbf{W}_{\text{sparse}} = \mathbf{M} \odot \mathbf{W}$$

The mask $\mathbf{M}$ is computed using connection sensitivity scores $S(w)$ under a target sparsity level $s \in [0, 1)$:

$$\mathbf{M}_{ij} = \mathbb{I}\left(S(w_{ij}) \ge \text{Threshold}(S, 1-s)\right)$$

### 3. Supported Pruning Criteria
* **Magnitude (`mag`)**: Prunes weights with the smallest absolute values $|w_{ij}|$.
* **SNIP (`snip`)**: Single-shot network pruning based on connection sensitivity $S(w) = |g(w) \odot w|$.
* **GraSP (`grasp`)**: Gradient Signal Preservation, maximizing gradient flow through the network.
* **SynFlow (`synflow`)**: Data-agnostic synaptic flow conservation.
* **Random (`rand`)**: Uniform stochastic parameter mask baseline.

---

## 📊 GLUE Benchmark Results

Results evaluated on the **GLUE benchmark** using **RoBERTa-base** (125M backbone parameters).

| Model / PEFT Method | Trainable Params | Ratio (%) | MNLI (m/mm) | QQP | QNLI | SST-2 | CoLA | MRPC | STS-B | RTE | **Avg** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Fine-Tuning** | 125M | 100.0% | 87.6 / 87.2 | 91.9 | 92.8 | 94.8 | 63.6 | 90.2 | 91.2 | 78.7 | **86.35** |
| **Houlsby Adapter** (Dense, $m=64$) | 0.89M | 0.71% | 87.2 / 86.8 | 91.5 | 92.6 | 94.2 | 62.4 | 89.5 | 90.8 | 75.8 | **85.50** |
| **Pfeiffer Adapter** (Dense, $m=64$) | 0.45M | 0.36% | 87.0 / 86.5 | 91.4 | 92.4 | 94.0 | 61.8 | 89.0 | 90.5 | 74.4 | **85.06** |
| **SparseAdapter (Mag, 50% Sparse)** | **0.45M** | **0.36%** | 87.4 / 87.0 | 91.7 | 92.7 | 94.5 | 63.1 | 90.0 | 91.0 | 77.3 | **85.96** |
| **SparseAdapter (SNIP, 50% Sparse)** | **0.45M** | **0.36%** | 87.5 / 87.1 | 91.8 | 92.7 | 94.6 | 63.3 | 90.1 | 91.1 | 77.5 | **86.08** |
| **SparseAdapter-L (Large-Sparse, $m=128$, 75% Sparse)** | **0.45M** | **0.36%** | **87.8 / 87.4** | **91.9** | **93.0** | **94.9** | **63.8** | **90.7** | **91.4** | **79.1** | **86.58** |

> **Takeaway**: SparseAdapter-L ($m=128$, 75% sparsity) achieves an average GLUE score of **86.58**, surpassing both dense adapters (+1.08 over Houlsby, +1.52 over Pfeiffer) and full fine-tuning (+0.23), while updating only **0.36%** of parameters.

---

## ⚙️ Installation

### Environment Requirements
- Python >= 3.8
- PyTorch == 1.13.1
- Transformers == 4.17.0
- Tokenizers == 0.10.1
- Datasets, NLTK, Scipy, Scikit-learn

### Setup with Conda
```bash
# 1. Create and activate virtual environment
conda create -n sparseadapter python=3.8 -y
conda activate sparseadapter

# 2. Clone repository
git clone https://github.com/shwai-he/SparseAdapter.git
cd SparseAdapter

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. GLUE Benchmark Fine-Tuning

Run SparseAdapter on any GLUE task (e.g., MRPC, SST-2, MNLI, QNLI):

```bash
cd examples/pytorch/text-classification

# Run GLUE training with SparseAdapter
bash run_glue.sh
```

Or invoke the Python runner directly:

```bash
python examples/pytorch/text-classification/run_glue_sparse.py \
  --model_name_or_path roberta-base \
  --task_name mrpc \
  --do_train --do_eval \
  --per_device_train_batch_size 48 \
  --per_device_eval_batch_size 48 \
  --learning_rate 1e-4 \
  --num_train_epochs 10 \
  --attn_mode adapter \
  --attn_bn 64 \
  --ffn_mode adapter \
  --ffn_bn 64 \
  --pruner snip \
  --sparsity 0.5 \
  --output_dir ./checkpoints/roberta-base/mrpc/snip_0.5
```

### 2. Question Answering (SQuAD v1.1 & v2.0)

```bash
cd examples/pytorch/question-answering
bash run_qa.sh
```

### 3. Abstractive Summarization (XSum & CNN/DailyMail)

```bash
cd examples/pytorch/summarization
bash run_summarization.sh
```

---

## 🔍 Pruning Strategies & Configuration Knobs

SparseAdapter provides full control over pruning algorithms and adapter configurations:

| Argument | Options | Description |
| :--- | :--- | :--- |
| `--pruner` | `snip`, `mag`, `grasp`, `synflow`, `rand` | Criterion used to calculate parameter importance scores. |
| `--sparsity` | Float between `0.0` and `1.0` (e.g. `0.5`, `0.75`) | Fraction of adapter parameters to prune to zero. |
| `--attn_mode` | `adapter`, `none` | Adapter insertion into self-attention block. |
| `--ffn_mode` | `adapter`, `none` | Adapter insertion into feed-forward network (FFN) block. |
| `--attn_bn` | Integer (e.g. `32`, `64`, `128`, `256`) | Bottleneck dimension $m$ for attention adapter. |
| `--ffn_bn` | Integer (e.g. `32`, `64`, `128`, `256`) | Bottleneck dimension $m$ for FFN adapter. |
| `--unfreeze_params` | `ef_`, `adapter` | Parameter name prefix to unfreeze for backpropagation. |

---

## 📦 Repository Structure

```plaintext
SparseAdapter/
├── Figures/
│   └── SparseAdapter.png           # Architecture and methodology overview diagram
├── examples/
│   └── pytorch/
│       ├── text-classification/    # GLUE benchmark fine-tuning scripts
│       ├── question-answering/     # SQuAD v1.1 and v2.0 QA runners
│       └── summarization/          # XSum and CNN/DM summarization runners
├── petl/                           # Parameter-Efficient Tuning Layer definitions
│   ├── options.py                  # PETL and pruning command-line flags
│   └── ...                         # Bottleneck & adapter modules
├── src/
│   └── transformers/
│       ├── pruning/                # Pruning algorithms (SNIP, GraSP, Mag, SynFlow)
│       └── trainer_sparse.py       # Sparse training & evaluation engine
├── docs/                           # Interactive GitHub Pages documentation
│   ├── index.html                  # Interactive Sparsity Calculator & Benchmark visualizer
│   └── Figures/                    # Documentation visual assets
├── requirements.txt                # Python environment specifications
└── README.md                       # Project documentation
```

---

## 📄 Citation

If you find SparseAdapter helpful in your research or applications, please cite our paper:

```bibtex
@inproceedings{he2022sparseadapter,
  title     = {SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters},
  author    = {He, Shwai and Ding, Liang and Dong, Daize and Zhang, Miao and Tao, Dacheng},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2022},
  month     = dec,
  year      = {2022},
  address   = {Abu Dhabi, United Arab Emirates},
  publisher = {Association for Computational Linguistics},
  pages     = {2184--2190},
  url       = {https://aclanthology.org/2022.findings-emnlp.160/},
  doi       = {10.18653/v1/2022.findings-emnlp.160}
}
```

---

<div align="center">
  <sub>Developed with ❤️ by the SparseAdapter Research Team.</sub>
</div>
