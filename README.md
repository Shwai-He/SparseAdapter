# SparseAdapter
[![Paper](https://img.shields.io/badge/arXiv-2210.04284-b31b1b.svg)](https://arxiv.org/abs/2210.04284)
[![EMNLP Findings](https://img.shields.io/badge/EMNLP%20Findings-2022-4b8bbe.svg)](https://aclanthology.org/2022.findings-emnlp.160/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.17.0-ffcc4d.svg)](https://github.com/huggingface/transformers)

Official implementation of the paper:

**SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters**  
Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao  
Findings of EMNLP 2022  
Paper: https://arxiv.org/abs/2210.04284

SparseAdapter revisits adapter parameter-efficiency via pruning. Under the same budget, sparse adapters can match or outperform dense adapters, and the proposed **Large-Sparse** setting further improves model capacity and performance.

<p align="center">
  <img src="Figures/SparseAdapter.png" width="1000" alt="SparseAdapter overview">
</p>

## Environment

- Python 3.8+
- torch==1.13.1
- transformers==4.17.0
- tokenizers==0.10.1
- nltk==3.5

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

You can run SparseAdapter training from the task-specific scripts under `examples/pytorch/`.

### 1) Text Classification (GLUE)

```bash
cd examples/pytorch/text-classification
bash run_glue.sh
```

Main entry:
- `examples/pytorch/text-classification/run_glue_sparse.py`

### 2) Question Answering (SQuAD)

```bash
cd examples/pytorch/question-answering
bash run_qa.sh
```

Main entry:
- `examples/pytorch/question-answering/run_qa_sparse.py`

### 3) Summarization (XSum/CNN-DM)

```bash
cd examples/pytorch/summarization
bash run_summarization.sh
```

Main entry:
- `examples/pytorch/summarization/run_summarization_sparse.py`

## Notes

- Default scripts are configured for multi-GPU usage. Adjust `device_ids` and batch size for your hardware.
- Key sparse-related flags include `--pruner`, `--sparsity`, `--attn_bn`, and `--ffn_bn`.
- Checkpoints and logs are written to `checkpoints/`.

## Citation

```bibtex
@inproceedings{he2022sparseadapter,
  title = {SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters},
  author = {He, Shwai and Ding, Liang and Dong, Daize and Zhang, Miao and Tao, Dacheng},
  booktitle = {Findings of EMNLP},
  year = {2022},
  url = {https://aclanthology.org/2022.findings-emnlp.160}
}
```
