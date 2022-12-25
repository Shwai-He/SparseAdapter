# SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters
This is the official implementation of the [paper](https://arxiv.org/abs/2210.04284):

```
SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters
Shwai He, Liang Ding, Daize Dong, Miao Zhang, Dacheng Tao
EMNLP 2022 Findings. 
```

Adapter Tuning, which freezes the pretrained language models (PLMs) and only fine-tunes a few extra modules, becomes an appealing efficient alternative to the full model fine-tuning. Although computationally efficient, the recent Adapters often increase parameters (e.g. bottleneck dimension) for matching the performance of full model fine-tuning, which we argue goes against their original intention. In this work, we re-examine the parameter-efficiency of Adapters through the lens of network pruning (we name such plug-in concept as \texttt{SparseAdapter}) and find that SparseAdapter can achieve comparable or better performance than standard Adapters when the sparse ratio reaches up to 80\%. Based on our findings, we introduce an easy but effective setting ``\textit{Large-Sparse}'' to improve the model capacity of Adapters under the same parameter budget. Experiments on five competitive Adapters upon three advanced PLMs show that with proper sparse method (e.g. SNIP) and ratio (e.g. 40\%) SparseAdapter can consistently outperform their corresponding counterpart. Encouragingly, with the \textit{Large-Sparse} setting, we can obtain further appealing gains, even outperforming the full fine-tuning by a large margin.

## Requirements
- torch==1.13.1
- transformers==4.17.0
- tokenizers==0.10.1
- nltk==3.5

To install requirements, run `pip install -r requirements.txt`.

## Usage
To fine-tune the SparseAdapter model, run: \

`examples/pytorch/text-classification/run_glue_sparse.py`, \
`examples/pytorch/summarization/run_summarization_sparse.py`. \
`examples/pytorch/question-answering/run_qa_sparse.py`, \

You can also run the following scripts: \

`examples/pytorch/text-classification/run_glue.sh`, \
`examples/pytorch/summarization/run_summarization.sh`. \
`examples/pytorch/question-answering/run_qa.sh`, \

## Citation

```
@misc{https://doi.org/10.48550/arxiv.2210.04284,
  doi = {10.48550/ARXIV.2210.04284},
  url = {https://arxiv.org/abs/2210.04284},
  author = {He, Shwai and Ding, Liang and Dong, Daize and Zhang, Miao and Tao, Dacheng},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {SparseAdapter: An Easy Approach for Improving the Parameter-Efficiency of Adapters},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```



