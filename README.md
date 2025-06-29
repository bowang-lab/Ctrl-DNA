# Ctrl-DNA: Controllable Cell-Type-Specific Regulatory DNA Design via Constrained RL

<p align="center">
  <img src="Ctrl-DNA.png" alt="Ctrl-DNA logo" width="500">
</p>

## Overview
We present Ctrl-DNA, a constrained reinforcement learning framework for the controllable design of cell-type-specific regulatory DNA sequences. Ctrl-DNA fine-tunes autoregressive genomic language models by framing sequence generation as a biologically informed constrained optimization problem. Using a value-model free, Lagrangian-guided policy optimization strategy, Ctrl-DNA iteratively refines sequences to maximize gene expression in a target cell type while suppressing activity in off-target cell types. Applied to human enhancer and promoter datasets, Ctrl-DNA generates biologically plausible, high-fitness sequences enriched for key transcription factor motifs, achieving state-of-the-art specificity and performance in regulatory sequence design.

![ctrl-DNA Architecture](./assets/fig1-1.png)

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/bowang-lab/Ctrl-DNA.git 
cd ctrl-dna
pip install -r requirements.txt
```

## Data Preprocessing

We follow the preprocessing pipeline from [Genentech/regLM](https://github.com/Genentech/regLM). Please refer to their repository for detailed instructions.

### Training

To train the model on the enhancer and promoter dataset using our method, run:

```bash
bash reinforce_lagrange_promoters.sh
bash reinforce_lagrange_enhancers.sh
```

### Acknowledgements

Our implementation builds upon several open-source projects:

- [regLM](https://github.com/Genentech/regLM): Provided the reward model architecture and data preprocessing pipeline.
- [TACO](https://github.com/yangzhao1230/TACO): Supplied the reinforcement learning framework that our method extends.

We sincerely thank the authors of these projects for making their code and datasets publicly available.

### Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{chen2025ctrldnacontrollablecelltypespecificregulatory,
      title={Ctrl-DNA: Controllable Cell-Type-Specific Regulatory DNA Design via Constrained RL}, 
      author={Xingyu Chen and Shihao Ma and Runsheng Lin and Jiecong Lin and Bo Wang},
      year={2025},
      eprint={2505.20578},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.20578}, 
}
```
