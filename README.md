# regConCtrl-DNA: Controllable Cell-Type-Specific Regulatory DNA Design via Constrained RL
We present ctrl-DNA, a policy-only RL framework for generating cell-type-specific regulatory DNA sequences with controllable off-target suppression.

ctrl-DNA fine-tunes DNA language models using a value-free, Lagrangian-constrained optimization approach. It optimizes expression in the target cell type while enforcing soft constraints on off-targets — enabling precise, biologically grounded sequence generation.
![ctrl-DNA Architecture](./assets/fig1.pdf)