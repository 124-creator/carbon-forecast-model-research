# Primary-source review and execution-order correction
Six actual web queries were performed (PCGrad; CAGrad/NashMTL; TAG; validation reweighting; time-series affinity; ETAP). Both initial training and identical repeat inadvertently preceded actual retrieval; neither is literature-first. Initial artifacts are archived under exploratory_pre_retrieval. Actual SEARCH_ACTUAL and PRIMARY_HTTP receipts were obtained afterward. No training result qualifies as a pristine literature-first preregistered run; this is an execution-order failure, explicitly retained. References below are verified only when covered by PRIMARY_HTTP or actual search receipts; ETAP/scalable affinity links remain unverified leads, not established evidence.

Primary sources retrieved:
- https://arxiv.org/abs/2001.06782 PCGrad: projects conflicting task gradients. Does not identify persistent generative task differences.
- https://arxiv.org/abs/2110.14048 CAGrad: conflict-averse optimization balancing average and worst task progress.
- https://arxiv.org/abs/2202.01017 NashMTL: bargaining formulation of gradient combination.
- https://proceedings.neurips.cc/paper/2021/file/e77910ebb93b511588557806310f78f1-Paper.pdf TAG: donor shared update and recipient lookahead loss already define asymmetric transfer affinity.
- https://proceedings.mlr.press/v80/ren18a.html Ren et al.: validation-gradient-driven training weights already exist. Candidate A=H_tail G_core^T is a first-order validation transfer estimator plus softmax weighting, not a new mathematical class.
- https://arxiv.org/abs/2101.09645 shared-attention time-series MTL already exists.
- https://proceedings.iclr.cc/paper_files/paper/2026/hash/b595b039d8f3e04649e45879f487e707-Abstract-Conference.html ETAP: gradient-affinity plus learned correction predicts task-group gains. Author PDF https://aronlaszka.com/papers/ayman2026ensemble.pdf.
- https://arxiv.org/abs/2409.06091 scalable gradient-based task affinity is another close route.

Contribution requirement: distinguish noisy/optimization-induced conflict from persistent true task difference using multiple historical temporal blocks, identifiable or falsifiable persistence assumptions, calibrated transfer prediction on withheld historical blocks, comparisons to same-validation-information weighting and TAG-like affinity, chronology random-tail control and no-common controls. Current single fixed tail cannot identify true task difference: low transfer can equally represent finite-sample noise or drift. Temporal splitting is an evaluation constraint, not novelty. No claim of exhaustiveness from six queries. Stop this candidate after the fixed confirmation grid; no threshold adjustment.
