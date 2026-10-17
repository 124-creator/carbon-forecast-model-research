# Pre-fit and publication implementation log

The first test run had19 passes and1 failure: an exact equality check on P.T=P differed by5.55e-17 under matrix multiplication. Before ANY market fit, the check was changed to absolute tolerance1e-14. Model/training/protocol unchanged. Original failure log retained in the work package. This is not an empirical model rescue.

During source publication, the first unreferenced audit blob used seed=seed instead of the executed seed=17 literal. Before publishing an executable tree or any remote execution it was replaced by blob924236b240f179e3830955f3f671186b2d84a624 matching local bytes. Learning code and local execution unchanged.
