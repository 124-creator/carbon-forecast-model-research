# Stage C: Carbon forecasting research

Read REPORT_ZH.md first. Candidate failed the predeclared continuation gate; this is not a successful new-model manuscript.

Contains a pinned author-reported GDEA snapshot, 81 configuration archives, a conditional horizon-response decoder and controls, 3 fixed-parameter ablations, 15 synthetic paths, and audit evidence. No original EU results reused. No official exchange-vintage claim.

## Replay

Use Python 3.13 and requirements.txt (CPU PyTorch build is sufficient).

```
python reproduce.py --destination ./stageC_replay
```

A fresh directory is required. To resume a reproduction use the same destination plus --resume. run_status.json records live progress; per-configuration checkpoints are validated before skipping. The delivered outputs are never overwritten by this command. Do not treat a short checkpoint pause as deadlock: inspect process activity and recent log/checkpoint updates.

Data notice: DATA_NOTICE.md. Source hash and failed direct download attempts: metadata/.
