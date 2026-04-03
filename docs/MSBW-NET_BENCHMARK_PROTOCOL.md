# MSBW-Net Benchmark Protocol

## Scope
This protocol defines how MSBW-Net shared-backbone models are benchmarked for accuracy, latency, and edge readiness.

## Required Metrics
- Segmentation mIoU
- Fire IoU
- Classification precision, recall, f1, accuracy
- Latency p50, p95, p99 (ms)
- Model size (MB)

## Targets
- mIoU >= 0.80
- Latency p95 < 100ms
- Model size < 2MB

## Dataset Split
- Source: FLAME paired image-mask data
- Train/Validation: 80/20
- Shuffle enabled with fixed seed when final reporting

## Evaluation Steps
1. Train model using staged LR schedule.
2. Load best checkpoint by validation mIoU.
3. Evaluate segmentation on validation set.
4. Evaluate confusion matrix for classification and derive precision/recall/f1.
5. Run latency benchmark with warmup and repeated runs.
6. Export metrics to data/processed/Output/Classification/shared_backbone_metrics.json.

## Reporting
- Keep a tabular comparison in docs/SOTA_COMPARISON.md.
- For notebook reporting, include a SOTA comparison section with MSBW-Net row + references.
