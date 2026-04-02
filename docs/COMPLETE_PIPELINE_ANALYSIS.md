# Complete Pipeline Analysis (Consolidated)

## End-to-End Flow
1. Data ingestion from raw FLAME frames.
2. Preprocessing and augmentation into processed output folders.
3. Shared-backbone training (classification + segmentation).
4. Validation using mIoU, Fire IoU, and FP/FN analysis.
5. Latency and model-size benchmarking for edge readiness.
6. Export of official metrics and model checkpoints.

## Key Output Artifacts
- data/processed/Output/Classification/stage2_results_table.csv
- data/processed/Output/Classification/shared_backbone_metrics.json
- models/trained/dicta_shared_backbone/best_dicta_model.pth

## Current DICTA Readiness Snapshot
- mIoU: 0.8296 (target >= 0.80)
- Fire IoU: 0.6620
- Latency p95: ~10.05ms (target < 100ms)
- Model size: ~1.748MB (target < 2MB)

## Supporting Docs
- docs/SOTA_COMPARISON.md
- docs/DICTA_BENCHMARK_PROTOCOL.md
- docs/MODEL_ARCHITECTURE.md
- docs/EDGE_DEPLOYMENT.md
