# EoMT Training Curve Summary

- Log: `outputs\checkpoints\eomt_dinov3_v12_continue_unfreeze2_lr1e5\training_log.csv`
- Best validation mIoU: `0.8002` at step `2000`
- Final validation mIoU: `0.7954` at step `3000`
- Final loss: `14.1738`
- Figure: `outputs\analysis\eomt_dinov3_v12_continue_unfreeze2_lr1e5\training_curve.png`

## Interpretation

- Validation mIoU peaked before the final checkpoint, so the next run should use stronger early stopping or a lower learning rate.
- Loss is noisy because each logged point is a single minibatch loss, not an epoch average.
