# EoMT Training Curve Summary

- Log: `outputs\checkpoints\eomt_dinov3_v11_unfreeze2\training_log.csv`
- Best validation mIoU: `0.7926` at step `3000`
- Final validation mIoU: `0.7926` at step `3000`
- Final loss: `14.8151`
- Figure: `outputs\analysis\eomt_dinov3_v11_unfreeze2\training_curve.png`

## Interpretation

- Validation mIoU is still highest at the final checkpoint, so this run likely has remaining training headroom.
- Loss is noisy because each logged point is a single minibatch loss, not an epoch average.
