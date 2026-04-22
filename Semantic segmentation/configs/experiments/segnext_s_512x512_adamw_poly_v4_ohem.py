_base_ = ["./segnext_s_512x512_adamw_poly_v1.py"]

# 覆寫 model 的 decode_head，加入 OHEM sampler
model = dict(
    decode_head=dict(
        sampler=dict(type="OHEMPixelSampler", thresh=0.7, min_kept=100000)
    )
)

work_dir = "./outputs/logs/exp_v4_ohem"
