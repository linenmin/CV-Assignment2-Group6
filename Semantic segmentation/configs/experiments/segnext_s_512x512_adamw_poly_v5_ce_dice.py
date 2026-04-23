_base_ = ["./segnext_s_512x512_adamw_poly_v1.py"]

# V5 keeps the V1 architecture and data pipeline, and only changes the
# decode-head objective from pure CE to CE + Dice.
model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
            dict(type="DiceLoss", use_sigmoid=False, activate=True, loss_weight=0.5, eps=1e-5),
        ]
    )
)

work_dir = "./outputs/logs/exp_v5_ce_dice"
