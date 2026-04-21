from ga2_seg.split import build_train_val_split


def test_build_train_val_split_is_deterministic_and_complete():
    train_ids_a, val_ids_a = build_train_val_split(seed=42, val_ratio=0.15)
    train_ids_b, val_ids_b = build_train_val_split(seed=42, val_ratio=0.15)

    assert train_ids_a == train_ids_b
    assert val_ids_a == val_ids_b
    assert len(train_ids_a) + len(val_ids_a) == 749
    assert set(train_ids_a).isdisjoint(set(val_ids_a))
    assert len(val_ids_a) == 112
