from collections import Counter

from ga2_seg.rare_focus import RARE_FOCUS_CLASS_NAMES
from ga2_seg.split import build_rare_focus_train_split, build_train_val_split
from ga2_seg.stats import load_train_labels_df


def test_build_train_val_split_is_deterministic_and_complete():
    train_ids_a, val_ids_a = build_train_val_split(seed=42, val_ratio=0.15)
    train_ids_b, val_ids_b = build_train_val_split(seed=42, val_ratio=0.15)

    assert train_ids_a == train_ids_b
    assert val_ids_a == val_ids_b
    assert len(train_ids_a) + len(val_ids_a) == 749
    assert set(train_ids_a).isdisjoint(set(val_ids_a))
    assert len(val_ids_a) == 112


def test_build_rare_focus_train_split_duplicates_target_samples_only():
    base_train_ids, _ = build_train_val_split(seed=42, val_ratio=0.15)
    rare_focus_ids = build_rare_focus_train_split(base_train_ids=base_train_ids, duplicate_factor=2)
    counts = Counter(rare_focus_ids)
    train_df = load_train_labels_df()
    rare_mask = train_df.loc[base_train_ids, list(RARE_FOCUS_CLASS_NAMES)].sum(axis=1) > 0

    assert len(rare_focus_ids) > len(base_train_ids)
    assert counts[base_train_ids[0]] in (1, 2)

    duplicated_ids = {sample_id for sample_id, count in counts.items() if count == 2}
    expected_ids = set(train_df.loc[base_train_ids].index[rare_mask].astype(int))
    assert duplicated_ids == expected_ids
