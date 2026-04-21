from pathlib import Path

from mmengine.dataset import BaseDataset
from mmseg.registry import DATASETS

from .labels import ALL_CLASSES, PALETTE


@DATASETS.register_module()
class GA2SegDataset(BaseDataset):
    METAINFO = dict(classes=ALL_CLASSES, palette=PALETTE)

    def __init__(self, **kwargs):
        super().__init__(serialize_data=False, **kwargs)

    def load_data_list(self):
        data_list = []
        with open(self.ann_file, encoding="utf-8") as split_file:
            sample_ids = [line.strip() for line in split_file if line.strip()]

        for sample_id in sample_ids:
            img_root = Path(self.data_prefix["img_path"])
            seg_root = Path(self.data_prefix["seg_map_path"])
            data_info = dict(
                img_path=str(img_root / f"train_{sample_id}.npy"),
                seg_map_path=str(seg_root / f"train_{sample_id}.npy"),
                seg_fields=[],
            )
            data_list.append(data_info)
        return data_list
