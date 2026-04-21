import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadNpyImageFromFile(BaseTransform):
    def transform(self, results: dict) -> dict:
        image = np.load(results["img_path"]).astype(np.uint8)
        results["img"] = image
        results["img_shape"] = image.shape[:2]
        results["ori_shape"] = image.shape[:2]
        return results


@TRANSFORMS.register_module()
class LoadNpySegAnnotations(BaseTransform):
    def transform(self, results: dict) -> dict:
        seg_map = np.load(results["seg_map_path"]).astype(np.uint8)
        results["gt_seg_map"] = seg_map
        results.setdefault("seg_fields", []).append("gt_seg_map")
        return results
