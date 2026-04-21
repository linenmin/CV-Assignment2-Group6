import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.registry import TRANSFORMS
from mmseg.datasets.transforms import RandomCrop


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


@TRANSFORMS.register_module()
class RareClassFocusedCrop(RandomCrop):
    def __init__(
        self,
        crop_size,
        rare_class_ids: list[int] | tuple[int, ...],
        rare_prob: float = 0.5,
        cat_max_ratio: float = 1.0,
        ignore_index: int = 255,
        max_attempts: int = 10,
    ):
        super().__init__(crop_size=crop_size, cat_max_ratio=cat_max_ratio, ignore_index=ignore_index)
        self.rare_class_ids = tuple(int(class_id) for class_id in rare_class_ids)
        self.rare_prob = float(rare_prob)
        self.max_attempts = int(max_attempts)

    def _generate_random_bbox(self, img: np.ndarray) -> tuple[int, int, int, int]:
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        return offset_h, offset_h + self.crop_size[0], offset_w, offset_w + self.crop_size[1]

    def _generate_centered_bbox(self, seg_map: np.ndarray, img: np.ndarray) -> tuple[int, int, int, int] | None:
        present_class_ids = [class_id for class_id in self.rare_class_ids if np.any(seg_map == class_id)]
        if not present_class_ids:
            return None

        chosen_class_id = int(np.random.choice(present_class_ids))
        ys, xs = np.where(seg_map == chosen_class_id)
        if len(ys) == 0:
            return None

        pixel_index = np.random.randint(0, len(ys))
        center_y = int(ys[pixel_index])
        center_x = int(xs[pixel_index])

        crop_h, crop_w = self.crop_size
        max_top = max(img.shape[0] - crop_h, 0)
        max_left = max(img.shape[1] - crop_w, 0)
        top = int(np.clip(center_y - crop_h // 2, 0, max_top))
        left = int(np.clip(center_x - crop_w // 2, 0, max_left))
        return top, top + crop_h, left, left + crop_w

    def _is_valid_bbox(self, seg_map: np.ndarray, crop_bbox: tuple[int, int, int, int]) -> bool:
        if self.cat_max_ratio >= 1.0:
            return True

        seg_crop = self.crop(seg_map, crop_bbox)
        labels, counts = np.unique(seg_crop, return_counts=True)
        counts = counts[labels != self.ignore_index]
        return len(counts) <= 1 or float(np.max(counts) / np.sum(counts)) < self.cat_max_ratio

    @cache_randomness
    def crop_bbox(self, results: dict) -> tuple[int, int, int, int]:
        img = results["img"]
        seg_map = results["gt_seg_map"]

        if np.random.rand() < self.rare_prob:
            for _ in range(self.max_attempts):
                crop_bbox = self._generate_centered_bbox(seg_map=seg_map, img=img)
                if crop_bbox is None:
                    break
                if self._is_valid_bbox(seg_map, crop_bbox):
                    return crop_bbox

        crop_bbox = self._generate_random_bbox(img)
        if self.cat_max_ratio < 1.0:
            for _ in range(self.max_attempts):
                if self._is_valid_bbox(seg_map, crop_bbox):
                    break
                crop_bbox = self._generate_random_bbox(img)
        return crop_bbox

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(crop_size={self.crop_size}, "
            f"rare_class_ids={self.rare_class_ids}, rare_prob={self.rare_prob})"
        )
