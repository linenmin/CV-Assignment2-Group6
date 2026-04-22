import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
# 加上與專案一致的相容性設定
import sys
import os

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def get_dataset_root() -> Path:
    return get_project_root().parent / "kul-computer-vision-ga-2-2026"

def calculate_md5(file_path: Path) -> str:
    """計算檔案的 MD5 Hash 值"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    dataset_root = get_dataset_root()
    
    if not dataset_root.exists():
        print(f"錯誤：找不到資料集路徑 {dataset_root}")
        return

    print(f"開始檢查資料集: {dataset_root}")
    print("-" * 50)

    corrupted_files = []
    hash_dict = defaultdict(list)
    total_images = 0

    # 檢查 train 和 test 資料夾
    for split in ["train", "test"]:
        split_dir = dataset_root / split
        if not split_dir.exists():
            print(f"警告：找不到資料夾 {split_dir}")
            continue
            
        print(f"正在掃描 {split} 資料夾...")
        # 尋找所有圖片檔案 (*.npy)
        for img_path in split_dir.rglob("img/*.npy"):
            total_images += 1
                
            # 1. 檢查檔案是否損毀
            try:
                data = np.load(img_path)
                # 基本檢查：確保有讀出東西
                if data.size == 0:
                    raise ValueError("Numpy array is empty.")
            except Exception as e:
                corrupted_files.append((img_path, str(e)))
                continue
            
            # 2. 紀錄 Hash 值以檢查重複
            file_hash = calculate_md5(img_path)
            hash_dict[file_hash].append(img_path)

    print("-" * 50)
    print("檢查報告")
    print("-" * 50)
    print(f"總共掃描了 {total_images} 張圖片。")

    # 報告損毀檔案
    if corrupted_files:
        print(f"\n[!] 發現 {len(corrupted_files)} 個損毀或無法讀取的圖片：")
        for path, error in corrupted_files:
            print(f"  - {path.relative_to(dataset_root)} (錯誤: {error})")
    else:
        print("\n[OK] 沒有發現損毀或無法讀取的圖片。")

    # 報告重複檔案
    duplicates = {h: paths for h, paths in hash_dict.items() if len(paths) > 1}
    if duplicates:
        print(f"\n[!] 發現 {len(duplicates)} 組重複的圖片：")
        for h, paths in duplicates.items():
            print(f"  - Hash: {h}")
            for p in paths:
                print(f"    -> {p.relative_to(dataset_root)}")
    else:
        print("\n[OK] 沒有發現完全重複的圖片。")

if __name__ == "__main__":
    main()
