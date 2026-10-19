"""
prepare_dataset.py
------------------
This script solves all dataset problems automatically and creates a clean, balanced dataset ready for YOLOv26.
"""

import os
import re
import zipfile
import shutil
import random
import yaml
from glob import glob
from collections import defaultdict

# Set fixed seed so anyone running this script gets the exact same train/val/test split
random.seed(42)

ZIP_DIR = "/workspace/total_dataset"
TEMP_DIR = "/workspace/temp_extract"
OUTPUT_DIR = "/workspace/curated_dataset"

# Our 5 target classes in order (ID 0 to 4)
TARGET_CLASSES = ['daisy', 'sunflower', 'rose', 'tulip', 'dandelion']


def get_target_id(class_name, zip_name):
    """Maps different class names, synonyms, and languages into our 5 standard class IDs."""
    name = str(class_name).lower().strip()
    z_name = str(zip_name).lower()

    if 'daisy' in name or 'papatya' in name or 'chamomile' in name:
        return 0
    if 'sunflower' in name or 'aycicegi' in name or 'helianthus' in name:
        return 1
    if 'rose' in name or 'gul' in name or 'rosa' in name:
        return 2
    if 'tulip' in name or 'lale' in name:
        return 3
    if 'dandelion' in name or 'karahindiba' in name or 'taraxacum' in name:
        return 4

    if 'daisy' in z_name or 'chamomile' in z_name:
        return 0
    if 'sunflower' in z_name:
        return 1
    if 'rose' in z_name:
        return 2
    if 'tulip' in z_name:
        return 3
    if 'dandelion' in z_name:
        return 4

    return None


def get_canonical_zip_name(filename):
    """Removes 'Copy of ' prefixes to detect and skip duplicate ZIP files."""
    clean = re.sub(r'^(copy of\s+)+', '', filename, flags=re.IGNORECASE)
    clean = re.sub(r'\s*\(\d+\)(?=\.zip$)', '', clean, flags=re.IGNORECASE)
    return clean.strip().lower()


def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split in ['train', 'valid', 'test']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    # Step 1: Filter out duplicate ZIP archives
    all_zip_files = sorted(glob(f"{ZIP_DIR}/*.zip"))
    unique_zips = []
    seen_canonical = set()

    for z_path in all_zip_files:
        canon = get_canonical_zip_name(os.path.basename(z_path))
        if canon not in seen_canonical:
            seen_canonical.add(canon)
            unique_zips.append(z_path)

    print(f"Found {len(all_zip_files)} total ZIPs. Processing {len(unique_zips)} unique archives...")

    samples_by_class = defaultdict(list)

    # Step 2: Extract each unique ZIP and standardize labels
    for idx, z_path in enumerate(unique_zips, 1):
        zip_name = os.path.basename(z_path)
        canon_name = get_canonical_zip_name(zip_name)
        print(f"[{idx}/{len(unique_zips)}] Reading: {zip_name}")

        extract_sub = f"{TEMP_DIR}/z{idx:02d}"
        os.makedirs(extract_sub, exist_ok=True)
        with zipfile.ZipFile(z_path, 'r') as zf:
            zf.extractall(extract_sub)

        yaml_files = glob(f"{extract_sub}/**/data.yaml", recursive=True)
        if not yaml_files:
            continue

        with open(yaml_files[0], 'r', encoding='utf-8') as f:
            data_cfg = yaml.safe_load(f)
        names = data_cfg.get('names', [])
        if isinstance(names, dict):
            names = [names[k] for k in sorted(names.keys())]

        id_map = {}
        for old_id, cname in enumerate(names):
            new_id = get_target_id(cname, zip_name)
            if new_id is not None:
                id_map[old_id] = new_id

        if not id_map:
            fallback_id = get_target_id("", zip_name)
            if fallback_id is not None:
                for old_id in range(len(names) + 100):
                    id_map[old_id] = fallback_id

        yaml_root = os.path.dirname(yaml_files[0])
        for split in ['train', 'valid', 'val', 'test']:
            img_dir = f"{yaml_root}/{split}/images"
            lbl_dir = f"{yaml_root}/{split}/labels"
            if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
                continue

            for img_path in glob(f"{img_dir}/*.*"):
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    continue
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                lbl_path = f"{lbl_dir}/{base_name}.txt"
                if not os.path.exists(lbl_path):
                    continue

                clean_lines = []
                cls_counts = defaultdict(int)

                with open(lbl_path, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        old_cls = int(float(parts[0]))
                        if old_cls not in id_map:
                            continue
                        new_cls = id_map[old_cls]
                        coords = [float(x) for x in parts[1:]]

                        # Convert polygon annotations to standard YOLO bounding boxes
                        if len(coords) > 4:
                            xs, ys = coords[0::2], coords[1::2]
                            x_min, x_max = max(0.0, min(xs)), min(1.0, max(xs))
                            y_min, y_max = max(0.0, min(ys)), min(1.0, max(ys))
                            w, h = x_max - x_min, y_max - y_min
                            x_c, y_c = x_min + w / 2.0, y_min + h / 2.0
                        else:
                            x_c, y_c, w, h = coords[0], coords[1], coords[2], coords[3]

                        # Filter out tiny background noise annotations
                        if w * h >= 0.0025 and w > 0.03 and h > 0.03:
                            clean_lines.append(f"{new_cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                            cls_counts[new_cls] += 1

                # Keep clean images with 1 to 15 clear flower instances
                if 1 <= len(clean_lines) <= 15:
                    primary_cls = max(cls_counts, key=cls_counts.get)
                    prefix = f"z{idx:02d}_" + re.sub(r'[^a-z0-9]', '', canon_name)[:8]
                    samples_by_class[primary_cls].append((img_path, ext, f"{prefix}_{base_name}", clean_lines))

    # Step 3: Balance all 5 classes (2,000 images per class -> 10,000 total images)
    TARGET_PER_CLASS = 2000
    split_totals = {'train': 0, 'valid': 0, 'test': 0}

    print("\nBalancing classes and creating 80% Train / 10% Valid / 10% Test splits...")
    for cid in range(5):
        pool = samples_by_class[cid]
        random.shuffle(pool)
        if len(pool) >= TARGET_PER_CLASS:
            selected = pool[:TARGET_PER_CLASS]
        else:
            selected = pool.copy()
            while len(selected) < TARGET_PER_CLASS:
                selected.extend(pool[:min(len(pool), TARGET_PER_CLASS - len(selected))])

        n_total = len(selected)
        n_train = int(n_total * 0.80)
        n_val = int(n_total * 0.10)

        splits_data = {
            'train': selected[:n_train],
            'valid': selected[n_train:n_train + n_val],
            'test': selected[n_train + n_val:]
        }

        for sp_name, items in splits_data.items():
            for i_idx, (src_img, ext, base_id, lines) in enumerate(items):
                dst_name = f"c{cid}_{i_idx:04d}_{base_id}"
                shutil.copy2(src_img, f"{OUTPUT_DIR}/{sp_name}/images/{dst_name}{ext}")
                with open(f"{OUTPUT_DIR}/{sp_name}/labels/{dst_name}.txt", 'w') as out_f:
                    out_f.writelines(lines)
                split_totals[sp_name] += 1

    shutil.rmtree(TEMP_DIR, ignore_errors=True)

    # Step 4: Save the final data.yaml file
    master_yaml = {
        'train': f"{OUTPUT_DIR}/train/images",
        'val': f"{OUTPUT_DIR}/valid/images",
        'test': f"{OUTPUT_DIR}/test/images",
        'nc': 5,
        'names': TARGET_CLASSES
    }
    with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f:
        yaml.dump(master_yaml, f, sort_keys=False)

    print("=" * 65)
    print(f"Dataset preparation complete! Saved to: {OUTPUT_DIR}")
    print(f"Total Images: {sum(split_totals.values())} (Train: {split_totals['train']}, Valid: {split_totals['valid']}, Test: {split_totals['test']})")
    print("=" * 65)


if __name__ == "__main__":
    main()