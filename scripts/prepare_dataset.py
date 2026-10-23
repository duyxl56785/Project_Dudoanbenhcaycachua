"""Extract preselected PlantVillage archive; never create a new random split.

Download tomato_6502_images.zip from the NV1 Google Drive folder first.
Run: python -m scripts.prepare_dataset --archive /path/to/tomato_6502_images.zip
"""
import argparse
from pathlib import Path

from src.data.dataset import prepare_archive
from scripts.check_dataset import check, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    args = parser.parse_args()
    prepare_archive(args.archive, args.data_root)
    summary = check(data_root=args.data_root)
    print(f"Extracted dataset: {args.data_root}")
    print(f"NV1 image-path verification: PASS ({sum(summary['split_counts'].values())} files)")


if __name__ == "__main__":
    main()
