"""Download a LeRobot dataset from HuggingFace Hub.

This script downloads popular LeRobot datasets for testing VIVE Labeler.

Usage:
    python scripts/download_lerobot_dataset.py --dataset pusht --output-dir ./datasets
"""

import argparse
from pathlib import Path


def download_dataset(dataset_name, output_dir, subset=None):
    """Download LeRobot dataset from HuggingFace Hub.

    Args:
        dataset_name: Name of the dataset (pusht, aloha, etc.)
        output_dir: Output directory
        subset: Optional subset specification
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface-hub")
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map common names to repo IDs
    dataset_repos = {
        "pusht": "lerobot/pusht",
        "aloha": "lerobot/aloha_sim_insertion_human",
        "xarm": "lerobot/xarm_lift_medium",
        "metaworld": "lerobot/metaworld_mt50",
        "toto": "lerobot/toto",
    }

    repo_id = dataset_repos.get(dataset_name.lower(), dataset_name)

    print(f"Downloading dataset: {repo_id}")
    print(f"Output directory: {output_dir}")
    print("\nThis may take a while depending on dataset size...")
    print("You can also use streaming mode in VIVE Labeler to avoid download.\n")

    try:
        dataset_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir / dataset_name,
            local_dir_use_symlinks=False,
        )

        print("\n" + "=" * 60)
        print("✅ Dataset downloaded successfully!")
        print("=" * 60)
        print(f"Location: {dataset_path}")
        print(f"\nTo use in VIVE Labeler:")
        print(f"  1. Launch: python -m vive_labeler")
        print(f"  2. File → Open Dataset")
        print(f"  3. Select: {dataset_path}")
        print("=" * 60)

        return dataset_path

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nAlternative: Use streaming mode in VIVE Labeler")
        print(f"Just enter the repo ID: {repo_id}")
        return None


def list_popular_datasets():
    """List popular LeRobot datasets."""
    datasets = {
        "pusht": {
            "repo": "lerobot/pusht",
            "description": "Push-T task - 2D pushing manipulation",
            "size": "~500MB",
        },
        "aloha": {
            "repo": "lerobot/aloha_sim_insertion_human",
            "description": "ALOHA bimanual insertion task",
            "size": "~2GB",
        },
        "xarm": {
            "repo": "lerobot/xarm_lift_medium",
            "description": "xArm lift task",
            "size": "~1GB",
        },
        "metaworld": {
            "repo": "lerobot/metaworld_mt50",
            "description": "MetaWorld multi-task benchmark",
            "size": "~5GB",
        },
        "toto": {
            "repo": "lerobot/toto",
            "description": "Toto mobile manipulation",
            "size": "~3GB",
        },
    }

    print("\nPopular LeRobot Datasets:")
    print("=" * 70)
    for name, info in datasets.items():
        print(f"\n{name}:")
        print(f"  Repo: {info['repo']}")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size']}")
    print("\n" + "=" * 70)
    print("\nNote: You can also browse all datasets at:")
    print("https://huggingface.co/lerobot")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Download LeRobot datasets from HuggingFace Hub"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name (pusht, aloha, xarm, metaworld, toto) or full repo ID"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="Output directory (default: ./datasets)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List popular datasets"
    )

    args = parser.parse_args()

    if args.list or not args.dataset:
        list_popular_datasets()
        if not args.dataset:
            return

    download_dataset(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()
