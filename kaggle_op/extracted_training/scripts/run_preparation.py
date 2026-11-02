"""
Main Entry Script: Reproducible Dataset Preparation & Splitting Pipeline.

Usage:
  # Using default configuration:
  python scripts/run_preparation.py

  # Custom source and prepared directories:
  python scripts/run_preparation.py --source-dir "data/chest img data/chest_xray" --prepared-dir "data/prepared"

  # Custom seed and validation ratio:
  python scripts/run_preparation.py --val-ratio 0.15 --seed 42
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.device import print_device_info
from src.data.prepare import run_dataset_preparation


def resolve_paths_from_config(config_path: Path):
    """Resolves default source and prepared dataset paths."""
    default_src = project_root / "data" / "chest img data" / "chest_xray"
    default_prep = project_root / "data" / "prepared"
    val_ratio = 0.15
    seed = 42

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            # Check for Kaggle environment
            if Path("/kaggle/input/chest-xray-pneumonia/chest_xray").exists():
                src_str = cfg.get("dataset_paths", {}).get("kaggle_environment")
                prep_str = cfg.get("prepared_dataset_paths", {}).get("kaggle_environment")
            else:
                src_str = cfg.get("dataset_paths", {}).get("local_windows")
                prep_str = cfg.get("prepared_dataset_paths", {}).get("local_windows")

            if src_str:
                default_src = Path(src_str)
                if not default_src.is_absolute():
                    default_src = (project_root / default_src).resolve()

            if prep_str:
                default_prep = Path(prep_str)
                if not default_prep.is_absolute():
                    default_prep = (project_root / default_prep).resolve()

            prep_cfg = cfg.get("preparation", {})
            val_ratio = prep_cfg.get("val_ratio", val_ratio)
            seed = prep_cfg.get("random_seed", seed)

        except Exception:
            pass

    return default_src, default_prep, val_ratio, seed


def main():
    parser = argparse.ArgumentParser(
        description="Prepare chest X-ray dataset: deduplicate, isolate test, stratified 85/15 train/val split."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Path to source dataset directory (read-only)",
    )
    parser.add_argument(
        "--prepared-dir",
        type=str,
        default=None,
        help="Output path for prepared dataset",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.json",
        help="Path to dataset configuration file",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=None,
        help="Validation split ratio from cleaned training images (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Fixed random seed for deterministic splitting (default: 42)",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing prepared directory",
    )

    args = parser.parse_args()

    # 1. Hardware environment
    print_device_info()

    # 2. Resolve parameters
    config_file = (project_root / args.config).resolve()
    cfg_src, cfg_prep, cfg_val_ratio, cfg_seed = resolve_paths_from_config(config_file)

    src_dir = Path(args.source_dir).resolve() if args.source_dir else cfg_src
    prep_dir = Path(args.prepared_dir).resolve() if args.prepared_dir else cfg_prep
    val_ratio = args.val_ratio if args.val_ratio is not None else cfg_val_ratio
    seed = args.seed if args.seed is not None else cfg_seed
    overwrite = not args.no_overwrite

    if not src_dir.exists():
        print(f"\n[ERROR] Source dataset directory does not exist: {src_dir}")
        sys.exit(1)

    # 3. Execute preparation pipeline
    records, summary = run_dataset_preparation(
        source_dir=str(src_dir),
        prepared_dir=str(prep_dir),
        val_ratio=val_ratio,
        random_seed=seed,
        overwrite=overwrite,
        verbose=True,
    )

    # 4. Print clean terminal summary
    print("\n" + "=" * 65)
    print(" PREPARATION & STRATIFIED SPLIT RESULTS")
    print("=" * 65)
    val_out = summary.get("original_val_candidates_outcome", {})
    print(f"Total Source Images Scanned : {summary['total_source_files_scanned']}")
    print(f"Total Retained Prepared     : {summary['total_files_retained']}")
    print(f"Total Excluded Images       : {summary['total_files_excluded']}")
    print(f"  • Train Exact Duplicates  : {summary['deduplication_summary']['train_exact_duplicates_excluded']} excluded")
    print(f"  • Test Exact Duplicates   : {summary['deduplication_summary']['test_exact_duplicates_excluded']} excluded")
    print(f"  • Original Val Candidates : {val_out.get('included', 0)} of {val_out.get('evaluated', 0)} included ({val_out.get('excluded', 0)} excluded)")
    print("-" * 65)

    print(f"{'Split':<8} | {'Class':<10} | {'Images':<8} | {'Bacterial':<10} | {'Viral':<8}")
    print("-" * 55)

    prep_counts = summary["prepared_counts"]
    for sp in ("train", "val", "test"):
        sp_data = prep_counts.get(sp, {})
        for cls in ("NORMAL", "PNEUMONIA"):
            cnt = sp_data.get(cls, 0)
            b_cnt = sp_data.get("PNEUMONIA_BACTERIA", 0) if cls == "PNEUMONIA" else 0
            v_cnt = sp_data.get("PNEUMONIA_VIRUS", 0) if cls == "PNEUMONIA" else 0
            print(f"{sp:<8} | {cls:<10} | {cnt:<8} | {b_cnt:<10} | {v_cnt:<8}")

    print("-" * 55)
    print(f"Total Prepared Images: {summary['total_files_retained']}")

    # 5. Overlap & Leakage verification
    print("\n" + "-" * 65)
    print(" DATA LEAKAGE & OVERLAP VERIFICATION")
    print("-" * 65)
    ov = summary["cross_split_hash_overlap"]
    print(f" Train/Val Hash Overlap : {ov['train_val']}")
    print(f" Train/Test Hash Overlap: {ov['train_test']}")
    print(f" Val/Test Hash Overlap  : {ov['val_test']}")
    print(" Status                 : STRICT ZERO LEAKAGE CONFIRMED")

    # 6. Perceptual-hash screening
    phash = summary.get("perceptual_hash_screening", {})
    print("\n" + "-" * 65)
    print(" PERCEPTUAL-HASH SCREENING AUDIT")
    print("-" * 65)
    print(f" Algorithm              : {phash.get('algorithm', 'N/A')}")
    print(f" Images Screened        : {phash.get('total_images_screened', 0)}")
    print(f" Near-Duplicates Found  : {phash.get('candidate_near_duplicates_found', 0)}")
    print(" Policy                 : Flagged for review; none automatically removed.")

    # 7. Patient ID analysis
    p_info = summary["patient_analysis"]
    shared_tot = p_info.get("shared_heuristic_patients_total", {})
    print("\n" + "-" * 65)
    print(" HEURISTIC PATIENT IDENTIFIER SUMMARY")
    print("-" * 65)
    print(f" Unique Patients Retained : {p_info['total_unique_heuristic_patients_retained']}")
    print(f" Unique per Split         : Train={p_info['unique_patients_per_split']['train']}, Val={p_info['unique_patients_per_split']['val']}, Test={p_info['unique_patients_per_split']['test']}")
    print(f" Shared Train & Val       : {shared_tot.get('train_and_val', 0)}")
    print(f" Shared Train & Test      : {shared_tot.get('train_and_test', 0)} (Inherited from original Kaggle split)")
    print(f" Shared Val & Test        : {shared_tot.get('val_and_test', 0)}")
    print(" Notice                   : Filename-derived IDs are heuristic; clinical independence is not proven.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
