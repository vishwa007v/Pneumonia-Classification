"""
Main Entry Script: Dataset Integrity and Environment Audit.

Usage Examples:
  # Using default local Windows path from config:
  python scripts/run_audit.py

  # Explicit custom path:
  python scripts/run_audit.py --data-dir "data/chest img data/chest_xray"

  # In Kaggle environment:
  python scripts/run_audit.py --data-dir "/kaggle/input/chest-xray-pneumonia/chest_xray"
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported cleanly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.device import print_device_info
from src.data.audit import DatasetAuditor


def resolve_default_dataset_path(config_path: Path) -> Path:
    """
    Determines the appropriate dataset path:
    1. Environment variable DATASET_DIR if defined.
    2. Kaggle path if running in a Kaggle environment (/kaggle/input/...).
    3. Local Windows path from config file.
    4. Relative fallback 'data/chest img data/chest_xray'.
    """
    if env_path := os.environ.get("DATASET_DIR"):
        return Path(env_path)

    kaggle_path = Path("/kaggle/input/chest-xray-pneumonia/chest_xray")
    if kaggle_path.exists():
        return kaggle_path

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                local_p = cfg.get("dataset_paths", {}).get("local_windows")
                if local_p:
                    candidate = (project_root / local_p).resolve()
                    if candidate.exists():
                        return candidate
        except Exception:
            pass

    return project_root / "data" / "chest img data" / "chest_xray"


def main():
    parser = argparse.ArgumentParser(
        description="Audit Chest X-Ray dataset integrity, class balances, and compute environment."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset directory (defaults to path in configs/dataset_config.json)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.json",
        help="Path to JSON dataset configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save audit JSON report",
    )
    parser.add_argument(
        "--hash-algo",
        type=str,
        choices=["md5", "sha256"],
        default="md5",
        help="Cryptographic hash algorithm for duplicate detection (default: md5)",
    )
    args = parser.parse_args()

    # 1. Print hardware and environment diagnostics
    print_device_info()

    # 2. Resolve dataset path
    config_file = (project_root / args.config).resolve()
    if args.data_dir:
        dataset_path = Path(args.data_dir)
        if not dataset_path.is_absolute():
            dataset_path = (project_root / dataset_path).resolve()
    else:
        dataset_path = resolve_default_dataset_path(config_file)

    print("\n" + "=" * 65)
    print(" DATASET AUDIT AND INTEGRITY VERIFICATION")
    print("=" * 65)

    if not dataset_path.exists():
        print(f"\n[ERROR] Dataset path not found: {dataset_path}")
        print("Please specify a valid path using --data-dir or verify configs/dataset_config.json")
        sys.exit(1)

    # 3. Execute audit
    auditor = DatasetAuditor(dataset_root=str(dataset_path), hash_algo=args.hash_algo)
    report = auditor.run_audit(verbose=True)

    # 4. Print clean summary table
    print("\n" + "=" * 65)
    print(" DATASET AUDIT SUMMARY RESULTS")
    print("=" * 65)
    print(f"{'Split':<8} | {'Class':<10} | {'Images':<8} | {'Bacterial':<10} | {'Viral':<8}")
    print("-" * 55)

    total_valid_all = 0
    split_stats = report["split_statistics"]
    for split_name, sinfo in split_stats.items():
        if sinfo.get("status") == "PRESENT":
            for cls_name, cinfo in sinfo.get("classes", {}).items():
                v_count = cinfo.get("valid_images", 0)
                b_count = cinfo.get("bacteria", 0)
                vir_count = cinfo.get("virus", 0)
                total_valid_all += v_count
                print(f"{split_name:<8} | {cls_name:<10} | {v_count:<8} | {b_count:<10} | {vir_count:<8}")
        else:
            print(f"{split_name:<8} | [MISSING SPLIT]")

    print("-" * 55)
    print(f"Total Verified Images Across Splits: {total_valid_all}")

    # 5. Display Image Dimension & Mode Info
    res_summary = report.get("resolution_summary", {})
    min_wh = res_summary.get("min_dimensions_wh")
    max_wh = res_summary.get("max_dimensions_wh")
    modes = res_summary.get("color_modes", {})
    if min_wh and max_wh:
        print(f"\nImage Resolution Range : Min ({min_wh[0]}x{min_wh[1]}) | Max ({max_wh[0]}x{max_wh[1]})")
    print(f"Image Color Modes      : {dict(modes)}")

    # 6. Display Issues & Duplicate Findings
    unreadable = report["unreadable_corrupt_images"]["count"]
    intra_dups = report["duplicate_analysis"]["intra_split_duplicate_groups"]
    dups_by_split = report["duplicate_analysis"].get("intra_split_duplicates_by_split", {})
    split_dup_str = ", ".join(f"{s}: {count} groups" for s, count in dups_by_split.items())
    class_conflicts = report["duplicate_analysis"].get("cross_class_conflict_groups", 0)
    cross_leaks = report["duplicate_analysis"]["cross_split_leakage_groups"]

    print("\n" + "-" * 65)
    print(" INTEGRITY & DATA LEAKAGE CHECKS")
    print("-" * 65)
    print(f" Corrupt / Unreadable Images        : {unreadable}")
    print(f" Intra-Split Duplicate Hash Groups  : {intra_dups} ({split_dup_str})")
    print(f" Cross-Class Label Conflict Groups  : {class_conflicts}")
    print(f" Cross-Split Data Leakage Groups    : {cross_leaks}")

    if class_conflicts > 0:
        print("\n [WARNING] Cross-class duplicate label conflicts detected:")
        for item in report["duplicate_analysis"]["cross_class_conflicts"]:
            print(f"   • Hash {item['hash'][:10]}... has conflicting classes: {item['conflicting_classes']}")

    if cross_leaks > 0:
        print("\n [WARNING] Cross-split duplicate hashes detected:")
        for item in report["duplicate_analysis"]["cross_split_leakages"][:5]:
            splits_affected = ", ".join(item["splits"])
            names = [occ["name"] for occ in item["occurrences"]]
            print(f"   • Hash {item['hash'][:10]}... shared across [{splits_affected}]: {names}")

    # 7. Operating system artifacts
    artifacts = report.get("filesystem_artifacts", {})
    if artifacts.get("details"):
        print("\n [NOTICE] Filesystem Artifacts:")
        for d in artifacts["details"]:
            print(f"   • {d}")

    # 8. Save report
    out_dir = (project_root / args.output_dir).resolve()
    report_filename = "dataset_audit_report.json"
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                report_filename = cfg.get("reports", {}).get("report_filename", report_filename)
        except Exception:
            pass
    report_file = out_dir / report_filename
    auditor.save_report(report, str(report_file))
    print(f"\n Audit report saved to: {report_file}")
    print(f" Overall Audit Status : {report['overall_status']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
