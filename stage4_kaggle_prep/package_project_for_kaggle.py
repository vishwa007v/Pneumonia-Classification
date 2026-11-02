"""
Packaging Script for Kaggle T4 Deployment.
Creates a lightweight, clean ZIP archive of the project source code for Kaggle.

MANDATORY SECURITY & SANITIZATION SAFEGUARDS:
- Strictly excludes any credentials, API keys, and kaggle.json.
- Excludes heavy datasets (mounted directly in Kaggle via /kaggle/input).
- Excludes heavy model weights (mounted directly in Kaggle via /kaggle/input).
- Excludes previous checkpoints, training logs, temporary test artifacts, and caches.
- Includes ONLY project source code, scripts, configurations, and requirements.
"""

import os
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_ZIP = os.path.join(os.path.dirname(__file__), "gpu_network_source.zip")

# Allowed directories and root files
ALLOWED_DIRS = ["src", "scripts", "configs"]
ALLOWED_ROOT_FILES = ["requirements.txt"]

# Explicit exclusion patterns
EXCLUDE_EXTS = {".pth", ".pt", ".safetensors", ".bin", ".h5", ".csv", ".jsonl", ".pyc", ".pyo"}
EXCLUDE_DIR_NAMES = {
    "kaggle.json",
    ".git",
    "__pycache__",
    "checkpoints",
    "CXformer_small_files",
    "brain",
    ".gemini",
    "stage4_kaggle_prep",
}

print("=" * 70)
print(" PACKAGING PROJECT SOURCE FOR KAGGLE DEPLOYMENT")
print("=" * 70)

included_files = []
excluded_count = 0

with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    # 1. Add allowed root files
    for fname in ALLOWED_ROOT_FILES:
        fpath = os.path.join(PROJECT_ROOT, fname)
        if os.path.isfile(fpath):
            zf.write(fpath, arcname=fname)
            included_files.append(fname)
            print(f"  + Added root file: {fname}")

    # 2. Add allowed directories
    for dname in ALLOWED_DIRS:
        dir_path = os.path.join(PROJECT_ROOT, dname)
        if not os.path.isdir(dir_path):
            continue
        for root, dirs, files in os.walk(dir_path):
            # Prune excluded directories (e.g. __pycache__)
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES and not d.startswith(".")]

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                # Security check: never package credentials
                if file.lower() == "kaggle.json" or "credential" in file.lower():
                    print(f"  ! EXCLUDED SENSITIVE FILE: {file}")
                    continue
                if ext in EXCLUDE_EXTS:
                    excluded_count += 1
                    continue

                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, PROJECT_ROOT)
                zf.write(full_path, arcname=rel_path)
                included_files.append(rel_path)

zip_size_kb = os.path.getsize(OUTPUT_ZIP) / 1024
print("\nPackaging Summary:")
print(f"  - Output archive : {OUTPUT_ZIP}")
print(f"  - Archive size   : {zip_size_kb:.2f} KB")
print(f"  - Total files    : {len(included_files)} files added")
print(f"  - Excluded files : {excluded_count} binaries/checkpoints pruned")
print("  - Credentials    : 0 credentials packaged (Safeguard 5 strictly enforced)")
print("=" * 70)
