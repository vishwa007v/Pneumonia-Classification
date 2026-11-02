# Dataset Preparation & Stratified Splitting Report

* **Execution Timestamp**: 2026-10-10 23:23:54
* **Pipeline Status**: SUCCESS
* **Random Seed**: `42` (Deterministic)
* **Validation Ratio**: `15.0%`
* **Source Dataset**: `E:\Project1\gpu-network\data\chest img data\chest_xray` (Preserved Read-Only)
* **Prepared Dataset**: `E:\Project1\gpu-network\data\prepared`

---

## 1. Executive Summary & Inventory

| Category | Count | Description |
| :--- | :--- | :--- |
| **Total Source Files Scanned** | **5856** | Total images across original train, val, and test splits |
| **Total Files Retained** | **5824** | Cleaned, deduplicated images in prepared dataset |
| **Total Files Excluded** | **32** | Confirmed exact duplicate images |
| • Train Duplicates Excluded | 26 | Redundant byte-identical copies (1 Normal, 25 Pneumonia) |
| • Test Duplicates Excluded | 6 | Redundant byte-identical copies (3 Normal, 3 Pneumonia) |
| • Original Val Candidates Included | 16 of 16 | Passed integrity, decodability, label & duplicate checks |

---

## 2. Before & After Class Distribution Matrix

### Original Dataset Distribution (Before Cleaning)
| Split | NORMAL | PNEUMONIA (Total) | • Bacterial | • Viral | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `train` | 1341 | 3875 | 2530 | 1345 | 5216 |
| `val` | 8 | 8 | 8 | 0 | 16 |
| `test` | 234 | 390 | 242 | 148 | 624 |
| **Total** | **1583** | **4273** | **2780** | **1493** | **5856** |

### Prepared Dataset Distribution (Final Assignments)
| Split | NORMAL | PNEUMONIA (Total) | • Bacterial | • Viral | Total |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`train`** (Preserved 85%) | **1139** | **3272** | 2135 | 1137 | **4411** |
| **`val`** (15% split + 16 orig val) | **209** | **586** | 385 | 201 | **795** |
| **`test`** (Preserved benchmark) | **231** | **387** | 240 | 147 | **618** |
| **Prepared Total** | **1579** | **4245** | **2760** | **1485** | **5824** |

---

## 3. Data Leakage & Split Isolation Checks

* **Train / Val Exact-Hash Overlap**: `0`
* **Train / Test Exact-Hash Overlap**: `0`
* **Val / Test Exact-Hash Overlap**: `0`
* **Split Isolation Status**: **PASSED (Strict zero bitwise leakage across splits)**

---

## 4. Perceptual-Hash Screening Audit

* **Algorithm**: `256-bit Difference Hash (dHash 16x16) via Pillow`
* **Images Screened**: `5824`
* **Hamming Distance Threshold**: `<= 2` bits
* **Candidate Near-Duplicates Found**: `0`
* **Policy**: `Candidate near-duplicates are flagged for informational review. No images are automatically removed based on perceptual similarity.`

---

## 5. Patient-Level Identifier Analysis & Clinical Limitations

* **Unique Heuristic Patients Retained**: 3117
* **Unique Patients per Split**:
  * `train`: 2558
  * `val`: 687
  * `test`: 426
* **Shared Heuristic Patients Between Splits**:
  * `train` and `val`: 384
  * `train` and `test`: 155 (Directly inherited from Kermany Kaggle release)
  * `val` and `test`: 55

> [!WARNING] Clinical Generalization Limitation Notice
> Patient identifiers are heuristic approximations extracted from filename prefixes. For Pneumonia images, 'person<ID>' identifies pediatric subjects. For Normal images, 'IM-<ID>' and 'NORMAL2-IM-<ID>' represent DICOM acquisition instance/series numbers, NOT verified patient identifiers. Crucially, the original publication (Kermany et al., 2018) released the official dataset with known patient overlap between its original train and test directories (e.g., person100 appears in both train and test). Furthermore, no clinical demographic or patient cross-reference tables were published. Therefore, while exact bitwise image independence is strictly verified (0 exact-hash leaks across splits), clinical patient-level independence cannot be proven from filenames alone and must be acknowledged as a benchmark limitation.

---

## 6. Generated Manifests

All individual file assignments, hashes, and exclusion reasons are recorded in:
* `reports/manifests/dataset_manifest_full.csv` (Complete audit trail: 5,856 rows)
* `reports/manifests/train_manifest.csv` (Train set: 4,411 rows)
* `reports/manifests/val_manifest.csv` (Validation set: 795 rows)
* `reports/manifests/test_manifest.csv` (Test set: 618 rows)
* `reports/manifests/excluded_manifest.csv` (Exclusions: 32 rows)
