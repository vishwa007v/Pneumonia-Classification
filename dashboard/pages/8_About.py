import streamlit as st
import pandas as pd
from components.sidebar import render_sidebar
from components.navbar import render_top_navbar

st.set_page_config(page_title="About the Project — CXformer-Small", page_icon="ℹ️", layout="wide")
render_sidebar()
render_top_navbar()

st.title("ℹ️ About the Project & Clinical Research Summary")
st.markdown("""
### **GPU-Accelerated Medical Image Classification and Explainability System Using Deep Learning**
Academic Final-Year Engineering & Medical AI Research Project.
""")

st.error("""
⚠️ **MANDATORY CLINICAL DISCLAIMER:**
This software is an academic engineering research prototype developed solely for educational and investigative purposes.
It is NOT approved by the FDA, EMA, or any regulatory health body as a medical device or diagnostic software.
It MUST NOT be used for real-time patient diagnosis, clinical management, treatment planning, or emergency room triage.
""")

st.divider()

# Section 1: Problem Statement & Motivation
st.subheader("🏥 Problem Statement & Clinical Motivation")
st.markdown("""
Pneumonia is an acute respiratory infection responsible for millions of hospital admissions globally each year, particularly among pediatric and elderly demographics.
Chest radiography (CXR) remains the frontline diagnostic imaging modality due to its rapid acquisition, low radiation dose, and widespread availability.
However, rapid radiograph interpretation is constrained by:
1. **Specialist Bottlenecks:** Shortages of certified thoracic radiologists lead to substantial triage delays in crowded emergency departments.
2. **Subtle Manifestations:** Early viral, bacterial, and atypical pneumonic consolidations exhibit visual subtlety that overlaps with atelectasis, normal vascular markings, or pulmonary edema.
3. **The Sensitivity-Specificity Dilemma:** In clinical practice, missing an active pneumonia case (False Negative) can be life-threatening, whereas false alarms (False Positives) burden healthcare infrastructure with unnecessary antibiotic regimens and follow-up CT scans.
""")

st.divider()

# Section 2: Project Objectives
st.subheader("🎯 Primary Technical Objectives")
st.markdown("""
1. **Foundation Model Transfer Learning:** Adapt **CXformer-Small**, a Vision Transformer foundation model self-supervised pretrained on 600,000+ chest X-rays by M42 Health, for downstream binary classification (`NORMAL` vs `PNEUMONIA`).
2. **Strict Preprocessing Pipeline Fidelity:** Implement and enforce the official `CustomBitImageProcessor` pipeline (518 × 518 bicubic interpolation, channel-wise histogram equalization, ImageNet normalization).
3. **Hardware-Accelerated Training:** Train a newly initialized linear classification head on an **NVIDIA Tesla T4 GPU** while keeping the 22.06M backbone frozen to preserve pretrained medical representations.
4. **Honest Generalization Assessment:** Evaluate the model strictly on a held-out benchmark split of 618 radiographs without post-hoc threshold tweaking or data leakage.
5. **Explainability & Interpretability:** Implement Transformer Grad-CAM via the penultimate transformer layer to inspect model saliency and detect potential non-pulmonary shortcut artifacts.
6. **Cross-Validation Rigor:** Formulate a 5-fold stratified cross-validation protocol on the 5,206-image training pool to evaluate stability.
""")

st.divider()

# Section 3: Summary of Empirical Results
st.subheader("📊 Summary of Verified Empirical Findings")

res_data = {
    "Clinical Performance Dimension": [
        "Held-Out Test Accuracy",
        "Pneumonia Sensitivity / Recall",
        "Normal Specificity",
        "Pneumonia Precision",
        "Pneumonia F1-Score",
        "Discriminative Ranking (ROC-AUC)",
        "False Negative Count (Missed Infections)",
        "False Positive Count (False Alarms)",
        "Validation-to-Test Accuracy Gap",
        "Validation-to-Test Specificity Gap"
    ],
    "Empirical Finding": [
        "89.00 % (550 of 618 correct)",
        "99.74 % (386 of 387 pneumonia cases detected)",
        "71.00 % (164 of 231 healthy cases confirmed)",
        "85.21 % (Positive predictive value)",
        "91.90 % (Harmonic mean of precision and recall)",
        "98.52 % (High ranking separation)",
        "1 out of 387 cases (0.26 % miss rate)",
        "67 out of 231 normal cases (29.01 % false alarm rate)",
        "−8.99 percentage points (97.99% Val &rarr; 89.00% Test)",
        "−26.61 percentage points (97.61% Val &rarr; 71.00% Test)"
    ]
}
st.dataframe(pd.DataFrame(res_data), use_container_width=True, hide_index=True)

st.divider()

# Section 4: Critical Limitations & Scientific Caveats
st.subheader("⚠️ Critical Limitations & Scientific Caveats (Anti-Hype Policy)")

st.markdown("""
<div class="custom-alert-danger">
<strong>Scientific Caveats That Must Be Disclosed:</strong><br><br>
1. <strong>Substantial False-Positive Burden:</strong> The model misclassifies 29.01% of normal radiographs as pneumonia (67 false positives). Exactly 32 of these false positives exhibited high confidence (P &gt; 0.90).<br>
2. <strong>Validation-to-Test Performance Disconnect:</strong> The model demonstrated strong validation metrics (97.99% acc, 97.61% spec), but suffered a steep drop on the independent test set. The root cause remains unproven.<br>
3. <strong>Unresolved Patient-Level Independence:</strong> Filename-derived patient ID analysis identified 155 overlapping heuristic patient keys between train and test splits. Strict patient-grouped splitting could not be guaranteed.<br>
4. <strong>Saliency Beyond Pulmonary Fields:</strong> Qualitative Grad-CAM audits revealed activation outside the thoracic cage (e.g. cervical spine, clavicular edges, collimation borders), highlighting risk of artifact reliance.<br>
5. <strong>Single Institutional Dataset:</strong> The model has not been evaluated against external hospital cohorts (e.g. MIMIC-CXR, CheXpert, NIH ChestX-ray14).<br>
6. <strong>Incomplete Cross-Validation:</strong> At the time of project export, 5-fold cross-validation had completed folds 1–3, checkpointed fold 4, and left fold 5 pending.
</div>
""", unsafe_allow_html=True)

st.divider()

# Section 5: Future Enhancements
st.subheader("🚀 Roadmap & Future Enhancements")
st.markdown("""
- **Threshold Calibration:** Optimize decision thresholds (e.g. Youden's J statistic) on validation splits to balance sensitivity against specificity.
- **Backbone Layer Unfreezing:** Explore selective fine-tuning of the final 2 transformer blocks to adapt representations while mitigating catastrophic forgetting.
- **Attention Rollout & Vision Attention Maps:** Complement Grad-CAM with intrinsic Multi-Head Self-Attention rollout visualization.
- **External Clinical Generalization:** Validate on external multi-site datasets to test domain adaptation.
- **Multimodal Integration:** Combine radiographic tokens with patient clinical covariates (WBC count, temperature, oxygen saturation).
""")

st.divider()

# Section 6: References & Citations
st.subheader("📚 Academic References & Resources")
st.markdown("""
1. **CXformer Foundation Model:** M42 Health, Abu Dhabi. *CXformer-Small Pretrained Vision Transformer for Medical Imaging.* Hugging Face: `m42-health/CXformer` (2024). License: CC BY-NC 4.0.
2. **DINOv2 with Registers:** Oquab, M., Darcet, T., Moutakanni, T., et al. *DINOv2: Learning Robust Visual Features without Supervision.* Transactions on Machine Learning Research (2023).
3. **Vision Transformer Registers:** Darcet, T., Oquab, M., Mairal, J., & Bojanowski, P. *Vision Transformers Need Registers.* ICLR (2024).
4. **Grad-CAM:** Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* IEEE ICCV (2017).
5. **Prepared Chest Radiography Dataset:** Kermany, D. S., Goldbaum, M., Cai, W., et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell (2018).
""")
