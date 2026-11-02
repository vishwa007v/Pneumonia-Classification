import streamlit as st

def render_top_navbar():
    """
    Renders a modern, colorful horizontal top navigation bar across all pages.
    Replaces the sidebar navigation with a responsive horizontal bar.
    """
    # Top branding and status strip
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0d1b2e 0%, #112240 50%, #0a192f 100%); 
                border: 1px solid #1e3a5f; 
                border-radius: 12px; 
                padding: 12px 20px; 
                margin-bottom: 14px; 
                display: flex; 
                justify-content: space-between; 
                align-items: center; 
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.35);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.8rem;">🩺</span>
            <div>
                <span style="font-size: 1.25rem; font-weight: 800; color: #ffffff; letter-spacing: 0.02em;">
                    CXformer-Small
                </span>
                <span style="font-size: 0.85rem; color: #38bdf8; font-weight: 600; margin-left: 8px; background: rgba(56, 189, 248, 0.12); padding: 3px 8px; border-radius: 6px; border: 1px solid rgba(56, 189, 248, 0.3);">
                    Medical Vision Transformer
                </span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">
                    GPU-Accelerated Chest Radiograph Classification & Interpretability
                </div>
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 0.8rem; background: #064e3b; color: #6ee7b7; padding: 4px 10px; border-radius: 20px; font-weight: 700; border: 1px solid #059669;">
                ● Tesla T4 GPU
            </span>
            <span style="font-size: 0.8rem; background: #1e1b4b; color: #c084fc; padding: 4px 10px; border-radius: 20px; font-weight: 700; border: 1px solid #7c3aed;">
                DINOv2 + Registers
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 9-column horizontal navigation tabs
    nav_cols = st.columns([1.1, 1.1, 1.2, 1.1, 1.1, 1.2, 1.1, 1.2, 1.0])

    with nav_cols[0]:
        st.page_link("app.py", label="Overview", icon="🏠", use_container_width=True)
    with nav_cols[1]:
        st.page_link("pages/1_Dataset_Analytics.py", label="Dataset", icon="📊", use_container_width=True)
    with nav_cols[2]:
        st.page_link("pages/2_Model_Architecture.py", label="Architecture", icon="🧠", use_container_width=True)
    with nav_cols[3]:
        st.page_link("pages/3_Training_Performance.py", label="Training", icon="📈", use_container_width=True)
    with nav_cols[4]:
        st.page_link("pages/4_Model_Evaluation.py", label="Evaluation", icon="🎯", use_container_width=True)
    with nav_cols[5]:
        st.page_link("pages/5_Explainable_AI.py", label="Grad-CAM", icon="🔬", use_container_width=True)
    with nav_cols[6]:
        st.page_link("pages/6_Image_Classification.py", label="Inference", icon="🩻", use_container_width=True)
    with nav_cols[7]:
        st.page_link("pages/7_Cross_Validation.py", label="5-Fold CV", icon="🔁", use_container_width=True)
    with nav_cols[8]:
        st.page_link("pages/8_About.py", label="About", icon="ℹ️", use_container_width=True)

    st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
