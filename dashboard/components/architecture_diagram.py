import streamlit as st

def render_architecture_diagram():
    """
    Renders an HTML/CSS vertical pipeline diagram illustrating the exact
    CXformer-Small classification pipeline with status badges (Frozen vs Trainable).
    """
    html_content = """
    <div style="display: flex; flex-direction: column; align-items: center; gap: 12px; margin: 1.5rem 0;">
        
        <!-- Step 1: Input -->
        <div style="background: #111827; border: 2px solid #06b6d4; border-radius: 10px; width: 85%; max-width: 650px; padding: 14px 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #06b6d4;">🩻 INPUT CHEST X-RAY</span>
                <span style="background: #083344; color: #67e8f9; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">RAW INPUT</span>
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 6px;">
                RGB input &bull; Single image &bull; Resolution: <strong>518 &times; 518 pixels</strong> &bull; 3 channels
            </div>
        </div>

        <div style="font-size: 1.5rem; color: #94a3b8; font-weight: bold;">&darr;</div>

        <!-- Step 2: Preprocessing -->
        <div style="background: #111827; border: 2px solid #0d9488; border-radius: 10px; width: 85%; max-width: 650px; padding: 14px 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #0d9488;">🔧 OFFICIAL CXFORMER PREPROCESSING</span>
                <span style="background: #042f2e; color: #5eead4; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">CustomBitImageProcessor</span>
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 6px;">
                &bull; <strong>Channel-wise Histogram Equalization</strong> (cv2.equalizeHist)<br>
                &bull; Shortest-edge resize to 518 &bull; Center-crop to <strong>518 &times; 518</strong> (Bicubic, resample=3)<br>
                &bull; Rescale 1/255 &bull; ImageNet Normalization (Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225])<br>
                &bull; Output Tensor: <code>(3, 518, 518)</code> Float32
            </div>
        </div>

        <div style="font-size: 1.5rem; color: #94a3b8; font-weight: bold;">&darr;</div>

        <!-- Step 3: Patch Embedding -->
        <div style="background: #111827; border: 2px solid #3b82f6; border-radius: 10px; width: 85%; max-width: 650px; padding: 14px 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #3b82f6;">🧩 PATCH EMBEDDING & TOKENIZATION</span>
                <span style="background: #1e3a5f; color: #93c5fd; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">TOKEN PROJECTION</span>
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 6px;">
                &bull; Patch size: <strong>14 &times; 14</strong> pixels &bull; Grid: <strong>37 &times; 37</strong> patches<br>
                &bull; <strong>1,369</strong> patch tokens + <strong>4</strong> register tokens + <strong>1</strong> [CLS] token = <strong>1,374 total tokens</strong><br>
                &bull; Token embedding dimension: <strong>384</strong>
            </div>
        </div>

        <div style="font-size: 1.5rem; color: #94a3b8; font-weight: bold;">&darr;</div>

        <!-- Step 4: Frozen Backbone -->
        <div style="background: #181b29; border: 2.5px solid #f59e0b; border-radius: 10px; width: 85%; max-width: 650px; padding: 16px 20px; box-shadow: 0 6px 12px rgba(245, 158, 11, 0.15);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.2rem; font-weight: 700; color: #f59e0b;">🔒 DINOv2 BACKBONE WITH REGISTERS</span>
                <span style="background: #78350f; color: #fde68a; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 700;">🔒 FROZEN BACKBONE</span>
            </div>
            <div style="font-size: 0.95rem; color: #e2e8f0; margin-top: 8px; line-height: 1.6;">
                &bull; <strong>12 Vision Transformer Layers</strong> &bull; <strong>6 Attention Heads</strong> per layer<br>
                &bull; Hidden dimension: <strong>384</strong> &bull; Intermediate MLP dimension: <strong>1,536</strong><br>
                &bull; Pretrained on <strong>600,000+ Chest X-Rays</strong> by M42 Health (CC BY-NC 4.0)<br>
                &bull; <strong>22,058,112 parameters (100% Frozen, requires_grad = False)</strong><br>
                &bull; Evaluated under <code>torch.no_grad()</code> &bull; Zero backbone activation gradients
            </div>
        </div>

        <div style="font-size: 1.5rem; color: #94a3b8; font-weight: bold;">&darr;</div>

        <!-- Step 5: CLS Token Extraction -->
        <div style="background: #111827; border: 2px solid #8b5cf6; border-radius: 10px; width: 85%; max-width: 650px; padding: 14px 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #a78bfa;">🎯 [CLS] TOKEN REPRESENTATION</span>
                <span style="background: #2e1065; color: #ddd6fe; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">SEQUENCE INDEX 0</span>
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 6px;">
                &bull; Backbone output shape: <code>(1, 1374, 384)</code><br>
                &bull; Extracts sequence position <code>[:, 0, :]</code> &rarr; <strong>384-dimensional feature vector</strong><br>
                &bull; 4 register tokens and 1,369 patch tokens discarded from standard classification forward pass
            </div>
        </div>

        <div style="font-size: 1.5rem; color: #94a3b8; font-weight: bold;">&darr;</div>

        <!-- Step 6: Trainable Classification Head -->
        <div style="background: #112217; border: 2.5px solid #10b981; border-radius: 10px; width: 85%; max-width: 650px; padding: 16px 20px; box-shadow: 0 6px 12px rgba(16, 185, 129, 0.15);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.2rem; font-weight: 700; color: #10b981;">✏️ TRAINABLE CLASSIFICATION HEAD</span>
                <span style="background: #064e3b; color: #a7f3d0; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem; font-weight: 700;">✏️ TRAINABLE HEAD</span>
            </div>
            <div style="font-size: 0.95rem; color: #e2e8f0; margin-top: 8px; line-height: 1.6;">
                &bull; <strong>LayerNorm(384)</strong> &bull; Parameters: 768 (weight & bias)<br>
                &bull; <strong>Dropout(p = 0.3)</strong> &bull; Regularization during training<br>
                &bull; <strong>Linear(384, 2)</strong> &bull; Parameters: 770 (weight: 384&times;2 + bias: 2)<br>
                &bull; <strong>Total Head Parameters: 1,538 (100% Trainable, optimized with AdamW)</strong>
            </div>
        </div>

        <div style="font-size: 1.5rem; color: #94a3b8; font-weight: bold;">&darr;</div>

        <!-- Step 7: Output Probabilities -->
        <div style="background: #111827; border: 2px solid #06b6d4; border-radius: 10px; width: 85%; max-width: 650px; padding: 14px 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.15rem; font-weight: 700; color: #06b6d4;">📊 SOFTMAX PROBABILITIES & DECISION</span>
                <span style="background: #083344; color: #67e8f9; padding: 3px 10px; border-radius: 12px; font-size: 0.75rem; font-weight: 600;">BINARY OUTPUT</span>
            </div>
            <div style="font-size: 0.95rem; color: #cbd5e1; margin-top: 6px;">
                &bull; Logits &rarr; <code>Softmax(dim=-1)</code> &rarr; <code>[P(NORMAL), P(PNEUMONIA)]</code><br>
                &bull; Decision rule: <code>Argmax(P)</code> (threshold = 0.5)<br>
                &bull; Output: <strong>0 = NORMAL</strong> or <strong>1 = PNEUMONIA</strong>
            </div>
        </div>

    </div>
    """
    st.html(html_content)
