import os, sys
# Ensure project root is on path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from PIL import Image
import io
import base64
from model.advanced_deblur import load_deblur_model  # Use advanced hybrid deblurring model

# Import utilities directly
from app.utils import save_uploaded_file, create_download_link

# Page configuration
st.set_page_config(
    page_title="Image Deblurring AI",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .result-section {
        background: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .image-container {
        text-align: center;
        padding: 1rem;
        background: #f8f9fa;
        border-radius: 8px;
        margin: 0.5rem;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    .download-button {
        background: #28a745;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        text-decoration: none;
        font-weight: bold;
        display: inline-block;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .download-button:hover {
        background: #218838;
        text-decoration: none;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'deblurred_image' not in st.session_state:
    st.session_state.deblurred_image = None

@st.cache_resource
def get_model():
    """Load advanced hybrid deblurring model once and cache it."""
    model = load_deblur_model()
    return model

def main():
    # Header
    st.markdown("""    <div class="main-header">
        <h1>🖼️ Advanced Hybrid Deblurring AI</h1>
        <p>Precision deblurring with perfect color preservation using advanced classical and neural techniques.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload a Blurred Image")
    st.markdown("Supported formats: JPG, JPEG, PNG")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a blurred image to get a sharp version"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Save uploaded file
        st.session_state.uploaded_image = Image.open(uploaded_file)
        
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original (Blurred)")
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(st.session_state.uploaded_image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Deblurred Result")
            if st.session_state.deblurred_image is not None:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(st.session_state.deblurred_image, use_container_width=True)
                
                # Download button
                img_buffer = io.BytesIO()
                st.session_state.deblurred_image.save(img_buffer, format='JPEG', quality=95)
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                
                download_filename = f"{uploaded_file.name.split('.')[0]}_deblurred.jpg"
                href = f'<a href="data:image/jpeg;base64,{img_str}" download="{download_filename}" class="download-button">📥 Download Deblurred Image</a>'
                st.markdown(href, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.info("Click 'Deblur Image' to process")
                st.markdown('</div>', unsafe_allow_html=True)
          # Deblur button
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 Deblur Image", use_container_width=True):
                model = get_model()
                if model is not None:
                    with st.spinner("Deblurring image with DeblurGANv2..."):
                        try:
                            st.session_state.deblurred_image = model.deblur_image(st.session_state.uploaded_image)
                            st.success("Image deblurred successfully with DeblurGANv2!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error during deblurring: {e}")
                            st.info("Check the terminal for detailed error messages.")

    # Footer
    st.markdown("""
    <footer style="text-align: center; padding: 2rem 0;">
        <p style="color: #666;">&copy; 2023 Image Deblurring AI. All rights reserved.</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
