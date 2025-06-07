import streamlit as st
from PIL import Image
import main
import os
import time
import zipfile
import eigenface as eigenface

# Setup halaman
st.set_page_config(page_title="Face Recognition App", layout="wide")

st.markdown(
    """ 
    <style>
    .main {
        background-color: #f8faff;
        color: #003366;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
    }
    .stButton>button:hover {
        background-color: #005bb5 !important;
        color: white !important;
    }
    .stSidebar {
        background-color: #e6f0ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Face Recognition App")

# Sidebar config
st.sidebar.header("📁 Load Data")

# Opsi upload dataset sebagai ZIP
dataset_zip = st.sidebar.file_uploader("Upload Dataset (.zip)", type=["zip"])

# Simpan dan ekstrak jika di-upload
dataset_dir = ""
if dataset_zip is not None:
    dataset_dir = "uploaded_dataset"
    with open("dataset.zip", "wb") as f:
        f.write(dataset_zip.read())
    with zipfile.ZipFile("dataset.zip", "r") as zip_ref:
        zip_ref.extractall(dataset_dir)
    st.sidebar.success(f"✅ Dataset extracted to {dataset_dir}")

# Alternatif input path manual
manual_path = st.sidebar.text_input("Or enter dataset path (e.g., ./dataset)")
if manual_path:
    dataset_dir = manual_path

# Upload test image
test_image = st.sidebar.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png"])

# Threshold pengenalan wajah
st.sidebar.markdown("### 🔧 Recognition Settings")
threshold_percent = st.sidebar.slider("Minimum Match Percentage (%)", 0, 100, 15)  # Default
threshold_euclid = st.sidebar.slider("Maximum Euclidean Distance", 0, 50000, 20000)  # Default

# Test image preview
if test_image is not None:
    st.image(test_image, caption="Test Image", width=256)

# Run recognition
if st.sidebar.button("🔍 Start Recognition"):
    if not dataset_dir or (test_image is None and not os.path.exists("bin/sample_image.jpg")):
        st.error("❌ The dataset is not available yet and the test image has not been selected!")
    else:
        if test_image is not None:
            with open("temp_test_image.jpg", "wb") as f:
                f.write(test_image.read())
            test_path = "temp_test_image.jpg"
        else:
            test_path = "bin/sample_image.jpg"

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Loading dataset images...")
        progress_bar.progress(20)
        
        start_time = time.time()
        
        try:
            status_text.text("Processing eigenfaces...")
            progress_bar.progress(60)
            
            result_path, match_percentage = main.run(dataset_dir, test_path, threshold_percent, threshold_euclid)
            
            progress_bar.progress(100)
            status_text.text("Recognition complete!")
            
            exec_time = round(time.time() - start_time, 3)

            # Clear progress bar
            progress_bar.empty()
            status_text.empty()

            if result_path is None:
                st.warning(f"No sufficiently similar face found.")
                st.info(f"**Similarity found: {match_percentage:.2f}%**")
                st.info("💡 Try lowering the match threshold or increasing Euclidean distance.")
                
                # Threshold yang disarankan
                suggested_percent = max(5, int(match_percentage) - 5)
                suggested_euclid = threshold_euclid + 5000
                st.info(f"🔧 **Suggested settings:** {suggested_percent}% similarity, {suggested_euclid} Euclidean distance")
                
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Test Image")
                    st.image(test_path, use_container_width=True)

                with col2:
                    st.subheader("Closest Match")
                    st.image(result_path, use_container_width=True)
                    st.success(f"**Match Found!**")
                    st.success(f"**Match Percentage: {match_percentage:.2f}%**")
                    
                    # Matched file name
                    matched_filename = os.path.basename(result_path)
                    st.info(f"**File: {matched_filename}**")

            st.write(f"**Execution Time: {exec_time} seconds**")
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Error during recognition: {str(e)}")
            st.error("Please check your dataset format and try again.")

if dataset_dir and os.path.exists(dataset_dir):
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Info")
    
    try:
        image_paths = eigenface.get_image_paths(dataset_dir)
        num_images = len(image_paths)
        st.sidebar.info(f"**{num_images} images** found in dataset")
        
        if num_images > 0:
            # Show some sample filenames
            sample_files = [os.path.basename(path) for path in image_paths[:3]]
            st.sidebar.text("Sample files:")
            for file in sample_files:
                st.sidebar.text(f"• {file}")
            if num_images > 3:
                st.sidebar.text(f"... and {num_images-3} more")
                
    except Exception as e:
        st.sidebar.error(f"Error reading dataset: {str(e)}")

st.markdown("---")
st.caption("© 2025 Face Recognition App with Streamlit")