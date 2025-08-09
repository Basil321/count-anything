"""
CountAnything AI - Main Application
A Streamlit app for counting objects in images using YOLOv8 and OpenCV.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent))

from utils.image_processor import ImageProcessor
from database.db import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="CountAnything AI",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    .threshold-info {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🔢 CountAnything AI</h1>', unsafe_allow_html=True)
    st.markdown("### Upload an image and let AI count objects for you!")
    
    # Initialize components
    image_processor = ImageProcessor()
    db_manager = DatabaseManager()
    
    # Sidebar for navigation and settings
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Object type selection
        object_type = st.selectbox(
            "Select Object Type",
            ["general", "rice", "coins", "hair"],
            help="Choose the type of objects you want to count"
        )
        
        # Confidence threshold with better explanation
        st.markdown("### 🎯 Detection Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.1,
            help="Lower values detect more objects, higher values are more selective"
        )
        
        # Show threshold information
        with st.expander("ℹ️ Threshold Guide", expanded=False):
            st.markdown("""
            **Confidence Threshold Guide:**
            - **0.1-0.3**: Detects many objects, may include false positives
            - **0.3-0.6**: Balanced detection (recommended)
            - **0.6-0.9**: Very selective, only high-confidence objects
            
            **Object Type Parameters:**
            - **Rice**: Small, round grains (20-500 pixels)
            - **Coins**: Circular objects (100-5000 pixels)  
            - **Hair**: Thin strands (10-200 pixels)
            - **General**: Any objects (50-2000 pixels)
            """)
        
        # Navigation
        st.markdown("---")
        st.title("📊 Navigation")
        page = st.selectbox(
            "Choose a page",
            ["Count Objects", "View History", "About"]
        )
    
    # Main content area
    if page == "Count Objects":
        show_count_page(image_processor, db_manager, object_type, confidence_threshold)
    elif page == "View History":
        show_history_page(db_manager)
    elif page == "About":
        show_about_page()

def show_count_page(image_processor, db_manager, object_type, confidence_threshold):
    """Display the main counting page with enhanced UI"""
    st.header("📸 Count Objects in Image")
    
    # Instructions
    with st.expander("ℹ️ How to use", expanded=False):
        st.markdown("""
        1. **Upload an image** containing objects you want to count
        2. **Select the object type** from the sidebar (rice, coins, hair, or general)
        3. **Adjust confidence threshold** - lower for more detections, higher for accuracy
        4. **Click 'Count Objects'** to process the image
        5. **View results** with colored overlays (green=high confidence, yellow=medium, red=low)
        """)
    
    # File uploader with drag and drop
    uploaded_file = st.file_uploader(
        "📁 Choose an image file or drag and drop here",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image containing objects to count"
    )
    
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Process button with loading animation
        if st.button("🔍 Count Objects", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing image with threshold-based detection..."):
                try:
                    # Add a progress bar
                    progress_bar = st.progress(0)
                    
                    # Simulate processing steps
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Process the image with threshold-based detection
                    result = image_processor.process_image(
                        uploaded_file, 
                        object_type, 
                        confidence_threshold
                    )
                    
                    # Display results in a nice format
                    with col2:
                        st.subheader("🎯 Detection Results")
                        
                        # Success message with threshold info
                        st.markdown(f"""
                        <div class="success-box">
                            <h3>✅ Found {result['count']} objects!</h3>
                            <p>Confidence: {result.get('confidence', 0.0):.2f}</p>
                            <p>Object Type: {object_type.title()}</p>
                            <p>Threshold: {confidence_threshold:.1f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show processed image with overlays
                        if result['processed_image'] is not None:
                            st.image(result['processed_image'], caption="Processed Image with Threshold Detection", use_column_width=True)
                        
                        # Threshold information
                        st.markdown(f"""
                        <div class="threshold-info">
                            <h4>🎯 Detection Details</h4>
                            <p><strong>Method:</strong> Threshold-based detection with morphological operations</p>
                            <p><strong>Color Code:</strong> Green (high confidence), Yellow (medium), Red (low)</p>
                            <p><strong>Numbers:</strong> Show detected object areas in pixels</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Contextual insights
                        insight = image_processor.get_contextual_insight(result['count'], object_type)
                        st.markdown(f"""
                        <div class="info-box">
                            <h4>💡 Insight</h4>
                            <p>{insight}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Save to database
                    db_manager.save_count_result(
                        filename=uploaded_file.name,
                        count=result['count'],
                        confidence=result.get('confidence', 0.0),
                        object_type=object_type
                    )
                    
                    # Show metrics
                    st.markdown("### 📊 Detection Metrics")
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        st.metric("Objects Detected", result['count'])
                    
                    with metric_col2:
                        st.metric("Confidence", f"{result.get('confidence', 0.0):.2f}")
                    
                    with metric_col3:
                        st.metric("Object Type", object_type.title())
                    
                    with metric_col4:
                        st.metric("Threshold", f"{confidence_threshold:.1f}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
                    st.info("💡 Try adjusting the confidence threshold or uploading a different image.")

def show_history_page(db_manager):
    """Display the history page with enhanced UI"""
    st.header("📈 Count History")
    
    # Get history from database
    history = db_manager.get_count_history()
    
    if not history:
        st.info("📝 No count history available yet. Upload and process some images to see your history here!")
        return
    
    # Display summary statistics
    st.subheader("📊 Summary Statistics")
    total_counts = len(history)
    avg_count = sum(row['count'] for row in history) / total_counts if total_counts > 0 else 0
    avg_confidence = sum(row['confidence'] for row in history) / total_counts if total_counts > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images Processed", total_counts)
    with col2:
        st.metric("Average Objects per Image", f"{avg_count:.1f}")
    with col3:
        st.metric("Average Confidence", f"{avg_confidence:.2f}")
    
    # Display history in a table
    st.subheader("📋 Recent Counts")
    st.dataframe(
        history,
        column_config={
            "filename": st.column_config.TextColumn("File Name", width="medium"),
            "count": st.column_config.NumberColumn("Object Count", format="%d"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f"),
            "object_type": st.column_config.TextColumn("Object Type", width="small"),
            "timestamp": st.column_config.DatetimeColumn("Date/Time", format="DD/MM/YYYY HH:mm")
        },
        hide_index=True,
        use_container_width=True
    )

def show_about_page():
    """Display the about page"""
    st.header("ℹ️ About CountAnything AI")
    
    st.markdown("""
    ### What is CountAnything AI?
    
    CountAnything AI is a powerful computer vision application that uses advanced threshold-based detection to automatically count objects in images. 
    Whether you need to count rice grains, coins, hair strands, or any other small objects, our AI can help!
    
    ### Features
    - 🔍 **Advanced Threshold Detection**: Uses morphological operations and contour analysis
    - 🎯 **Multiple Object Types**: Specialized detection for rice, coins, hair, and general objects
    - 📊 **Real-time Results**: Get instant counts with confidence scores
    - 📈 **History Tracking**: Keep track of all your counting sessions
    - 🎨 **Visual Overlays**: See exactly what was detected with color-coded confidence
    - ⚙️ **Adjustable Thresholds**: Fine-tune detection sensitivity
    
    ### How it Works
    1. **Upload**: Simply upload an image containing objects to count
    2. **Select**: Choose the type of objects you're counting
    3. **Adjust**: Set confidence threshold for detection sensitivity
    4. **Process**: Our AI analyzes the image using threshold-based detection
    5. **Results**: Get accurate counts with visual feedback
    
    ### Detection Method
    - **Threshold-based**: Uses adaptive thresholding and morphological operations
    - **Contour Analysis**: Identifies objects based on shape and size
    - **Circularity Check**: Filters objects based on roundness
    - **Area Filtering**: Removes objects that are too small or too large
    
    ### Use Cases
    - 🍚 **Cooking**: Count rice grains to estimate servings
    - 💰 **Business**: Count coins for inventory
    - 💇 **Health**: Count hair strands for density analysis
    - 🧮 **General**: Count any small objects quickly and accurately
    
    ### Technology Stack
    - **Frontend**: Streamlit for beautiful web interface
    - **Image Processing**: OpenCV for threshold-based detection
    - **Database**: SQLite for history storage
    - **Language**: Python 3.12
    
    ### Made with ❤️ for Hackathon
    This project was built to demonstrate the power of computer vision in solving everyday counting problems.
    """)

if __name__ == "__main__":
    main()
