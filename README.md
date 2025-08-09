# 🔢 CountAnything AI

A powerful computer vision application that uses AI to automatically count objects in images. Perfect for counting rice grains, coins, hair strands, or any small objects quickly and accurately.

## 🚀 Features

- **🔍 Advanced Object Detection**: Uses YOLOv8 for accurate object detection
- **🎯 Multiple Object Types**: Specialized detection for rice, coins, hair, and general objects
- **📊 Real-time Results**: Get instant counts with confidence scores
- **📈 History Tracking**: Keep track of all your counting sessions
- **🎨 Visual Overlays**: See exactly what was detected with bounding boxes
- **💻 Beautiful UI**: Modern Streamlit interface with dark theme support

## 🛠️ Technology Stack

- **Frontend**: Streamlit 1.39.0
- **AI Model**: YOLOv8 (Ultralytics)
- **Image Processing**: OpenCV 4.10.0
- **Database**: SQLite
- **Language**: Python 3.12

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd CountAnythingAI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app/main.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 🎯 How to Use

1. **Upload an Image**: Drag and drop or select an image containing objects to count
2. **Select Object Type**: Choose from rice, coins, hair, or general objects
3. **Adjust Settings**: Fine-tune confidence threshold if needed
4. **Process**: Click "Count Objects" to analyze the image
5. **View Results**: See the count, confidence score, and visual overlays
6. **Check History**: Review your previous counting sessions

## 📁 Project Structure

```
CountAnythingAI/
├── app/
│   ├── main.py                 # Streamlit app and main logic
│   ├── models/
│   │   ├── count_model.py     # YOLOv8 model integration
│   ├── utils/
│   │   ├── image_processor.py # Image preprocessing and counting logic
│   ├── database/
│   │   ├── db.py             # SQLite database setup
├── data/
│   ├── images/               # Folder for uploaded images
│   ├── history.db            # SQLite database for count history
├── Instructions/
│   ├── prd.md                # Product Requirements Document
│   ├── tech-stack.md         # Tech Stack Overview
│   ├── file-structure.md     # File Structure
│   ├── guidelines.md         # Development Guidelines
│   ├── implementation-plan.md # Implementation Plan
├── requirements.txt           # Python dependencies
├── README.md                 # Project overview and setup instructions
```

## 🎨 Use Cases

- **🍚 Cooking**: Count rice grains to estimate servings
- **💰 Business**: Count coins for inventory management
- **💇 Health**: Count hair strands for density analysis
- **🧮 General**: Count any small objects quickly and accurately

## 🔧 Configuration

### Object Types
- **Rice**: Optimized for small, round grains
- **Coins**: Designed for circular objects with metallic appearance
- **Hair**: Specialized for thin, elongated strands
- **General**: Universal detection for any small objects

### Confidence Threshold
Adjust the confidence threshold in the sidebar to control detection sensitivity:
- **Low (0.1-0.3)**: More detections, may include false positives
- **Medium (0.3-0.6)**: Balanced detection (recommended)
- **High (0.6-0.9)**: Fewer detections, higher accuracy

## 📊 Database

The application uses SQLite to store:
- **Count History**: All processed images and their results
- **Image Metadata**: File information and processing details
- **User Settings**: Preferences and configuration

## 🐛 Troubleshooting

### Common Issues

1. **YOLOv8 Model Not Loading**
   - The app will automatically download the model on first run
   - Ensure you have internet connection for model download
   - Check available disk space (model is ~6MB)

2. **Slow Processing**
   - Reduce image size before uploading
   - Lower confidence threshold for faster processing
   - Close other applications to free up memory

3. **No Objects Detected**
   - Try adjusting the confidence threshold
   - Ensure good lighting in the image
   - Select the appropriate object type
   - Try a different image with clearer objects

### Performance Tips

- **Image Quality**: Use clear, well-lit images for best results
- **Object Size**: Ensure objects are clearly visible and not too small
- **Background**: Simple backgrounds work better than complex ones
- **Lighting**: Even lighting without harsh shadows improves detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics**: For the amazing YOLOv8 model
- **Streamlit**: For the beautiful web framework
- **OpenCV**: For powerful image processing capabilities
- **Hackathon Community**: For inspiration and support

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the project documentation
3. Open an issue on GitHub
4. Contact the development team

---

**Made with ❤️ for Hackathon**

*CountAnything AI - Making object counting simple and accurate with AI*
