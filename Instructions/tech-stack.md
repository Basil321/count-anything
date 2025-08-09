# Tech Stack Overview
The project is primarily Python-based, focusing on backend computer vision with a lightweight Streamlit frontend for user interaction. This stack is chosen for rapid development with Cursor AI and hackathon compatibility.

## Frontend
- **Streamlit (1.39.0)**: A Python library for creating a simple, interactive web interface to upload images and display results. Ideal for quick hackathon demos.
- **Pillow (11.0.0)**: For rendering images in the Streamlit interface.

## Backend
- **Python (3.12)**: Core language for the project, handling all logic and processing.
- **OpenCV (4.10.0)**: For image preprocessing and object detection/counting tasks.
- **Ultralytics YOLOv8 (8.3.15)**: A pre-trained computer vision model for accurate object detection, fine-tunable for objects like rice or hair.
- **NumPy (2.1.2)**: For numerical operations in image processing.
- **SQLite (via Python’s sqlite3)**: Lightweight, serverless database for storing count history.

## Tools
- **Cursor AI**: For generating Python code, integrating libraries, and debugging.
- **Pip**: For managing Python dependencies.