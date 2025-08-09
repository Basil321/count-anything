# File Structure
The project is organized for clarity and simplicity, with an Instructions folder for documentation to guide Cursor AI usage.CountAnythingAI/
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