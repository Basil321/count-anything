"""
Database Module for CountAnything AI
Handles SQLite database operations for storing count history and image metadata.
"""

import sqlite3
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

class DatabaseManager:
    """Manages SQLite database operations for CountAnything AI"""
    
    def __init__(self, db_path: str = "data/history.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self._create_tables()
    
    def _ensure_db_directory(self):
        """Ensure the database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create count_history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS count_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        count INTEGER NOT NULL,
                        confidence REAL NOT NULL,
                        object_type TEXT DEFAULT 'general',
                        image_path TEXT,
                        processing_time REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create image_metadata table for additional image information
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS image_metadata (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        file_size INTEGER,
                        image_width INTEGER,
                        image_height INTEGER,
                        file_type TEXT,
                        upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create settings table for user preferences
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_settings (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        setting_key TEXT UNIQUE NOT NULL,
                        setting_value TEXT NOT NULL,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                print("Database tables created successfully")
                
        except Exception as e:
            print(f"Error creating database tables: {e}")
    
    def save_count_result(self, filename: str, count: int, confidence: float, 
                         object_type: str = "general", image_path: Optional[str] = None,
                         processing_time: Optional[float] = None) -> bool:
        """
        Save count result to database
        
        Args:
            filename: Name of the uploaded file
            count: Number of objects detected
            confidence: Detection confidence score
            object_type: Type of objects detected
            image_path: Path to saved image (optional)
            processing_time: Time taken to process image (optional)
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO count_history 
                    (filename, count, confidence, object_type, image_path, processing_time)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (filename, count, confidence, object_type, image_path, processing_time))
                
                conn.commit()
                print(f"Count result saved: {filename} - {count} objects")
                return True
                
        except Exception as e:
            print(f"Error saving count result: {e}")
            return False
    
    def save_image_metadata(self, filename: str, file_size: int, image_width: int, 
                           image_height: int, file_type: str) -> bool:
        """
        Save image metadata to database
        
        Args:
            filename: Name of the uploaded file
            file_size: Size of the file in bytes
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels
            file_type: Type of the image file
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO image_metadata 
                    (filename, file_size, image_width, image_height, file_type)
                    VALUES (?, ?, ?, ?, ?)
                """, (filename, file_size, image_width, image_height, file_type))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving image metadata: {e}")
            return False
    
    def get_count_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get count history from database
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of count history records
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        id,
                        filename,
                        count,
                        confidence,
                        object_type,
                        processing_time,
                        timestamp
                    FROM count_history 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            print(f"Error getting count history: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total records
                cursor.execute("SELECT COUNT(*) FROM count_history")
                total_records = cursor.fetchone()[0]
                
                # Average count
                cursor.execute("SELECT AVG(count) FROM count_history")
                avg_count = cursor.fetchone()[0] or 0
                
                # Average confidence
                cursor.execute("SELECT AVG(confidence) FROM count_history")
                avg_confidence = cursor.fetchone()[0] or 0
                
                # Most common object type
                cursor.execute("""
                    SELECT object_type, COUNT(*) as count 
                    FROM count_history 
                    GROUP BY object_type 
                    ORDER BY count DESC 
                    LIMIT 1
                """)
                most_common_type = cursor.fetchone()
                top_object_type = most_common_type[0] if most_common_type else "general"
                
                # Recent activity (last 7 days)
                cursor.execute("""
                    SELECT COUNT(*) FROM count_history 
                    WHERE timestamp >= datetime('now', '-7 days')
                """)
                recent_activity = cursor.fetchone()[0]
                
                return {
                    'total_records': total_records,
                    'avg_count': round(avg_count, 2),
                    'avg_confidence': round(avg_confidence, 2),
                    'top_object_type': top_object_type,
                    'recent_activity': recent_activity
                }
                
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {
                'total_records': 0,
                'avg_count': 0,
                'avg_confidence': 0,
                'top_object_type': 'general',
                'recent_activity': 0
            }
    
    def save_user_setting(self, key: str, value: str) -> bool:
        """
        Save user setting to database
        
        Args:
            key: Setting key
            value: Setting value
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO user_settings (setting_key, setting_value)
                    VALUES (?, ?)
                """, (key, value))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error saving user setting: {e}")
            return False
    
    def get_user_setting(self, key: str, default: str = "") -> str:
        """
        Get user setting from database
        
        Args:
            key: Setting key
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT setting_value FROM user_settings WHERE setting_key = ?
                """, (key,))
                
                result = cursor.fetchone()
                return result[0] if result else default
                
        except Exception as e:
            print(f"Error getting user setting: {e}")
            return default
    
    def clear_history(self) -> bool:
        """
        Clear all count history
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM count_history")
                cursor.execute("DELETE FROM image_metadata")
                
                conn.commit()
                print("Count history cleared successfully")
                return True
                
        except Exception as e:
            print(f"Error clearing history: {e}")
            return False
    
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information
        
        Returns:
            Dictionary with database information
        """
        try:
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get table sizes
                cursor.execute("SELECT COUNT(*) FROM count_history")
                history_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM image_metadata")
                metadata_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM user_settings")
                settings_count = cursor.fetchone()[0]
                
                return {
                    'db_size_mb': round(db_size / (1024 * 1024), 2),
                    'history_records': history_count,
                    'metadata_records': metadata_count,
                    'settings_records': settings_count,
                    'db_path': self.db_path
                }
                
        except Exception as e:
            print(f"Error getting database info: {e}")
            return {
                'db_size_mb': 0,
                'history_records': 0,
                'metadata_records': 0,
                'settings_records': 0,
                'db_path': self.db_path
            }
