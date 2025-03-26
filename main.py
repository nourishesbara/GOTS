import sys
import os
import numpy as np
import sqlite3
import random
import nltk
import re
import hashlib
import cv2
import pytesseract
from datetime import datetime
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QMessageBox, QFileDialog,
                             QRadioButton, QButtonGroup, QProgressBar, QTableWidget,
                             QTableWidgetItem, QTextEdit, QGroupBox, QGridLayout, QStackedWidget)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, pyqtSignal


def resource_path(relative_path):
    try:
        base_dir = sys._MEIPASS
    except Exception:
        base_dir = os.path.abspath(".")

    return os.path.join(base_dir, relative_path)

# Database setup
if getattr(sys, 'frozen', False):
    APP_DIR = sys._MEIPASS
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(APP_DIR, "textquiz.db")  # Get the full path

"""Create the database tables if they don't exist"""
def create_database():

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Create quiz_results table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS quiz_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        extracted_text TEXT,
        score INTEGER,
        total_questions INTEGER,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')

    # Create quiz_questions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS quiz_questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        quiz_id INTEGER,
        question TEXT,
        correct_answer TEXT,
        options TEXT,
        user_answer TEXT,
        FOREIGN KEY (quiz_id) REFERENCES quiz_results (id)
    )
    ''')

    conn.commit()
    conn.close()


"""Clean and preprocess the extracted text"""
def preprocess_text(text):

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

"""Extract important sentences from the text"""
def extract_key_sentences(text, num_sentences=20):

    sentences = nltk.sent_tokenize(text)

    # Filter out very short sentences
    sentences = [s for s in sentences if len(s.split()) > 5]

    # If we don't have enough sentences, return what we have
    if len(sentences) <= num_sentences:
        return sentences

    # Otherwise, select a sample of sentences
    return random.sample(sentences, num_sentences)

"""Generate an MCQ question from a sentence"""
def generate_quiz_question(sentence):

    # Tokenize the sentence
    tokens = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokens)

    # Find nouns, verbs, or adjectives that can be used for questions
    candidates = [word for word, tag in tagged if tag.startswith(('NN', 'VB', 'JJ')) and len(word) > 3]

    if not candidates:
        return None

    # Select a random candidate word
    word_to_replace = random.choice(candidates)

    # Create the question by replacing the word with a blank
    question = sentence.replace(word_to_replace, "_______")

    # Generate incorrect options
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    # Get all unique words from the text that aren't the answer or stop words
    all_words = [word for word, tag in tagged if word.lower() not in stop_words and word != word_to_replace]

    # If we don't have enough words, add some common words
    common_words = ["example", "information", "knowledge", "important", "different", "process", "system"]

    # Generate 3 incorrect options
    incorrect_options = []
    while len(incorrect_options) < 3:
        if all_words:
            option = random.choice(all_words)
            if option not in incorrect_options:
                incorrect_options.append(option)
        elif common_words:
            option = random.choice(common_words)
            common_words.remove(option)
            if option not in incorrect_options:
                incorrect_options.append(option)
        else:
            # Generate a random word if we run out of options
            incorrect_options.append(f"option{len(incorrect_options) + 1}")

    # Combine correct and incorrect options
    options = [word_to_replace] + incorrect_options
    random.shuffle(options)

    return {
        "question": question,
        "correct_answer": word_to_replace,
        "options": options
    }

"""Generate a quiz based on the extracted text"""
def generate_quiz(text, num_questions=5):

    # Preprocess the text
    cleaned_text = preprocess_text(text)

    # Extract key sentences
    key_sentences = extract_key_sentences(cleaned_text)

    # Generate questions from the sentences
    questions = []
    attempts = 0
    max_attempts = min(30, len(key_sentences) * 2)  # Limit attempts to avoid infinite loops

    while len(questions) < num_questions and attempts < max_attempts:
        if not key_sentences:
            break

        sentence = random.choice(key_sentences)
        # Don't reuse the same sentence
        key_sentences.remove(sentence)
        question = generate_quiz_question(sentence)
        if question:
            questions.append(question)

        attempts += 1

    return questions



# Custom Widgets
"""Custom QPushButton with rounded corners"""
class RoundedButton(QPushButton):

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMinimumHeight(40)
        self.setMaximumWidth(150)  # Added maximum width constraint
        self.setStyleSheet("""
            QPushButton {
                background-color: #6FA3EF;  /* Light Blue */
                color: white;
                border-radius: 8px;
                padding: 10px 14px;
                font-weight: bold;
                font-size: 14px;
                border: none;
                transition: background-color 0.3s, transform 0.1s;
            }
            QPushButton:hover {
                background-color: #5A9BEF; /* Slightly darker blue */
                transform: scale(1.05);
            }
            QPushButton:pressed {
                background-color: #4A86E8; /* Slightly darker blue */
                transform: scale(0.98);
            }
            QPushButton:disabled {
                background-color: #A0A0A0;
                color: #E0E0E0;
            }
        """)


"""Login window for the application"""
class LoginWindow(QWidget):

    login_successful = pyqtSignal(int, str)
    register_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("TextQuiz - Login")
        self.resize(400, 500)  # Slightly increased height for better spacing

        # Create main layout
        main_layout = QVBoxLayout()
        # Add padding around the entire layout
        main_layout.setContentsMargins(40, 20, 40, 20)

        title_label = QLabel("TextQuiz")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 48px; 
            font-weight: bold; 
            margin: 20px 0;
            color: #333;
        """)

        # Create a centered form container
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(15)  # Consistent spacing between form elements
        form_layout.setContentsMargins(0, 0, 0, 0)

        # Username section
        username_label = QLabel("Username")
        username_label.setStyleSheet("font-weight: bold; color: #555;")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("username")
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)

        # Password section
        password_label = QLabel("Password")
        password_label.setStyleSheet("font-weight: bold; color: #555;")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)

        # Login button
        self.login_button = RoundedButton("Login")
        self.login_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.login_button.clicked.connect(self.login)

        # Register button
        self.register_button = QPushButton("Don't have an account? Register")
        self.register_button.setStyleSheet("""
            QPushButton {
                border: none; 
                color: #2196F3;
                background-color: transparent;
            }
            QPushButton:hover {
                text-decoration: underline;
            }
        """)
        self.register_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.register_button.clicked.connect(self.register)

        # Add widgets to form layout
        form_layout.addWidget(username_label)
        form_layout.addWidget(self.username_input)
        form_layout.addWidget(password_label)
        form_layout.addWidget(self.password_input)
        form_layout.addWidget(self.login_button)

        # Create a horizontal layout to center the form
        center_layout = QHBoxLayout()
        center_layout.addStretch(1)
        center_layout.addWidget(form_container)
        center_layout.addStretch(1)

        # Add layouts to main layout
        main_layout.addWidget(title_label)
        main_layout.addLayout(center_layout)
        main_layout.addWidget(self.register_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Set the main layout
        self.setLayout(main_layout)

    def login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "Login Error", "Please enter both username and password.")
            return

        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Check credentials in database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?",
                       (username, hashed_password))
        user = cursor.fetchone()

        conn.close()

        if user:
            self.login_successful.emit(user[0], username)
        else:
            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")

    def register(self):
        self.register_requested.emit()


"""Registration window for new users"""
class RegisterWindow(QWidget):
    register_successful = pyqtSignal()
    back_to_login = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("TextQuiz - Register")
        self.resize(400, 500)  # Slightly increased height for better spacing

        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 20, 40, 20)  # Add padding around the entire layout

        # Title label
        title_label = QLabel("Create Account")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("""
            font-size: 36px; 
            font-weight: bold; 
            margin: 20px 0;
            color: #333;
        """)

        # Create a centered form container
        form_container = QWidget()
        form_layout = QVBoxLayout(form_container)
        form_layout.setSpacing(15)  # Consistent spacing between form elements
        form_layout.setContentsMargins(0, 0, 0, 0)

        # Username section
        username_label = QLabel("Username")
        username_label.setStyleSheet("font-weight: bold; color: #555;")
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Choose a username")
        self.username_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)

        # Password section
        password_label = QLabel("Password")
        password_label.setStyleSheet("font-weight: bold; color: #555;")
        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("password")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)

        # Confirm Password section
        confirm_password_label = QLabel("Confirm Password")
        confirm_password_label.setStyleSheet("font-weight: bold; color: #555;")
        self.confirm_password_input = QLineEdit()
        self.confirm_password_input.setPlaceholderText("Confirm your password")
        self.confirm_password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_password_input.setStyleSheet("""
            QLineEdit {
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)

        # Register button
        self.register_button = RoundedButton("Register")
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.register_button.clicked.connect(self.register)

        # Back to login button
        self.back_button = QPushButton("Already have an account? Login")
        self.back_button.setStyleSheet("""
            QPushButton {
                border: none; 
                color: #2196F3;
                background-color: transparent;
            }
            QPushButton:hover {
                text-decoration: underline;
            }
        """)
        self.back_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.back_button.clicked.connect(self.go_back)

        # Add widgets to form layout
        form_layout.addWidget(username_label)
        form_layout.addWidget(self.username_input)
        form_layout.addWidget(password_label)
        form_layout.addWidget(self.password_input)
        form_layout.addWidget(confirm_password_label)
        form_layout.addWidget(self.confirm_password_input)
        form_layout.addWidget(self.register_button)

        # Create a horizontal layout to center the form
        center_layout = QHBoxLayout()
        center_layout.addStretch(1)
        center_layout.addWidget(form_container)
        center_layout.addStretch(1)

        # Add layouts to main layout
        main_layout.addWidget(title_label)
        main_layout.addLayout(center_layout)
        main_layout.addWidget(self.back_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Set the main layout
        self.setLayout(main_layout)

    def register(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()
        confirm_password = self.confirm_password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "Registration Error", "Please fill all fields.")
            return

        if password != confirm_password:
            QMessageBox.warning(self, "Registration Error", "Passwords do not match.")
            return

        # Check if username already exists
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            QMessageBox.warning(self, "Registration Error", "Username already exists.")
            conn.close()
            return

        # Hash the password
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Insert new user
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                           (username, hashed_password))
            conn.commit()
            QMessageBox.information(self, "Registration Successful",
                                    "Your account has been created successfully.")
            self.register_successful.emit()
        except sqlite3.Error as e:
            QMessageBox.critical(self, "Database Error", f"Error creating account: {str(e)}")
        finally:
            conn.close()

    def go_back(self):
        self.back_to_login.emit()


class MainMenuWindow(QWidget):
    """Main menu window for the application"""
    logout_requested = pyqtSignal()
    start_quiz_requested = pyqtSignal()
    view_history_requested = pyqtSignal()

    def __init__(self, user_id, username, parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.username = username
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("TextQuiz - Main Menu")
        self.resize(600, 450)  # Slightly increased window size to accommodate larger buttons

        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 20, 40, 20)

        # Welcome message
        welcome_label = QLabel(f"Welcome, {self.username}!")
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_label.setStyleSheet("""
            font-size: 28px; 
            font-weight: bold; 
            margin: 20px 0;
            color: #333;
        """)

        # Container for main buttons
        buttons_container = QWidget()
        buttons_layout = QVBoxLayout(buttons_container)
        buttons_layout.setSpacing(25)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        # Horizontal layout for Start Quiz and View History
        quiz_buttons_layout = QHBoxLayout()
        quiz_buttons_layout.setSpacing(25)

        # Start Quiz button
        self.start_quiz_button = RoundedButton("New Quiz")
        self.start_quiz_button.setMinimumHeight(80)  # Increased height
        self.start_quiz_button.setMinimumWidth(200)  # Added width
        self.start_quiz_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.start_quiz_button.clicked.connect(self.start_quiz)

        # View History button
        self.view_history_button = RoundedButton("Quiz History")
        self.view_history_button.setMinimumHeight(80)  # Increased height
        self.view_history_button.setMinimumWidth(200)  # Added width
        self.view_history_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 18px;
                font-weight: bold;
                padding: 15px;
            }
            QPushButton:hover {
                background-color: #1E88E5;
            }
        """)
        self.view_history_button.clicked.connect(self.view_history)

        # Add quiz buttons to horizontal layout
        quiz_buttons_layout.addWidget(self.start_quiz_button)
        quiz_buttons_layout.addWidget(self.view_history_button)

        # Logout button
        self.logout_button = RoundedButton("Logout")
        self.logout_button.setMinimumHeight(60)
        self.logout_button.setMinimumWidth(250)
        self.logout_button.setStyleSheet("""
            QPushButton {
                color: #d9534f;
                border: 2px solid #d9534f;
                border-radius: 10px;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #d9534f;
                color: white;
            }
        """)
        self.logout_button.clicked.connect(self.logout)

        # Create a horizontal layout to center the logout button
        logout_center_layout = QHBoxLayout()
        logout_center_layout.addStretch(1)
        logout_center_layout.addWidget(self.logout_button)
        logout_center_layout.addStretch(1)

        # Add layouts to buttons container
        buttons_layout.addLayout(quiz_buttons_layout)
        buttons_layout.addLayout(logout_center_layout)

        # Create a horizontal layout to center the buttons container
        center_layout = QHBoxLayout()
        center_layout.addStretch(1)
        center_layout.addWidget(buttons_container)
        center_layout.addStretch(1)

        # Add widgets to main layout
        main_layout.addWidget(welcome_label)
        main_layout.addLayout(center_layout)

        self.setLayout(main_layout)
    def start_quiz(self):
        self.start_quiz_requested.emit()

    def view_history(self):
        self.view_history_requested.emit()

    def logout(self):
        reply = QMessageBox.question(self, 'Logout', 'Are you sure you want to logout?',
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                     QMessageBox.StandardButton.No)

        if reply == QMessageBox.StandardButton.Yes:
            self.logout_requested.emit()


class ImageProcessingWindow(QWidget):
    """Window for uploading and processing images"""
    quiz_ready = pyqtSignal(str)
    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = None
        self.extracted_text = ""
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("TextQuiz - Image Processing")
        self.resize(700, 550)

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Text Quiz")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 30px; font-weight: bold; margin: 10px;")

        # Image display area
        self.image_label = QLabel("No image selected")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("border: 2px dashed #aaa; padding: 20px; background-color: #f8f9fa;")
        self.image_label.setMinimumHeight(250)

        # Upload button
        self.upload_button = RoundedButton("Upload Image")
        self.upload_button.clicked.connect(self.upload_image)

        # Process button
        self.process_button = RoundedButton("Process Image")
        self.process_button.clicked.connect(self.process_image)
        self.process_button.setEnabled(False)

        # Text preview
        text_preview_label = QLabel("Extracted Text:")
        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        self.text_preview.setMinimumHeight(100)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # Continue button
        self.continue_button = RoundedButton("Start Quiz")
        self.continue_button.clicked.connect(self.start_quiz)
        self.continue_button.setEnabled(False)

        # Back button
        self.back_button = RoundedButton("Main Menu")
        self.back_button.clicked.connect(self.go_back)

        # Add widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.image_label)

        # Buttons layout
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.upload_button)
        buttons_layout.addWidget(self.process_button)
        main_layout.addLayout(buttons_layout)

        main_layout.addWidget(text_preview_label)
        main_layout.addWidget(self.text_preview)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.continue_button)
        main_layout.addWidget(self.back_button)

        self.setLayout(main_layout)

    def upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            self.image_path = file_path

            # Display image
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # Scale pixmap to fit the label while maintaining aspect ratio
                pixmap = pixmap.scaled(
                    self.image_label.width(), self.image_label.height(),
                    Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                self.image_label.setPixmap(pixmap)
                self.process_button.setEnabled(True)
            else:
                QMessageBox.warning(self, "Image Error", "Failed to load the image.")
                self.image_label.setText("No image selected")
                self.process_button.setEnabled(False)

    def process_image(self):
        if not self.image_path:
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        try:
            # Load image with multiple libraries for robustness
            image = cv2.imread(self.image_path)
            if image is None:
                raise Exception("Failed to load image with OpenCV")

            self.progress_bar.setValue(10)

            # Advanced preprocessing techniques
            # 1. Resize image if too large to improve OCR performance
            height, width = image.shape[:2]
            if max(height, width) > 2000:
                scale_factor = 2000 / max(height, width)
                image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor,
                                   interpolation=cv2.INTER_AREA)

            self.progress_bar.setValue(20)

            # 2. Convert to grayscale with adaptive methods
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 3. Advanced noise reduction
            # Bilateral filtering to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(gray, 9, 75, 75)

            self.progress_bar.setValue(30)

            # 4. Multiple thresholding techniques
            # Adaptive thresholding for better performance on varying backgrounds
            adaptive_thresh = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )

            self.progress_bar.setValue(40)

            # 5. Deskew the image (correct text rotation)
            coords = np.column_stack(np.where(adaptive_thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]

            # Correct the angle of rotation
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            (h, w) = adaptive_thresh.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                adaptive_thresh,
                M,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

            self.progress_bar.setValue(50)

            # 6. Morphological operations for text enhancement
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            processed = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

            self.progress_bar.setValue(60)

            # 7. Multiple OCR engines with fallback
            # Primary: Tesseract with custom configuration
            custom_config = r'--oem 3 --psm 6'
            extracted_text = pytesseract.image_to_string(processed, config=custom_config)

            # Fallback: easyOCR for additional robustness
            if not extracted_text.strip():
                reader = easyocr.Reader(['en'])
                results = reader.readtext(image)
                extracted_text = ' '.join([result[1] for result in results])

            self.progress_bar.setValue(80)

            # Clean and process extracted text
            self.extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()

            # Update UI
            self.text_preview.setText(self.extracted_text)
            self.progress_bar.setValue(100)

            # Enable continue button if text was extracted
            if self.extracted_text:
                self.continue_button.setEnabled(True)
                QMessageBox.information(self, "Processing Complete", "Text extraction completed successfully.")
            else:
                QMessageBox.warning(self, "Processing Warning",
                                    "No text was extracted from the image. Try another image or adjust the image quality.")

        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Processing Error", f"Error processing image: {str(e)}")

        finally:
            self.progress_bar.setVisible(False)

    def start_quiz(self):
        if self.extracted_text.strip():
            self.quiz_ready.emit(self.extracted_text)
        else:
            QMessageBox.warning(self, "Error", "No text available for quiz generation.")

    def go_back(self):
        self.back_requested.emit()


class QuizWindow(QWidget):
    """Window for taking the quiz"""
    quiz_completed = pyqtSignal(list, str)
    back_requested = pyqtSignal()

    def __init__(self, extracted_text, parent=None):
        super().__init__(parent)
        self.extracted_text = extracted_text
        self.questions = []
        self.current_question = 0
        self.selected_answers = []
        self.init_ui()
        self.generate_quiz()

    def init_ui(self):
        self.setWindowTitle("TextQuiz - Quiz")
        self.resize(700, 500)

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Quiz")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; margin: 10px;")

        # Progress indicator
        self.progress_label = QLabel("Question 0/0")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Question display
        self.question_label = QLabel("Generating questions...")
        self.question_label.setWordWrap(True)
        self.question_label.setStyleSheet("font-size: 16px; margin: 15px 0;")

        # Options group
        self.options_group = QGroupBox("Select your answer:")
        self.options_layout = QVBoxLayout()
        self.options_group.setLayout(self.options_layout)

        # Radio buttons for options
        self.option_group = QButtonGroup(self)
        self.option_buttons = []

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_question)
        self.prev_button.setEnabled(False)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_question)

        self.submit_button = RoundedButton("Submit Quiz")
        self.submit_button.clicked.connect(self.submit_quiz)
        self.submit_button.setVisible(False)

        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        # Back button
        self.back_button = RoundedButton("Main Menu")
        self.back_button.clicked.connect(self.confirm_back)

        # Add widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.progress_label)
        main_layout.addWidget(self.question_label)
        main_layout.addWidget(self.options_group)
        main_layout.addLayout(nav_layout)
        main_layout.addWidget(self.submit_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.back_button)

        self.setLayout(main_layout)

    def generate_quiz(self):
        # Show progress or loading indicator
        self.question_label.setText("Generating questions... Please wait.")
        QApplication.processEvents()

        # Generate quiz questions
        try:
            self.questions = generate_quiz(self.extracted_text, num_questions=5)

            if not self.questions:
                QMessageBox.warning(self, "Quiz Generation Error",
                                    "Failed to generate quiz questions from the extracted text. "
                                    "The text might be too short or not contain enough meaningful content.")
                self.back_requested.emit()
                return

            # Initialize selected answers list
            self.selected_answers = [None] * len(self.questions)

            # Display first question
            self.current_question = 0
            self.display_question()

        except Exception as e:
            QMessageBox.critical(self, "Quiz Generation Error",
                                 f"An error occurred while generating the quiz: {str(e)}")
            self.back_requested.emit()

    def display_question(self):
        # Clear previous options
        while self.options_layout.count():
            item = self.options_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.option_buttons.clear()
        self.option_group = QButtonGroup(self)

        # Update progress label
        self.progress_label.setText(f"Question {self.current_question + 1}/{len(self.questions)}")

        # Get current question
        question_data = self.questions[self.current_question]

        # Set question text
        self.question_label.setText(question_data["question"])

        # Add options
        for i, option in enumerate(question_data["options"]):
            radio = QRadioButton(option)
            self.option_buttons.append(radio)
            self.option_group.addButton(radio, i)
            self.options_layout.addWidget(radio)

        # Restore previous selection if any
        if self.selected_answers[self.current_question] is not None:
            selected_idx = question_data["options"].index(self.selected_answers[self.current_question])
            if 0 <= selected_idx < len(self.option_buttons):
                self.option_buttons[selected_idx].setChecked(True)

        # Update navigation buttons
        self.prev_button.setEnabled(self.current_question > 0)

        if self.current_question == len(self.questions) - 1:
            self.next_button.setVisible(False)
            self.submit_button.setVisible(True)
        else:
            self.next_button.setVisible(True)
            self.submit_button.setVisible(False)

    def next_question(self):
        # Save current answer
        selected_button = self.option_group.checkedButton()
        if selected_button:
            selected_idx = self.option_group.id(selected_button)
            self.selected_answers[self.current_question] = self.questions[self.current_question]["options"][
                selected_idx]

        # Move to next question
        if self.current_question < len(self.questions) - 1:
            self.current_question += 1
            self.display_question()

    def prev_question(self):
        # Save current answer
        selected_button = self.option_group.checkedButton()
        if selected_button:
            selected_idx = self.option_group.id(selected_button)
            self.selected_answers[self.current_question] = self.questions[self.current_question]["options"][
                selected_idx]

        # Move to previous question
        if self.current_question > 0:
            self.current_question -= 1
            self.display_question()

    def submit_quiz(self):
        # Save answer for the last question
        selected_button = self.option_group.checkedButton()
        if selected_button:
            selected_idx = self.option_group.id(selected_button)
            self.selected_answers[self.current_question] = self.questions[self.current_question]["options"][
                selected_idx]

        # Check if all questions are answered
        if None in self.selected_answers:
            unanswered = self.selected_answers.count(None)
            reply = QMessageBox.question(
                self,
                "Incomplete Quiz",
                f"You have {unanswered} unanswered question(s). Do you want to submit anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                # Find the first unanswered question
                self.current_question = self.selected_answers.index(None)
                self.display_question()
                return

        # Build result data with questions and answers
        result_data = []
        for i, question_data in enumerate(self.questions):
            result_data.append({
                "question": question_data["question"],
                "correct_answer": question_data["correct_answer"],
                "options": question_data["options"],
                "user_answer": self.selected_answers[i] if i < len(self.selected_answers) else None
            })

        # Emit signal with quiz results
        self.quiz_completed.emit(result_data, self.extracted_text)

    def confirm_back(self):
        reply = QMessageBox.question(
            self,
            "Leave Quiz",
            "Are you sure you want to leave? Your progress will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.back_requested.emit()


class ResultsWindow(QWidget):
    """Window for displaying quiz results"""
    new_quiz_requested = pyqtSignal()
    back_to_menu_requested = pyqtSignal()

    def __init__(self, results, extracted_text, user_id, parent=None):
        super().__init__(parent)
        self.results = results
        self.extracted_text = extracted_text
        self.user_id = user_id
        self.init_ui()
        self.calculate_score()
        self.save_results()

    def init_ui(self):
        self.setWindowTitle("TextQuiz - Results")
        self.resize(700, 550)

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Quiz Results")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; margin: 10px;")

        # Score display
        self.score_label = QLabel()
        self.score_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.score_label.setStyleSheet("font-size: 18px; margin: 15px; font-weight: bold;")

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["Question", "Your Answer", "Correct Answer", "Result"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setWordWrap(True)

        # Resize table columns
        self.results_table.setColumnWidth(0, 280)  # Question column
        self.results_table.setColumnWidth(1, 120)  # Your Answer column
        self.results_table.setColumnWidth(2, 120)  # Correct Answer column

        # Action buttons
        buttons_layout = QHBoxLayout()

        self.new_quiz_button = RoundedButton("New Quiz")
        self.new_quiz_button.clicked.connect(self.new_quiz)

        self.menu_button = RoundedButton("Main Menu")
        self.menu_button.clicked.connect(self.back_to_menu)

        buttons_layout.addWidget(self.new_quiz_button)
        buttons_layout.addWidget(self.menu_button)

        # Add widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.score_label)
        main_layout.addWidget(self.results_table)
        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def calculate_score(self):
        correct_count = 0

        # Set table rows
        self.results_table.setRowCount(len(self.results))

        for i, result in enumerate(self.results):
            # Question text
            question_item = QTableWidgetItem(result["question"])
            question_item.setFlags(question_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.results_table.setItem(i, 0, question_item)

            # User answer
            user_answer = result["user_answer"] if result["user_answer"] else "No answer"
            user_answer_item = QTableWidgetItem(user_answer)
            user_answer_item.setFlags(user_answer_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.results_table.setItem(i, 1, user_answer_item)

            # Correct answer
            correct_answer_item = QTableWidgetItem(result["correct_answer"])
            correct_answer_item.setFlags(correct_answer_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.results_table.setItem(i, 2, correct_answer_item)

            # Result
            is_correct = result["user_answer"] == result["correct_answer"]
            if is_correct:
                correct_count += 1
                result_text = "Correct"
                result_color = "green"
            else:
                result_text = "Incorrect"
                result_color = "red"

            result_item = QTableWidgetItem(result_text)
            result_item.setFlags(result_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            result_item.setForeground(Qt.GlobalColor.green if is_correct else Qt.GlobalColor.red)
            self.results_table.setItem(i, 3, result_item)

        # Adjust row heights for word wrap
        for i in range(len(self.results)):
            self.results_table.resizeRowToContents(i)

        # Display score
        total_questions = len(self.results)
        self.score = correct_count
        self.total = total_questions

        # Show score label
        score_percent = (correct_count / total_questions) * 100 if total_questions > 0 else 0
        self.score_label.setText(f"Your Score: {correct_count}/{total_questions} ({score_percent:.1f}%)")

    def save_results(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            # Save quiz result
            cursor.execute(
                "INSERT INTO quiz_results (user_id, extracted_text, score, total_questions) VALUES (?, ?, ?, ?)",
                (self.user_id, self.extracted_text[:500], self.score, self.total)
            )
            quiz_id = cursor.lastrowid

            # Save individual questions
            for result in self.results:
                cursor.execute(
                    "INSERT INTO quiz_questions (quiz_id, question, correct_answer, options, user_answer) VALUES (?, ?, ?, ?, ?)",
                    (
                        quiz_id,
                        result["question"],
                        result["correct_answer"],
                        ','.join(result["options"]),
                        result["user_answer"] if result["user_answer"] else ""
                    )
                )

            conn.commit()
        except sqlite3.Error as e:
            QMessageBox.warning(self, "Database Error", f"Failed to save quiz results: {str(e)}")
        finally:
            conn.close()

    def new_quiz(self):
        self.new_quiz_requested.emit()

    def back_to_menu(self):
        self.back_to_menu_requested.emit()


class HistoryWindow(QWidget):
    """Window for displaying quiz history"""
    back_requested = pyqtSignal()
    view_details_requested = pyqtSignal(int)

    def __init__(self, user_id, parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.init_ui()
        self.load_history()

    def init_ui(self):
        self.setWindowTitle("TextQuiz - Quiz History")
        self.resize(700, 500)

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Quiz History")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; margin: 10px;")

        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(3)
        self.history_table.setHorizontalHeaderLabels(["Date", "Score", "questions"])
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.history_table.horizontalHeader().setStretchLastSection(True)

        # Set column widths
        self.history_table.setColumnWidth(0, 170)  # Date column
        self.history_table.setColumnWidth(1, 100)  # Score column
        self.history_table.setColumnWidth(2, 400)  # Text Preview column

        # Back button
        self.back_button = RoundedButton("Main Menu")
        self.back_button.clicked.connect(self.go_back)

        # Add widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.history_table)
        main_layout.addWidget(self.back_button)

        self.setLayout(main_layout)

    def load_history(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT id, date, score, total_questions, extracted_text FROM quiz_results WHERE user_id = ? ORDER BY date DESC",
                (self.user_id,)
            )
            results = cursor.fetchall()

            # Set table rows
            self.history_table.setRowCount(len(results))

            for i, (quiz_id, date, score, total, text) in enumerate(results):
                # Date
                date_item = QTableWidgetItem(datetime.fromisoformat(date).strftime("%Y-%m-%d %H:%M"))
                self.history_table.setItem(i, 0, date_item)

                # Score
                score_percent = (score / total) * 100 if total > 0 else 0
                score_item = QTableWidgetItem(f"{score}/{total}")
                self.history_table.setItem(i, 1, score_item)

                # Text preview
                preview = text[:50] + "..." if len(text) > 50 else text
                text_item = QTableWidgetItem(preview)
                self.history_table.setItem(i, 2, text_item)

                # View details button
                view_btn = QPushButton("Details")
                view_btn.clicked.connect(lambda checked, qid=quiz_id: self.view_details(qid))
                self.history_table.setCellWidget(i, 3, view_btn)

            # Adjust row heights
            for i in range(len(results)):
                self.history_table.resizeRowToContents(i)

        except sqlite3.Error as e:
            QMessageBox.warning(self, "Database Error", f"Failed to load quiz history: {str(e)}")
        finally:
            conn.close()

    def view_details(self, quiz_id):
        self.view_details_requested.emit(quiz_id)

    def go_back(self):
        self.back_requested.emit()


class QuizDetailsWindow(QWidget):
    """Window for displaying details of a specific quiz"""
    back_requested = pyqtSignal()

    def __init__(self, quiz_id, parent=None):
        super().__init__(parent)
        self.quiz_id = quiz_id
        self.init_ui()
        self.load_details()

    def init_ui(self):
        self.setWindowTitle("TextQuiz - Quiz Details")
        self.resize(700, 550)

        # Main layout
        main_layout = QVBoxLayout()

        # Title
        title_label = QLabel("Quiz Details")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 22px; font-weight: bold; margin: 10px;")

        # Quiz metadata
        self.meta_label = QLabel()
        self.meta_label.setStyleSheet("font-size: 16px; margin: 10px;")

        # Text preview
        text_group = QGroupBox("Extracted Text")
        text_layout = QVBoxLayout()
        self.text_preview = QTextEdit()
        self.text_preview.setReadOnly(True)
        text_layout.addWidget(self.text_preview)
        text_group.setLayout(text_layout)

        # Questions and answers
        questions_group = QGroupBox("Questions and Answers")
        self.questions_layout = QVBoxLayout()
        questions_group.setLayout(self.questions_layout)

        # Back button
        self.back_button =RoundedButton("History")
        self.back_button.clicked.connect(self.go_back)

        # Add widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(self.meta_label)
        main_layout.addWidget(text_group)
        main_layout.addWidget(questions_group)
        main_layout.addWidget(self.back_button)

        self.setLayout(main_layout)

    def load_details(self):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        try:
            # Get quiz metadata
            cursor.execute(
                "SELECT date, score, total_questions, extracted_text FROM quiz_results WHERE id = ?",
                (self.quiz_id,)
            )
            result = cursor.fetchone()

            if not result:
                QMessageBox.warning(self, "Error", "Quiz details not found.")
                self.go_back()
                return

            date, score, total, text = result

            # Display metadata
            date_str = datetime.fromisoformat(date).strftime("%Y-%m-%d %H:%M")
            score_percent = (score / total) * 100 if total > 0 else 0
            self.meta_label.setText(f"Date: {date_str}  |  Score: {score}/{total} ({score_percent:.1f}%)")

            # Display text
            self.text_preview.setText(text)

            # Get questions
            cursor.execute(
                "SELECT question, correct_answer, options, user_answer FROM quiz_questions WHERE quiz_id = ?",
                (self.quiz_id,)
            )
            questions = cursor.fetchall()

            # Display questions
            for i, (question, correct, options_str, user_answer) in enumerate(questions):
                options = options_str.split(',')

                # Create question group
                q_group = QGroupBox(f"Question {i + 1}")
                q_layout = QVBoxLayout()

                # Question text
                q_label = QLabel(question)
                q_label.setWordWrap(True)
                q_layout.addWidget(q_label)

                # Options
                options_grid = QGridLayout()
                for j, option in enumerate(options):
                    option_label = QLabel(f"• {option}")

                    # Highlight correct and user answers
                    if option == correct and option == user_answer:
                        option_label.setStyleSheet("color: green; font-weight: bold;")
                    elif option == correct:
                        option_label.setStyleSheet("color: green;")
                    elif option == user_answer:
                        option_label.setStyleSheet("color: red; font-weight: bold;")

                    options_grid.addWidget(option_label, j // 2, j % 2)

                q_layout.addLayout(options_grid)

                # Result
                is_correct = correct == user_answer
                result_label = QLabel(f"Result: {'Correct' if is_correct else 'Incorrect'}")
                result_label.setStyleSheet(f"color: {'green' if is_correct else 'red'}; font-weight: bold;")
                q_layout.addWidget(result_label)

                q_group.setLayout(q_layout)
                self.questions_layout.addWidget(q_group)

        except sqlite3.Error as e:
            QMessageBox.warning(self, "Database Error", f"Failed to load quiz details: {str(e)}")
        finally:
            conn.close()

    def go_back(self):
        self.back_requested.emit()


class MainApplication(QMainWindow):
    """Main application class"""

    def __init__(self):
        super().__init__()

        # Create database if it doesn't exist
        create_database()

        # Initialize UI
        self.init_ui()

        # Set up central widget with stacked layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.stacked_layout = QStackedWidget()

        main_layout = QVBoxLayout(self.central_widget)
        main_layout.addWidget(self.stacked_layout)

        # Create and add login window
        self.login_window = LoginWindow()
        self.login_window.login_successful.connect(self.handle_login)
        self.login_window.register_requested.connect(self.show_register)
        self.stacked_layout.addWidget(self.login_window)

        # Create register window
        self.register_window = RegisterWindow()
        self.register_window.register_successful.connect(self.show_login)
        self.register_window.back_to_login.connect(self.show_login)
        self.stacked_layout.addWidget(self.register_window)

        # Set initial view to login
        self.show_login()

    def init_ui(self):
        self.setWindowTitle("TextQuiz")
        self.resize(800, 600)
        self.setMinimumSize(600, 450)

    def show_login(self):
        self.stacked_layout.setCurrentWidget(self.login_window)

    def show_register(self):
        self.stacked_layout.setCurrentWidget(self.register_window)

    def handle_login(self, user_id, username):
        # Create main menu window
        self.main_menu = MainMenuWindow(user_id, username)
        self.main_menu.logout_requested.connect(self.handle_logout)
        self.main_menu.start_quiz_requested.connect(self.start_new_quiz)
        self.main_menu.view_history_requested.connect(self.view_history)

        # Add to stacked layout
        self.stacked_layout.addWidget(self.main_menu)
        self.stacked_layout.setCurrentWidget(self.main_menu)

        # Store user info
        self.current_user_id = user_id
        self.current_username = username

    def handle_logout(self):
        # Remove user-specific pages from the stack
        current_widget = self.stacked_layout.currentWidget()
        if current_widget != self.login_window and current_widget != self.register_window:
            self.stacked_layout.removeWidget(current_widget)
            current_widget.deleteLater()

        # Show login window
        self.show_login()

        # Clear user info
        self.current_user_id = None
        self.current_username = None

    def start_new_quiz(self):
        # Create image processing window
        self.image_processor = ImageProcessingWindow()
        self.image_processor.back_requested.connect(self.back_to_main_menu)
        self.image_processor.quiz_ready.connect(self.start_quiz)

        # Add to stacked layout
        self.stacked_layout.addWidget(self.image_processor)
        self.stacked_layout.setCurrentWidget(self.image_processor)

    def start_quiz(self, extracted_text):
        # Create quiz window
        self.quiz_window = QuizWindow(extracted_text)
        self.quiz_window.back_requested.connect(self.back_to_main_menu)
        self.quiz_window.quiz_completed.connect(self.show_results)

        # Add to stacked layout
        self.stacked_layout.addWidget(self.quiz_window)
        self.stacked_layout.setCurrentWidget(self.quiz_window)

    def show_results(self, results, extracted_text):
        # Create results window
        self.results_window = ResultsWindow(results, extracted_text, self.current_user_id)
        self.results_window.new_quiz_requested.connect(self.start_new_quiz)
        self.results_window.back_to_menu_requested.connect(self.back_to_main_menu)

        # Add to stacked layout
        self.stacked_layout.addWidget(self.results_window)
        self.stacked_layout.setCurrentWidget(self.results_window)

    def view_history(self):
        # Create history window
        self.history_window = HistoryWindow(self.current_user_id)
        self.history_window.back_requested.connect(self.back_to_main_menu)
        self.history_window.view_details_requested.connect(self.view_quiz_details)

        # Add to stacked layout
        self.stacked_layout.addWidget(self.history_window)
        self.stacked_layout.setCurrentWidget(self.history_window)

    def view_quiz_details(self, quiz_id):
        # Create quiz details window
        self.details_window = QuizDetailsWindow(quiz_id)
        self.details_window.back_requested.connect(self.back_to_history)

        # Add to stacked layout
        self.stacked_layout.addWidget(self.details_window)
        self.stacked_layout.setCurrentWidget(self.details_window)

    def back_to_main_menu(self):
        # Remove current widget from stack
        current_widget = self.stacked_layout.currentWidget()
        if current_widget != self.main_menu:
            self.stacked_layout.removeWidget(current_widget)
            current_widget.deleteLater()

        # Create a new main menu if needed
        if not hasattr(self, 'main_menu') or self.main_menu is None:
            self.main_menu = MainMenuWindow(self.current_user_id, self.current_username)
            self.main_menu.logout_requested.connect(self.handle_logout)
            self.main_menu.start_quiz_requested.connect(self.start_new_quiz)
            self.main_menu.view_history_requested.connect(self.view_history)
            self.stacked_layout.addWidget(self.main_menu)

        # Show main menu
        self.stacked_layout.setCurrentWidget(self.main_menu)

    def back_to_history(self):
        # Remove details window from stack
        self.stacked_layout.removeWidget(self.details_window)
        self.details_window.deleteLater()

        # Show history window
        self.stacked_layout.setCurrentWidget(self.history_window)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle('Fusion')

    # Create and show the application
    main_app = MainApplication()
    main_app.show()

    sys.exit(app.exec())
