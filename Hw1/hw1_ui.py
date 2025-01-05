# Import
from PyQt5 import QtWidgets # Contains all the different widgets(buttons, label, text fields)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton, QLabel, QGroupBox, QLineEdit

# Define the MainWindow class
class UI_MainWindow:
    # Initialize
    def setup_ui(self, MainWindow): 
        MainWindow.setWindowTitle("Hw1")  # Set the window title
        MainWindow.setGeometry(100, 100, 700, 800)    # Set the window's position and size

        ### LOAD IMAGE BUTTON AND LABEL
        # "Load Image 1" button and label
        self.load_image1_button = QPushButton("Load Image 1", MainWindow) # When using layout, no need to pass self as a argument, but it is okay if you write it
        self.image1_label = QLabel("No image loaded", MainWindow)
        self.load_image1_button.setGeometry(30, 250, 130, 30)
        self.image1_label.setGeometry(45, 280, 110, 30)
        # "Load Image 2" button and label
        self.load_image2_button = QPushButton("Load Image 2", MainWindow)
        self.image2_label = QLabel("No image loaded", MainWindow)
        self.load_image2_button.setGeometry(30, 440, 130, 30)
        self.image2_label.setGeometry(45, 470, 110, 30)

        ### Question 1~3 BUTTON & LABEL
        # Question 1
        self.question1_label = QGroupBox("1. Image Processing", MainWindow)
        self.question1_label.setGeometry(200, 50, 180, 190)
        self.color_separation_button = QPushButton("1.1 Color Separation", MainWindow)
        self.color_transformation_button = QPushButton("1.2 Color Transformation", MainWindow)
        self.color_extraction_button = QPushButton("1.3 Color Extraction", MainWindow)
        self.color_separation_button.setGeometry(210, 90, 160, 30)
        self.color_transformation_button.setGeometry(210, 140, 160, 30)
        self.color_extraction_button.setGeometry(210, 190, 160, 30)
        # Question 2
        self.question2_label = QGroupBox("2. Image Smoothing", MainWindow)
        self.question2_label.setGeometry(200, 260, 180, 190)
        self.gaussian_blur_button = QPushButton("2.1 Gaussian blur", MainWindow)
        self.bilateral_filter_button = QPushButton("2.2 Bilateral filter", MainWindow)
        self.median_filter_button = QPushButton("2.3 Median filter", MainWindow)
        self.gaussian_blur_button.setGeometry(210, 300, 160, 30)
        self.bilateral_filter_button.setGeometry(210, 350, 160, 30)
        self.median_filter_button.setGeometry(210, 400, 160, 30)
        # Question 3
        self.question3_label = QGroupBox("3. Edge Detection", MainWindow)
        self.question3_label.setGeometry(200, 470, 180, 250)
        self.sobel_x_button = QPushButton("3.1 Sobel X", MainWindow)
        self.sobel_y_button = QPushButton("3.2 Sobel Y", MainWindow)
        self.combination_and_threshold_button = QPushButton("3.3 Combination and \nThreshold", MainWindow)
        self.gradient_angle_button = QPushButton("3.4 Gradient Angle", MainWindow)
        self.sobel_x_button.setGeometry(210, 510, 160, 30)
        self.sobel_y_button.setGeometry(210, 560, 160, 30)
        self.combination_and_threshold_button.setGeometry(210, 610, 160, 40)
        self.gradient_angle_button.setGeometry(210, 670, 160, 30)

        ### Question 4 LABEL & INPUT TEXTBOX & BUTTON
        self.question4_label = QtWidgets.QGroupBox("4. Transforms", MainWindow)
        self.question4_label.setGeometry(420, 50, 195, 290)
        # Rotation
        self.rotation_label = QLabel("Rotation:", MainWindow)
        self.rotation_label.setGeometry(430, 90, 60, 30)
        self.rotation_input = QLineEdit(MainWindow)
        self.rotation_input.setGeometry(500, 90, 70, 30)
        self.rotation_input.setAlignment(Qt.AlignCenter)
        self.rotation_unit_label = QLabel("deg", MainWindow)
        self.rotation_unit_label.setGeometry(580, 90, 40, 30)
        # Scaling
        self.scaling_label = QLabel("Scaling:", MainWindow)
        self.scaling_label.setGeometry(430, 140, 60, 30)
        self.scaling_input = QLineEdit(MainWindow)
        self.scaling_input.setGeometry(500, 140, 70, 30)
        self.scaling_input.setAlignment(Qt.AlignCenter)
        # Tx
        self.tx_label = QLabel("Tx:", MainWindow)
        self.tx_label.setGeometry(430, 190, 60, 30)
        self.tx_input = QLineEdit(MainWindow)
        self.tx_input.setGeometry(500, 190, 70, 30)
        self.tx_input.setAlignment(Qt.AlignCenter)
        self.tx_unit = QLabel("pixel", MainWindow)
        self.tx_unit.setGeometry(580, 190, 40, 30)
        # Ty
        self.ty_label = QLabel("Ty:", MainWindow)
        self.ty_label.setGeometry(430, 240, 60, 30)
        self.ty_input = QLineEdit(MainWindow)
        self.ty_input.setGeometry(500, 240, 70, 30)
        self.ty_input.setAlignment(Qt.AlignCenter)
        self.ty_unit = QLabel("pixel", MainWindow)
        self.ty_unit.setGeometry(580, 240, 40, 30)
        # Transforms button
        self.transforms_button = QPushButton("4. Transforms", MainWindow)
        self.transforms_button.setGeometry(435, 290, 160, 30)
        


