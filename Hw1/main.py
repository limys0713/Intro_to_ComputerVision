# Import
import sys
import cv2
import numpy as np
### sys module is part of Python’s standard library and provides system-specific functions. In PyQt applications, sys is mainly used to pass command-line arguments to QApplication and to handle the program’s exit behavior.
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
### QApplication is the core of any PyQt application. It sets up application-wide settings and handles events.
### QMainWindow is a type of widget that provides a more complex window structure than QWidget. It includes a title bar, status bar, and space for toolbars or a menu.
from hw1_ui import UI_MainWindow

class MainWindow(QMainWindow, UI_MainWindow):
    # Initialize
    ### Simpler version of __init__, since I'm building a standalone application window without any special requirements for parent-child relationships or custom window behaviors
    def __init__(self):
        ### Tells python to look for the parent class of MainWondow and call its methods
        ### self refers to the current instance of MainWindow
        ### Calls the __init__ method of the parent class without any additional arguments(default)
        ### super().__init__() is preferred in Python 3
        super(MainWindow, self).__init__()
        # Setup the UI from ui.py
        self.setup_ui(self)  
        
        # LOAD IMAGE
        ### Using lambda creates an anonymous function that calls self.load_image_button_click(self.image1_label) only when the button is clicked.
        ### If you use a unique function for each button, you don’t need to pass self.image1_label or self.image2_label as arguments, because each function can directly access the specific label.
        self.load_image1_button.clicked.connect(lambda: self.load_image_button_click(self.image1_label))
        self.load_image2_button.clicked.connect(lambda: self.load_image_button_click(self.image2_label))

        # Question 1
        self.color_separation_button.clicked.connect(self.color_separation)
        self.color_transformation_button.clicked.connect(self.color_transformation)
        self.color_extraction_button.clicked.connect(self.color_extraction)

        # Question 2 
        self.gaussian_blur_button.clicked.connect(self.gaussian_blur)
        self.bilateral_filter_button.clicked.connect(self.bilateral_filter)
        self.median_filter_button.clicked.connect(self.median_filter)

        # Question 3 
        self.sobel_x_button.clicked.connect(self.sobel_x)
        self.sobel_y_button.clicked.connect(self.sobel_y)
        self.combination_and_threshold_button.clicked.connect(self.combine_threshold)
        self.gradient_angle_button.clicked.connect(self.gradient_angle)

        # Question 4
        self.transforms_button.clicked.connect(self.transforms)
        
    # Load image function
    def load_image_button_click(self, label):
        file_name = QFileDialog.getOpenFileName(self, 'Open File', '.')
        f_name = file_name[0]
        if f_name:  # Check if a file was selected
            # Set the label text to the full file path
            label.setText(f_name)  # Display the full file path
            self.image_path = f_name    # Set image path variable
        #df = pd.DataFrame(pd.read_excel(f_name))
    
    # Question 1.1 
    def color_separation(self):
        #print("Image path:", self.image_path)  # Debugging line
        image = cv2.imread(self.image_path)
        b, g, r = cv2.split(image)
        zeros = np.zeros_like(b)

        self.b_image = cv2.merge([b, zeros, zeros])
        self.g_image = cv2.merge([zeros, g, zeros])
        self.r_image = cv2.merge([zeros, zeros, r])

        cv2.imshow("b_image", self.b_image)
        cv2.imshow("g_image", self.g_image)
        cv2.imshow("r_image", self.r_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 1.2 
    def color_transformation(self):
        image = cv2.imread(self.image_path)
        cv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        b = cv2.split(self.b_image)[0]
        g = cv2.split(self.g_image)[1]
        r = cv2.split(self.r_image)[2]
        avg_gray = ((b / 3 + g / 3 + r / 3)).astype(np.uint8)

        cv2.imshow("cv_gray", cv_gray)
        cv2.imshow("avg_gray", avg_gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Question 1.3
    def color_extraction(self):
        image = cv2.imread(self.image_path)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_bound = np.array([18, 0, 25])
        upper_bound = np.array([85, 255, 255])
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        mask_inverse = cv2.bitwise_not(mask)
        extracted_image = cv2.bitwise_and(image, image, mask=mask_inverse)
        
        cv2.imshow("Yellow-Green mask", mask)
        #cv2.imshow("The inversed mask", mask_inverse)
        cv2.imshow("Image without yellow and green", extracted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 2.1 
    def gaussian_blur(self):
        image = cv2.imread(self.image_path)
        # Default kernel radius
        initial_m = 1
        max_m = 5
        
        def trackbar(m):
            kernel_size = 2 * m + 1
            blur = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            cv2.imshow("Gaussian Blur", blur)
        
        ### The initial cv2.imshow("Gaussian Blur", image) line is necessary because it creates the window where the trackbar is displayed. The createTrackbar function must be attached to an already created window.
        cv2.imshow("Gaussian Blur", image)
        cv2.createTrackbar("m", "Gaussian Blur", initial_m, max_m, trackbar)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 2.2 
    def bilateral_filter(self):
        image = cv2.imread(self.image_path)
        # Default kernel radius
        initial_m = 1
        max_m = 5
        
        def trackbar(m):
            d = 2 * m + 1
            bilateral = cv2.bilateralFilter(image, d, 90, 90)
            cv2.imshow("Bilateral Filter", bilateral)
        
        cv2.imshow("Bilateral Filter", image)
        cv2.createTrackbar("m", "Bilateral Filter", initial_m, max_m, trackbar)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 2.3 
    def median_filter(self):
        image = cv2.imread(self.image_path)
        # Default kernel radius
        initial_m = 1
        max_m = 5
        
        def trackbar(m):
            kernel = 2 * m + 1
            median = cv2.medianBlur(image, kernel)
            cv2.imshow("Median Filter", median)
        
        cv2.imshow("Median Filter", image)
        cv2.createTrackbar("m", "Median Filter", initial_m, max_m, trackbar)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 3.1 
    def sobel_x(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)   # Apply Grayscale
        image = cv2.GaussianBlur(image, (3, 3), 0)  # Apply Gaussian smoothing

        # Sobel X kernel
        sobel_x_kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]])

        # Perform convolution manually
        sobel_x_result = self.apply_convolution(image, sobel_x_kernel)
        self.sobel_x_result = sobel_x_result  # Store result for later use in combination

        cv2.imshow("Sobel x", sobel_x_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 3.2 
    def sobel_y(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.GaussianBlur(image, (3, 3), 0)  # Apply Gaussian smoothing

        # Sobel Y kernel
        sobel_y_kernel = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]])

        # Perform convolution manually
        sobel_y_result = self.apply_convolution(image, sobel_y_kernel)
        self.sobel_y_result = sobel_y_result  # Store result for later use in combination

        cv2.imshow("Sobel y", sobel_y_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 3.3
    def combine_threshold(self):
        # Sobel X and Sobel Y from 3.1 and 3.2
        sobel_x_result = self.sobel_x_result.astype(np.float32)  # Cast to float32 for precision
        sobel_y_result = self.sobel_y_result.astype(np.float32)

        # Combine Sobel X and Y 
        new_pixel = np.sqrt(sobel_x_result ** 2 + sobel_y_result ** 2)
        
        normalized = cv2.normalize(new_pixel, None, 0, 255, cv2.NORM_MINMAX)
        # Convert to uint8 for having a normal picture in imshow
        normalized = normalized.astype(np.uint8)
        self.combination_image = normalized

        # Apply thresholding at levels 128 and 28
        _, threshold_128 = cv2.threshold(normalized, 128, 255, cv2.THRESH_BINARY)
        _, threshold_28 = cv2.threshold(normalized, 28, 255, cv2.THRESH_BINARY)

        cv2.imshow("Combination of Sobel x and Sobel y", normalized)
        cv2.imshow("Threshold 128", threshold_128)
        cv2.imshow("Threshold 28", threshold_28)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Question 3.4
    def gradient_angle(self):
        # Load the image and convert it to grayscale
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        
        # Define Sobel kernels for X and Y gradients
        sobel_x_kernel = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=np.float32)
        
        sobel_y_kernel = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=np.float32)
        
        # Use apply_convolution_3_4 function to calc Sobel x and y
        sobel_x_result = self.apply_convolution_3_4(image, sobel_x_kernel)
        sobel_y_result = self.apply_convolution_3_4(image, sobel_y_kernel)

        # Retrieve the combined gradient image from question 3.3
        combination_result = self.combination_image.astype(np.uint8)
        #combination_result = cv2.normalize(combination_result, None, 0, 255, cv2.NORM_MINMAX)

        height, width = image.shape[:2]
        # Calculate the gradient angle for each pixel
        angle = np.zeros((height, width), dtype="float32")
        for x in range(height):
            for y in range(width):
                angle[x, y] = (np.degrees(np.arctan2(sobel_y_result[x, y], sobel_x_result[x, y])) + 360) % 360

        # Create masks for the specified angle ranges
        # Mask for angles between 170° and 190°
        mask1 = (angle >= 170) & (angle <= 190)
        mask1 = mask1.astype(np.uint8) * 255  # Convert to a binary mask with values 0 and 255

        # Mask for angles between 260° and 280°
        mask2 = (angle >= 260) & (angle <= 280)
        mask2 = mask2.astype(np.uint8) * 255  # Convert to a binary mask with values 0 and 255

        # Apply the masks to the combination_result using cv2.bitwise_and
        result1 = cv2.bitwise_and(combination_result, combination_result, mask=mask1)
        result2 = cv2.bitwise_and(combination_result, combination_result, mask=mask2)

        # Display the results
        cv2.imshow("Angle 170-190", result1)
        cv2.imshow("Angle 260-280", result2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Function that perform manual convolution for Sobel x and y
    def apply_convolution(self, image, kernel):
        # Get the dimensions of the image
        image_height, image_width = image.shape

        # Initialize the output image with float data type 
        output = np.zeros((image_height, image_width), dtype=np.float32)

        # Define the offset based on kernel size (assuming a 3x3 kernel)
        offset = kernel.shape[0] // 2

        # Loop over each pixel in the image (excluding borders)
        for y in range(offset, image_height - offset):
            for x in range(offset, image_width - offset):
                # Extract the region corresponding to the kernel
                region = image[y - offset:y + offset + 1, x - offset:x + offset + 1]
                convolution_sum = np.sum(region * kernel)
                output[y, x] = convolution_sum

        # Take the absolute value to retain only the edge intensities
        output = np.abs(output)
        # Normalize the remaining values to enhance the edges
        output = np.clip(output, 0, 255)
        return output.astype(np.uint8)
    
    def apply_convolution_3_4(self, image, kernel):
        # Get the dimensions of the image
        image_height, image_width = image.shape

        # Initialize the output image with float data type
        output = np.zeros((image_height, image_width), dtype=np.float32)

        # Define the offset based on kernel size (assuming a 3x3 kernel)
        offset = kernel.shape[0] // 2

        # Loop over each pixel in the image (excluding borders)
        for y in range(offset, image_height - offset):
            for x in range(offset, image_width - offset):
                # Extract the region corresponding to the kernel
                region = image[y - offset:y + offset + 1, x - offset:x + offset + 1]
                convolution_sum = np.sum(region * kernel)
                output[y, x] = convolution_sum

        return output

    # Question 4
    def transforms(self):
        image = cv2.imread(self.image_path)
        h, w = image.shape[:2]

        # Retrieve values from textboxes
        angle = int(self.rotation_input.text()) if self.rotation_input.text() else 0
        scale = float(self.scaling_input.text()) if self.scaling_input.text() else 1
        tx = int(self.tx_input.text()) if self.tx_input.text() else 0
        ty = int(self.ty_input.text()) if self.ty_input.text() else 0

        # Set the center
        center = (240, 200)
        
        # Rotation and scaling matrix around the defined center
        rotation_scaling = cv2.getRotationMatrix2D(center, angle, scale)
        # Apply rotation and scaling
        rotated_scaled = cv2.warpAffine(image, rotation_scaling, (w, h))

        # Translation matrix
        translation = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        # Apply translation to the rotated and scaled image
        transformed = cv2.warpAffine(rotated_scaled, translation, (w, h))

        cv2.imshow("Transformed Image", transformed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def application():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

application()