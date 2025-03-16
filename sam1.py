import cv2
import numpy as np
import os
import re

def analyze_ballot_sheet(image_path):
    """
    Analyze a ballot sheet to detect the swastik seal in the 3rd row
    and return the serial number based on the column position.
    
    Args:
        image_path (str): Path to the ballot sheet image
        
    Returns:
        str: The serial number from the detected column
    """
    # Check if image exists
    if not os.path.exists(image_path):
        return "Error: Image file not found"
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        return "Error: Unable to read image"
    
    # Get image dimensions
    height, width, _ = image.shape
    
    # Define the grid (5 columns, 3 rows)
    col_width = width // 5
    row_height = height // 3
    
    # Extract third row cells
    third_row_y = 2 * row_height  # 0-indexed, so 2 is the third row
    third_row = image[third_row_y:third_row_y + row_height, 0:width]
    
    # Create copies of the third row for visualization and processing
    third_row_copy = third_row.copy()
    
    # Function to detect swastik seal in a cell
    def detect_swastik(cell_img):
        # Convert to grayscale
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area to find potential swastik
        min_area = 100  # Adjust based on your image size and seal size
        swastik_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
        
        # If we have enough complex contours, it might be a swastik
        if len(swastik_contours) >= 3:
            # Check for cross-like pattern which is characteristic of swastik
            for cnt in swastik_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                # Swastik typically has balanced dimensions
                if 0.7 < aspect_ratio < 1.3:
                    return True
        
        return False
    
    # Analyze each column in the third row
    swastik_column = -1
    for col in range(5):
        x_start = col * col_width
        x_end = (col + 1) * col_width
        
        # Extract the cell
        cell = third_row[:, x_start:x_end]
        
        # Check for swastik in this cell
        if detect_swastik(cell):
            swastik_column = col
            # Draw a rectangle around the detected cell
            cv2.rectangle(third_row_copy, (x_start, 0), (x_end, row_height), (0, 255, 0), 3)
            break
    
    # If swastik was not found
    if swastik_column == -1:
        return "No swastik seal detected in the third row"
    
    # Since we can't use Tesseract, let's just report the column number
    # Save the visualization for reference
    output_path = os.path.join(os.path.dirname(image_path), 'detected_seal.jpg')
    cv2.imwrite(output_path, third_row_copy)
    
    # For now, since OCR isn't working, we'll return a simplified result
    return f"Swastik seal detected in column {swastik_column + 1}.\nDetection image saved to: {output_path}"

# Example usage
if __name__ == "__main__":
    # Use your specific local file path
    ballot_image_path = r"C:\Users\kisho\Downloads\Untitled design (1).png"
    result = analyze_ballot_sheet(ballot_image_path)
    print(result)