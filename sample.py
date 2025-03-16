import cv2
import numpy as np
import os
import re
import time

def analyze_ballot_frame(frame):
    """
    Analyze a single frame from camera to detect ONLY the swastik seal in the 3rd row
    
    Args:
        frame: The camera frame to analyze
        
    Returns:
        tuple: (processed_frame, result_text)
    """
    # Get image dimensions
    height, width, _ = frame.shape
    
    # Define the grid (5 columns, 3 rows)
    col_width = width // 5
    row_height = height // 3
    
    # Extract ONLY third row cells
    third_row_y = 2 * row_height  # 0-indexed, so 2 is the third row
    third_row = frame[third_row_y:third_row_y + row_height, 0:width]
    second_row = 2 * row_height 
    
    # Create copies for visualization and processing
    result_frame = frame.copy()
    
    # Highlight the third row where we're looking for the seal
    cv2.rectangle(result_frame, (0, third_row_y), (width, third_row_y + row_height), (255, 255, 0), 2)
    
    # Draw column separators in the third row
    for i in range(1, 5):
        x = i * col_width
        cv2.line(result_frame, (x, third_row_y), (x, third_row_y + row_height), (255, 255, 0), 2)
    
    # Function to detect ONLY swastik seal in a cell
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
    
    # Analyze each column ONLY in the third row
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
            cv2.rectangle(result_frame, 
                         (x_start, third_row_y), 
                         (x_end, third_row_y + row_height), 
                         (0, 0, 255), 3)
            break
    
    # If swastik was not found
    if swastik_column == -1:
        result_text = "No swastik seal detected in the third row"
    else:
        result_text = f"Swastik seal detected in column {swastik_column + 1}"
    
    return result_frame, result_text, swastik_column

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)  # Use 0 for default camera, try 1 if doesn't work
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Camera initialized. Press 'c' to capture and analyze, 'q' to quit.")
    
    # Variables to track detection
    last_result = ""
    last_column = -1
    saved_count = 0
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Add instructions to display frame
        display_frame = frame.copy()
        cv2.putText(display_frame, "Position ballot in frame and press 'c' to capture", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Show the instruction frame
        cv2.imshow('Ballot Swastik Detector', display_frame)
        
        # Check for keypresses
        key = cv2.waitKey(1) & 0xFF
        
        # If 'q' pressed, quit
        if key == ord('q'):
            break
        
        # If 'c' pressed, capture and analyze
        elif key == ord('c'):
            
            # Analyze the current frame
            analyzed_frame, result_text, detected_column = analyze_ballot_frame(frame)
            
            # Update result tracking variables
            last_result = result_text
            last_column = detected_column
            
            # Show the analyzed frame
            cv2.imshow('Analysis Result', analyzed_frame)
            
            # If swastik was detected, save the analyzed frame
            if detected_column != -1:
                output_dir = os.path.expanduser('~/Desktop')
                if not os.path.exists(output_dir):
                    output_dir = os.getcwd()  # Fallback to current directory
                
                output_path = os.path.join(output_dir, f'swastik_detected_{saved_count}.jpg')
                cv2.imwrite(output_path, analyzed_frame)
                saved_count += 1
                
                print(f"{result_text}\nDetection image saved to: {output_path}")
            else:
                print(result_text)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()