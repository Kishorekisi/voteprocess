import cv2
import numpy as np
import argparse

def detect_objects(image_path, confidence_threshold=0.5):
    """
    Detect objects in an image using a pre-trained YOLO model
    
    Args:
        image_path (str): Path to the input image
        confidence_threshold (float): Minimum confidence threshold for detections
        
    Returns:
        image: Image with bounding boxes drawn
        detections: List of detected objects with their coordinates and confidence
    """
    # Load YOLO model
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    
    # Load COCO class names
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Load and prepare image
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    
    # Pass blob through network
    net.setInput(blob)
    outputs = net.forward(output_layers)
    
    # Process detections
    class_ids = []
    confidences = []
    boxes = []
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > confidence_threshold:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression to remove overlapping boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
    
    # Prepare detections list
    detections = []
    
    # Draw bounding boxes on image
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0)  # Green color for boxes
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (x, y - 10), font, 0.5, color, 2)
            
            detections.append({
                'label': label,
                'confidence': confidence,
                'box': (x, y, w, h)
            })
    
    return image, detections

def main():
    parser = argparse.ArgumentParser(description='Object Detection with YOLO')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--output', default='output.jpg', help='Path to output image')
    args = parser.parse_args()
    
    # Detect objects
    image, detections = detect_objects(args.image, args.confidence)
    
    # Print detections
    print(f"Found {len(detections)} objects:")
    for i, detection in enumerate(detections, 1):
        print(f"{i}. {detection['label']} - Confidence: {detection['confidence']:.2f}")
    
    # Save output image
    cv2.imwrite(args.output, image)
    print(f"Output image saved to {args.output}")
    
    # Display image (uncomment if running in environment with display)
    # cv2.imshow('Object Detection', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()