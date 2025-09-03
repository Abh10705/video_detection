
import cv2
import numpy as np
import datetime
import json
import csv


import cv2
import numpy as np
import datetime

class PersonDetector:
    """Lightweight person detector using OpenCV's HOG + SVM"""
    
    def __init__(self):
        print("🔧 Initializing HOG Person Detector...")
        
        # Initialize HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        print("✅ HOG Person Detector initialized")
        print("   - Using built-in people detector")
        print("   - No external model files needed")
        print("   - Ready for detection!")
    
    def detect_persons(self, frame):
        """
        Detect persons in a frame using HOG
        Returns: list of bounding boxes [(x1, y1, x2, y2), ...]
        """
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect people in the frame - FIXED PARAMETERS
        try:
            # Method 1: Try with all parameters (newer OpenCV)
            boxes, weights = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(16, 16),
                scale=1.05,
                groupThreshold=2  # Changed from finalThreshold to groupThreshold
            )
        except:
            try:
                # Method 2: Try with basic parameters (older OpenCV)
                boxes, weights = self.hog.detectMultiScale(
                    gray,
                    winStride=(8, 8),
                    padding=(16, 16),
                    scale=1.05
                )
            except:
                # Method 3: Minimal parameters (fallback)
                boxes, weights = self.hog.detectMultiScale(gray)
        
        # Convert from (x, y, w, h) to (x1, y1, x2, y2) format
        detection_boxes = []
        for (x, y, w, h) in boxes:
            detection_boxes.append((x, y, x + w, y + h))
        
        print(f"   🔍 Detected {len(detection_boxes)} person(s)")
        return detection_boxes
    
    def draw_detections(self, frame, boxes):
        """Draw bounding boxes around detected persons"""
        annotated_frame = frame.copy()
        
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            # Draw rectangle around person
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add detection label
            label = f"Person {i+1}"
            cv2.putText(annotated_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add detection count
        count_text = f"Detected: {len(boxes)} persons"
        cv2.putText(annotated_frame, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return annotated_frame

def test_detector_on_video(video_path):
    """Test the detector on a video file"""
    print(f"\n🎬 Testing detector on video: {video_path}")
    
    # Initialize detector
    detector = PersonDetector()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        print("   Make sure the video file exists and is in a supported format")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video properties:")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Total frames: {total_frames}")
    print(f"   - Duration: {total_frames/fps:.1f} seconds")
    
    frame_count = 0
    detections_log = []
    
    print(f"\n🔍 Starting detection...")
    print("Press 'q' to quit, 's' to save current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("📽️ Reached end of video")
            break
        
        # Calculate timestamp
        timestamp = datetime.datetime.now() + datetime.timedelta(seconds=frame_count/fps)
        
        # Detect persons
        try:
            detection_boxes = detector.detect_persons(frame)
        except Exception as e:
            print(f"⚠️ Detection error on frame {frame_count}: {e}")
            detection_boxes = []
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detection_boxes)
        
        # Add timestamp to frame
        timestamp_str = timestamp.strftime("%H:%M:%S")
        cv2.putText(annotated_frame, f"Time: {timestamp_str}", 
                   (10, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                   (10, height-50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Log detection
        detections_log.append({
            'frame': frame_count,
            'timestamp': timestamp_str,
            'detections': len(detection_boxes),
            'boxes': detection_boxes
        })
        
        # Show frame
        cv2.imshow('Person Detection Test - Press Q to quit', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            print("⏹️ Stopped by user")
            break
        elif key == ord('s'):
            # Save current frame
            save_name = f"detection_frame_{frame_count}.jpg"
            cv2.imwrite(save_name, annotated_frame)
            print(f"💾 Saved frame as {save_name}")
        
        frame_count += 1
        
        # Progress update every 30 frames (or every 10 for short videos)
        update_interval = 10 if total_frames < 300 else 30
        if frame_count % update_interval == 0:
            progress = (frame_count / total_frames) * 100
            print(f"⏳ Progress: {progress:.1f}% - Frame {frame_count}/{total_frames}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    total_detections = sum(log['detections'] for log in detections_log)
    avg_detections = total_detections / len(detections_log) if detections_log else 0
    frames_with_detections = sum(1 for log in detections_log if log['detections'] > 0)
    
    print(f"\n📈 DETECTION SUMMARY:")
    print(f"   - Total frames processed: {frame_count}")
    print(f"   - Total person detections: {total_detections}")
    print(f"   - Average detections per frame: {avg_detections:.1f}")
    print(f"   - Frames with detections: {frames_with_detections}/{frame_count}")
    print(f"   - Detection rate: {(frames_with_detections/frame_count)*100:.1f}%")
    
    return detections_log

def test_detector_on_webcam():
    """Test the detector on webcam feed"""
    print(f"\n📹 Testing detector on webcam...")
    
    # Initialize detector
    detector = PersonDetector()
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("   Make sure your webcam is connected and not used by another app")
        return
    
    print("✅ Webcam opened successfully")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error reading from webcam")
            break
        
        # Detect persons
        try:
            detection_boxes = detector.detect_persons(frame)
        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            detection_boxes = []
        
        # Draw detections
        annotated_frame = detector.draw_detections(frame, detection_boxes)
        
        # Add frame counter
        cv2.putText(annotated_frame, f"Frame: {frame_count}", 
                   (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show frame
        cv2.imshow('Webcam Person Detection - Press Q to quit', annotated_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("⏹️ Webcam test stopped by user")
            break
        elif key == ord('s'):
            # Save current frame
            save_name = f"webcam_detection_{frame_count}.jpg"
            cv2.imwrite(save_name, annotated_frame)
            print(f"💾 Saved frame as {save_name}")
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def main():
    """Main function to test the detector"""
    print("🎯 PERSON DETECTOR TESTING MENU - FIXED VERSION")
    print("=" * 48)
    print("1. Test on video file")
    print("2. Test on webcam")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == '1':
        video_path = input("Enter video file path: ").strip()
        test_detector_on_video(video_path)
    elif choice == '2':
        test_detector_on_webcam()
    elif choice == '3':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()