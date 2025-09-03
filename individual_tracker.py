import cv2
import numpy as np
import datetime
import json
import csv
from collections import OrderedDict

class PersonDetector:
    """Lightweight person detector using OpenCV's HOG + SVM"""
    
    def __init__(self):
        print("🔧 Initializing HOG Person Detector...")
        
        # Initialize HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        print("✅ HOG Person Detector initialized")
    
    def detect_persons(self, frame):
        """Detect persons in a frame using HOG"""
        # Convert to grayscale for faster processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect people with fallback methods for different OpenCV versions
        try:
            boxes, weights = self.hog.detectMultiScale(
                gray,
                winStride=(8, 8),
                padding=(16, 16),
                scale=1.05,
                groupThreshold=2  # Works with newer OpenCV
            )
        except:
            try:
                boxes, weights = self.hog.detectMultiScale(
                    gray,
                    winStride=(8, 8),
                    padding=(16, 16),
                    scale=1.05
                )
            except:
                boxes, weights = self.hog.detectMultiScale(gray)
        
        # Convert from (x, y, w, h) to (x1, y1, x2, y2) format
        detection_boxes = []
        for (x, y, w, h) in boxes:
            detection_boxes.append((x, y, x + w, y + h))
        
        return detection_boxes

class SimpleTracker:
    """Simple centroid-based tracker for individual person tracking"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        print("🔧 Initializing Simple Tracker...")
        
        # Initialize the next unique object ID 
        self.next_id = 1
        
        # Dictionary to store centroid of each tracked person
        self.objects = OrderedDict()
        
        # Dictionary to store number of consecutive frames person has been missing
        self.disappeared = OrderedDict()
        
        # Maximum number of frames person can be missing before deregistering
        self.max_disappeared = max_disappeared
        
        # Maximum distance between centroids to consider same person
        self.max_distance = max_distance
        
        # Track timestamps and info for each person
        self.person_timestamps = {}
        
        print(f"✅ Simple Tracker initialized")
        print(f"   - Max disappeared frames: {max_disappeared}")
        print(f"   - Max tracking distance: {max_distance} pixels")
    
    def register(self, centroid, timestamp):
        """Register a new person with a unique ID"""
        person_id = f"Person_{self.next_id:03d}"
        self.objects[person_id] = centroid
        self.disappeared[person_id] = 0
        
        # Record entry timestamp and details
        self.person_timestamps[person_id] = {
            'first_seen': timestamp,
            'last_seen': timestamp,
            'total_frames': 1,
            'status': 'active'
        }
        
        self.next_id += 1
        print(f"👋 {person_id} entered at {timestamp.strftime('%H:%M:%S')}")
        return person_id
    
    def deregister(self, person_id, timestamp):
        """Remove a person from active tracking"""
        if person_id in self.objects:
            # Update final timestamp
            if person_id in self.person_timestamps:
                self.person_timestamps[person_id]['last_seen'] = timestamp
                self.person_timestamps[person_id]['status'] = 'left'
                
                # Calculate total duration
                first_seen = self.person_timestamps[person_id]['first_seen']
                duration = (timestamp - first_seen).total_seconds()
                self.person_timestamps[person_id]['total_duration'] = duration
                
                print(f"🚪 {person_id} left at {timestamp.strftime('%H:%M:%S')} (duration: {duration:.1f}s)")
            
            del self.objects[person_id]
            del self.disappeared[person_id]
    
    def calculate_centroid(self, box):
        """Calculate centroid from bounding box (x1, y1, x2, y2)"""
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        return (cx, cy)
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def update(self, detection_boxes, timestamp):
        """Update tracker with new detections"""
        # If no detections, mark all existing objects as disappeared
        if len(detection_boxes) == 0:
            for person_id in list(self.disappeared.keys()):
                self.disappeared[person_id] += 1
                
                # Deregister if disappeared too long
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister(person_id, timestamp)
            
            return self.get_current_objects(timestamp)
        
        # Calculate centroids for all detections
        input_centroids = []
        for box in detection_boxes:
            centroid = self.calculate_centroid(box)
            input_centroids.append(centroid)
        
        # If no existing objects, register all detections as new
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid, timestamp)
        else:
            # Match existing objects to new detections using distance
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distances between existing and new centroids
            distances = np.zeros((len(object_centroids), len(input_centroids)))
            
            for i, obj_centroid in enumerate(object_centroids):
                for j, input_centroid in enumerate(input_centroids):
                    distances[i][j] = self.calculate_distance(obj_centroid, input_centroid)
            
            # Find the minimum distances (simplified Hungarian algorithm)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Update existing objects with matched detections
            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Check if distance is within threshold
                if distances[row, col] <= self.max_distance:
                    person_id = object_ids[row]
                    self.objects[person_id] = input_centroids[col]
                    self.disappeared[person_id] = 0
                    
                    # Update timestamp and frame count
                    self.person_timestamps[person_id]['last_seen'] = timestamp
                    self.person_timestamps[person_id]['total_frames'] += 1
                    
                    used_rows.add(row)
                    used_cols.add(col)
            
            # Handle unmatched existing objects (mark as disappeared)
            unused_rows = set(range(0, distances.shape[0])).difference(used_rows)
            for row in unused_rows:
                person_id = object_ids[row]
                self.disappeared[person_id] += 1
                
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister(person_id, timestamp)
            
            # Register new objects for unmatched detections
            unused_cols = set(range(0, distances.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], timestamp)
        
        return self.get_current_objects(timestamp)
    
    def get_current_objects(self, timestamp):
        """Get currently tracked objects with their detailed info"""
        current_objects = {}
        for person_id, centroid in self.objects.items():
            if person_id in self.person_timestamps:
                first_seen = self.person_timestamps[person_id]['first_seen']
                current_duration = (timestamp - first_seen).total_seconds()
                
                current_objects[person_id] = {
                    'centroid': centroid,
                    'first_seen': first_seen,
                    'last_seen': timestamp,
                    'current_duration': current_duration,
                    'total_frames': self.person_timestamps[person_id]['total_frames'],
                    'status': 'active'
                }
        
        return current_objects
    
    def get_all_person_data(self):
        """Get data for all persons (active + left)"""
        return self.person_timestamps.copy()

class VideoAnalyzer:
    """Main class that combines detection, individual tracking, and timestamp logging"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        print("🔧 Initializing Video Analyzer with Individual Tracking...")
        
        # Initialize detector and tracker
        self.detector = PersonDetector()
        self.tracker = SimpleTracker(max_disappeared, max_distance)
        
        # Logging
        self.frame_logs = []
        self.start_time = None
        
        print("✅ Video Analyzer with Individual Tracking initialized!")
    
    def process_video(self, video_path, output_path=None, show_video=True):
        """Process entire video with individual person tracking and timestamps"""
        print(f"\n🎬 Processing video with individual tracking: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video properties:")
        print(f"   - Resolution: {frame_width}x{frame_height}")
        print(f"   - FPS: {fps}")
        print(f"   - Total frames: {total_frames}")
        print(f"   - Duration: {total_frames/fps:.1f} seconds")
        
        # Setup output video if requested
        out_video = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_video = cv2.VideoWriter(f"{output_path}.mp4", fourcc, fps, 
                                      (frame_width, frame_height))
            print(f"💾 Output video will be saved as: {output_path}.mp4")
        
        frame_count = 0
        self.start_time = datetime.datetime.now()
        
        print(f"\n🔍 Starting individual tracking...")
        print("Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("📽️ Reached end of video")
                break
            
            # Calculate timestamp for this frame
            video_timestamp = self.start_time + datetime.timedelta(seconds=frame_count/fps)
            
            # Step 1: Detect persons in frame
            try:
                detection_boxes = self.detector.detect_persons(frame)
            except Exception as e:
                print(f"⚠️ Detection error on frame {frame_count}: {e}")
                detection_boxes = []
            
            # Step 2: Update tracker with detections 
            tracked_objects = self.tracker.update(detection_boxes, video_timestamp)
            
            # Step 3: Draw tracking information
            annotated_frame = self.draw_tracking_annotations(
                frame, detection_boxes, tracked_objects, video_timestamp, frame_count, total_frames
            )
            
            # Step 4: Log frame data
            frame_log = {
                'frame': frame_count,
                'timestamp': video_timestamp.isoformat(),
                'detections': len(detection_boxes),
                'active_persons': len(tracked_objects),
                'persons': {
                    pid: {
                        'centroid': obj['centroid'],
                        'duration': obj['current_duration'],
                        'frames': obj['total_frames']
                    } for pid, obj in tracked_objects.items()
                }
            }
            self.frame_logs.append(frame_log)
            
            # Step 5: Save frame to output video
            if out_video:
                out_video.write(annotated_frame)
            
            # Step 6: Show video (optional)
            if show_video:
                cv2.imshow('Individual Person Tracking - Press Q to quit', annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q'):
                    print("⏹️ Stopped by user")
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_name = f"tracking_frame_{frame_count}.jpg"
                    cv2.imwrite(save_name, annotated_frame)
                    print(f"💾 Saved frame as {save_name}")
            
            frame_count += 1
            
            # Progress update
            update_interval = 10 if total_frames < 300 else 30
            if frame_count % update_interval == 0:
                progress = (frame_count / total_frames) * 100
                active_count = len(tracked_objects)
                total_seen = self.tracker.next_id - 1
                print(f"⏳ Progress: {progress:.1f}% | Active: {active_count} | Total seen: {total_seen}")
        
        # Cleanup
        cap.release()
        if out_video:
            out_video.release()
        cv2.destroyAllWindows()
        
        # Save results
        if output_path:
            self.save_tracking_results(output_path)
        
        # Print final summary
        self.print_final_summary(frame_count)
        
        return self.get_analysis_summary()
    
    def draw_tracking_annotations(self, frame, detection_boxes, tracked_objects, timestamp, frame_num, total_frames):
        """Draw all tracking annotations on the frame"""
        annotated_frame = frame.copy()
        
        # Draw detection boxes (green)
        for (x1, y1, x2, y2) in detection_boxes:
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw tracked objects with persistent IDs and info
        for person_id, obj in tracked_objects.items():
            centroid = obj['centroid']
            duration = obj['current_duration']
            frames_seen = obj['total_frames']
            
            # Draw centroid (red circle)
            cv2.circle(annotated_frame, centroid, 8, (0, 0, 255), -1)
            
            # Draw person ID (yellow text)
            text_y = centroid[1] - 30
            cv2.putText(annotated_frame, person_id, 
                       (centroid[0] - 40, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw duration (white text)
            cv2.putText(annotated_frame, f"{duration:.1f}s", 
                       (centroid[0] - 30, text_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw frame count (light blue text)
            cv2.putText(annotated_frame, f"F:{frames_seen}", 
                       (centroid[0] - 25, text_y + 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)
        
        # Draw summary information (top-left)
        cv2.putText(annotated_frame, f"Active: {len(tracked_objects)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(annotated_frame, f"Total seen: {self.tracker.next_id - 1}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(annotated_frame, f"Detections: {len(detection_boxes)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw timestamp (bottom-left)
        timestamp_str = timestamp.strftime("%H:%M:%S")
        cv2.putText(annotated_frame, f"Time: {timestamp_str}", 
                   (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw frame counter (bottom-left)
        cv2.putText(annotated_frame, f"Frame: {frame_num}/{total_frames}", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame
    
    def save_tracking_results(self, output_path):
        """Save comprehensive tracking results"""
        print(f"\n💾 Saving tracking results...")
        
        # Save detailed frame logs as JSON
        with open(f"{output_path}_frames.json", 'w') as f:
            json.dump(self.frame_logs, f, indent=2, default=str)
        
        # Save individual person summary as CSV
        all_persons = self.tracker.get_all_person_data()
        person_summary = []
        
        for person_id, data in all_persons.items():
            first_seen_str = data['first_seen'].strftime("%H:%M:%S")
            last_seen_str = data['last_seen'].strftime("%H:%M:%S")
            duration = data.get('total_duration', 
                              (data['last_seen'] - data['first_seen']).total_seconds())
            
            person_summary.append({
                'Person_ID': person_id,
                'First_Seen': first_seen_str,
                'Last_Seen': last_seen_str,
                'Duration_Seconds': round(duration, 1),
                'Total_Frames': data['total_frames'],
                'Status': data['status']
            })
        
        # Sort by Person_ID for better readability
        person_summary.sort(key=lambda x: x['Person_ID'])
        
        # Save to CSV
        with open(f"{output_path}_individual_tracking.csv", 'w', newline='') as f:
            if person_summary:
                writer = csv.DictWriter(f, fieldnames=person_summary[0].keys())
                writer.writeheader()
                writer.writerows(person_summary)
        
        print(f"✅ Results saved:")
        print(f"   - {output_path}_frames.json (detailed frame logs)")
        print(f"   - {output_path}_individual_tracking.csv (person summaries)")
        print(f"   - {output_path}.mp4 (annotated video)")
    
    def print_final_summary(self, total_frames_processed):
        """Print comprehensive final summary"""
        all_persons = self.tracker.get_all_person_data()
        active_persons = len([p for p in all_persons.values() if p['status'] == 'active'])
        left_persons = len([p for p in all_persons.values() if p['status'] == 'left'])
        
        print(f"\n🎉 INDIVIDUAL TRACKING COMPLETE!")
        print(f"=" * 45)
        print(f"📊 Final Statistics:")
        print(f"   - Total frames processed: {total_frames_processed}")
        print(f"   - Unique individuals detected: {len(all_persons)}")
        print(f"   - Still active: {active_persons}")
        print(f"   - Left the scene: {left_persons}")
        
        if all_persons:
            avg_duration = np.mean([p.get('total_duration', 0) for p in all_persons.values()])
            print(f"   - Average time per person: {avg_duration:.1f}s")
        
        # Show first few people for verification
        print(f"\n👥 Individual Person Summary:")
        for i, (person_id, data) in enumerate(list(all_persons.items())[:5]):
            duration = data.get('total_duration', 
                              (data['last_seen'] - data['first_seen']).total_seconds())
            print(f"   {person_id}: {data['first_seen'].strftime('%H:%M:%S')} - {data['last_seen'].strftime('%H:%M:%S')} ({duration:.1f}s)")
        
        if len(all_persons) > 5:
            print(f"   ... and {len(all_persons) - 5} more (see CSV file)")
    
    def get_analysis_summary(self):
        """Get summary for return"""
        all_persons = self.tracker.get_all_person_data()
        return {
            'total_persons': len(all_persons),
            'total_frames': len(self.frame_logs),
            'tracking_data': all_persons
        }

def main():
    """Main function to run individual tracking"""
    print("🎯 INDIVIDUAL PERSON TRACKING SYSTEM")
    print("=" * 40)
    print("1. Test on video file")
    print("2. Exit")
    
    choice = input("\nSelect option (1-2): ").strip()
    
    if choice == '1':
        video_path = input("Enter video file path: ").strip()
        output_path = input("Enter output path (optional, press Enter to skip): ").strip()
        
        if not output_path:
            # Generate automatic output path
            import os
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"{video_name}_tracking_{timestamp}"
        
        # Initialize and run analyzer
        analyzer = VideoAnalyzer(max_disappeared=30, max_distance=100)
        
        try:
            summary = analyzer.process_video(
                video_path=video_path,
                output_path=output_path,
                show_video=True
            )
            
            print(f"\n✅ Analysis complete! Check your files:")
            print(f"   - {output_path}_individual_tracking.csv")
            print(f"   - {output_path}_frames.json")
            print(f"   - {output_path}.mp4")
            
        except KeyboardInterrupt:
            print("\n⏹️ Processing interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
    
    elif choice == '2':
        print("👋 Goodbye!")
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()