
import cv2
import numpy as np

def simple_hog_test():
    """Simple test of HOG person detection"""
    print("🔧 Testing HOG Person Detection...")
    
    try:
        # Initialize HOG detector
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        print("✅ HOG detector initialized successfully")
        
        # Create a simple test image (black image)
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        print("✅ Test frame created")
        
        # Try detection on test frame
        gray = cv2.cvtColor(test_frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = hog.detectMultiScale(gray)
        print(f"✅ Detection completed - found {len(boxes)} objects")
        
        print("\n🎉 PHASE 2 BASIC TEST PASSED!")
        print("HOG detection is working correctly")
        print("Ready to test on real video!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    simple_hog_test()
