
import sys

def test_dependencies():
    print("🚀 PHASE 1: TESTING MINIMAL DEPENDENCIES")
    print("=" * 50)
    
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy', 
        'datetime': 'built-in',
        'json': 'built-in',
        'csv': 'built-in',
        'argparse': 'built-in',
        'os': 'built-in'
    }
    
    missing_packages = []
    
    for package, install_name in required_packages.items():
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ OpenCV imported successfully (v{cv2.__version__})")
                
                # Test HOG detector
                hog = cv2.HOGDescriptor()
                hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
                print("✅ HOG People Detector initialized successfully")
                
            elif package == 'numpy':
                import numpy as np
                print(f"✅ NumPy imported successfully (v{np.__version__})")
            else:
                exec(f"import {package}")
                print(f"✅ {package} imported successfully")
                
        except ImportError:
            print(f"❌ {package} not found")
            if install_name != 'built-in':
                missing_packages.append(install_name)
        except Exception as e:
            print(f"⚠️ {package} imported but error: {e}")
    
    if missing_packages:
        print(f"\n📦 To install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n🎉 PHASE 1 COMPLETE!")
        print("All dependencies are working correctly.")
        print("Ready to move to Phase 2: Basic Person Detection")
        return True

if __name__ == "__main__":
    success = test_dependencies()
    sys.exit(0 if success else 1)
