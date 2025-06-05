#!/usr/bin/env python3
"""
TEST GARMENT LOADING
===================
Check if garments are loaded correctly.
"""

import cv2
import os

def test_garments():
    """Test loading and displaying garments."""
    print("Testing garment loading...")
    
    garment_folders = ['Garments/tops/', 'Shirts/']
    
    for folder in garment_folders:
        print(f"\n📁 Checking folder: {folder}")
        
        if not os.path.exists(folder):
            print(f"❌ Folder does not exist: {folder}")
            continue
            
        for filename in ['1.png', '3.png', '4.png']:
            filepath = os.path.join(folder, filename)
            
            if os.path.exists(filepath):
                img = cv2.imread(filepath)
                
                if img is not None:
                    h, w = img.shape[:2]
                    print(f"✅ {filepath}: {w}x{h} pixels")
                    
                    # Show the image
                    cv2.imshow(f'{folder}{filename}', img)
                    cv2.waitKey(1000)  # Show for 1 second
                    cv2.destroyAllWindows()
                else:
                    print(f"❌ Failed to load: {filepath}")
            else:
                print(f"❌ File not found: {filepath}")

if __name__ == "__main__":
    test_garments()
    print("\n✅ Garment test complete!")
