#!/usr/bin/env python3
"""
Screenshot Test Script
Tests that all main virtual try-on versions save screenshots to the correct folder.
"""

import os
import sys

def test_screenshot_paths():
    """Verify that all versions use Screenshots/ folder for saving"""
    
    print("🧪 TESTING SCREENSHOT PATHS")
    print("=" * 40)
    
    files_to_check = [
        "simple_perfect_tryon_1.py",
        "simple_perfect_tryon_2.py", 
        "actually_working_fixed_tryon.py",
        "ultimate_precision_tryon.py",
        "super_accurate_perfect_fit.py"
    ]
    
    results = []
    
    for filename in files_to_check:
        if os.path.exists(filename):
            print(f"\n📄 Checking {filename}...")
            
            with open(filename, 'r') as f:
                content = f.read()
                
            # Check if it has the correct screenshot logic
            if 'os.makedirs("Screenshots"' in content and 'Screenshots/' in content:
                print(f"   ✅ Uses Screenshots/ folder")
                results.append(f"✅ {filename}: FIXED")
            elif 'cv2.imwrite' in content:
                print(f"   ❌ Still saves to root folder")
                results.append(f"❌ {filename}: NEEDS FIX")
            else:
                print(f"   ⚪ No screenshot functionality found")
                results.append(f"⚪ {filename}: NO SCREENSHOTS")
        else:
            print(f"\n❌ {filename} not found")
            results.append(f"❌ {filename}: NOT FOUND")
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 SCREENSHOT TEST RESULTS:")
    print("=" * 40)
    
    for result in results:
        print(f"   {result}")
    
    fixed_count = len([r for r in results if "FIXED" in r])
    total_count = len([r for r in results if "NOT FOUND" not in r])
    
    print(f"\n📈 Summary: {fixed_count}/{total_count} files properly save to Screenshots/")
    
    if fixed_count == total_count:
        print("🎉 ALL FILES SAVE SCREENSHOTS TO CORRECT FOLDER!")
    else:
        print("⚠️  Some files need fixing")
    
    return fixed_count == total_count

if __name__ == "__main__":
    success = test_screenshot_paths()
    
    if success:
        print("\n🚀 Ready to use! All screenshots will be saved to Screenshots/ folder")
    else:
        print("\n🔧 Some files may need manual fixing")
    
    sys.exit(0 if success else 1)
