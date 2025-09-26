#!/usr/bin/env python3
"""
Test script for CSV transformation functions
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from canlock.data.csv_transform_data import transform_single_csv, transform_all_csv_files

def test_single_file():
    """Test transformation of a single CSV file"""
    print("Testing single file transformation...")
    
    data_dir = Path(__file__).parent / "data" / "heavy_truck_data"
    test_file = data_dir / "part_1" / "part_1" / "20201123075441304067.csv"
    
    if test_file.exists():
        df = transform_single_csv(test_file)
        print(f"Successfully transformed {test_file.name}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head(3))
        print(f"Timestamp dtype: {df['timestamp'].dtype}")
        print()
        return True
    else:
        print(f"Test file not found: {test_file}")
        return False

def test_all_files():
    """Test transformation of all CSV files (just count them)"""
    print("Testing all files transformation...")
    
    data_dir = Path(__file__).parent / "data" / "heavy_truck_data"
    
    # Just find and count files without transforming all (to save time)
    csv_files = list(data_dir.rglob("*.csv"))
    print(f"Found {len(csv_files)} CSV files total")
    
    # Transform just the first few files as a test
    test_files = csv_files[:3] if len(csv_files) > 3 else csv_files
    
    transformed_count = 0
    for csv_file in test_files:
        try:
            df = transform_single_csv(csv_file)
            transformed_count += 1
            print(f"✓ Transformed {csv_file.name} -> {df.shape[0]} rows")
        except Exception as e:
            print(f"✗ Error transforming {csv_file.name}: {e}")
    
    print(f"Successfully tested {transformed_count}/{len(test_files)} files")

if __name__ == "__main__":
    print("CSV Transformation Test")
    print("=" * 40)
    
    success = test_single_file()
    if success:
        test_all_files()
    else:
        print("Single file test failed, skipping bulk test")