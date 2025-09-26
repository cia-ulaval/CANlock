import pandas as pd
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from multiprocessing import Pool, cpu_count
import os


def transform_single_csv(input_file_path: Path) -> pd.DataFrame:
    """
    Transform a single CSV file using the specified logic.
    
    Args:
        input_file_path: Path to the input CSV file
        
    Returns:
        pd.DataFrame: Transformed dataframe with timestamp, id, dlc, and data columns
    """
    correct_pandas_format_data = []
    
    with open(input_file_path, mode="r") as f:
        csv_reader = csv.reader(f, delimiter=";")

        for row_id, row in enumerate(csv_reader):
            if row_id == 0:  # Ne nous ennuyons pas avec le header, nous le connaissons
                continue
            
            correct_pandas_format_data.append(
                {
                    "timestamp": row[0],
                    "id": row[1],
                    "dlc": row[2],
                    "data": row[3:],
                }
            )

    df = pd.DataFrame(correct_pandas_format_data)
    
    # Convertir la colonne timestamp en datetime with explicit format
    df["timestamp"] = pd.to_datetime(df["timestamp"], format='%Y-%m-%d %H:%M:%S.%f')
    
    return df


def _process_single_file(args: Tuple[Path, Path, Path]) -> Tuple[bool, str, str]:
    """
    Process a single CSV file for multiprocessing.
    
    Args:
        args: Tuple of (csv_file, data_directory, output_directory)
        
    Returns:
        Tuple of (success, relative_path, error_message)
    """
    csv_file, data_directory, output_directory = args
    
    try:
        # Transform the CSV file
        df_transformed = transform_single_csv(csv_file)
        
        # Create output file path maintaining directory structure
        relative_path = csv_file.relative_to(data_directory)
        output_file = output_directory / relative_path.parent / f"{csv_file.stem}_bis.csv"
        
        # Create subdirectories if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save transformed data
        df_transformed.to_csv(output_file, index=False)
        
        row_count = len(df_transformed)
        
        # Clear dataframe from memory immediately
        del df_transformed
        
        return True, str(relative_path), f"✓ Saved: {output_file.name} ({row_count:,} rows)"
        
    except Exception as e:
        return False, str(csv_file.name), f"✗ Error: {str(e)}"


def transform_all_csv_files_multiprocessing_with_progress(
    data_directory: Path, 
    output_directory: Path, 
    num_processes: Optional[int] = None,
    batch_size: int = 100
) -> List[str]:
    """
    Transform all CSV files using multiprocessing with progress tracking and batching.
    
    Args:
        data_directory: Path to the data directory containing CSV files
        output_directory: Path to save transformed CSV files
        num_processes: Number of processes to use (default: number of CPU cores)
        batch_size: Number of files to process in each batch
        
    Returns:
        List[str]: List of successfully transformed file paths
    """
    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    
    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files recursively
    csv_files = list(data_directory.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_directory}")
        return []
    
    # Set number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), 8)  # Limit to 8 processes to avoid overwhelming
    
    print(f"🚀 Found {len(csv_files)} CSV files to transform")
    print(f"💻 Using {num_processes} processes in batches of {batch_size}")
    print(f"📁 Output directory: {output_directory}")
    print()
    
    successfully_transformed = []
    failed_files = []
    
    # Process files in batches
    total_batches = (len(csv_files) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(csv_files))
        batch_files = csv_files[start_idx:end_idx]
        
        print(f"📦 Processing batch {batch_num + 1}/{total_batches} ({start_idx + 1}-{end_idx} of {len(csv_files)})")
        
        # Prepare arguments for this batch
        process_args = [(csv_file, data_directory, output_directory) for csv_file in batch_files]
        
        try:
            # Use multiprocessing Pool for this batch
            with Pool(processes=num_processes) as pool:
                batch_results = pool.map(_process_single_file, process_args)
                
                # Process batch results
                batch_success = 0
                for success, file_path, message in batch_results:
                    if success:
                        successfully_transformed.append(file_path)
                        batch_success += 1
                    else:
                        failed_files.append(file_path)
                        print(message)
                
                print(f"✅ Batch {batch_num + 1} complete: {batch_success}/{len(batch_files)} files successful")
                print(f"📊 Total progress: {len(successfully_transformed)}/{len(csv_files)} files ({len(successfully_transformed)/len(csv_files)*100:.1f}%)")
                print()
        
        except KeyboardInterrupt:
            print(f"\n⚠️  Processing interrupted by user at batch {batch_num + 1}")
            break
        except Exception as e:
            print(f"❌ Batch {batch_num + 1} error: {e}")
            continue
    
    print(f"\n🎉 Processing complete!")
    print(f"✅ Successfully transformed: {len(successfully_transformed)}/{len(csv_files)} files")
    if failed_files:
        print(f"❌ Failed files: {len(failed_files)}")
    
    return successfully_transformed


def transform_all_csv_files_multiprocessing(
    data_directory: Path, 
    output_directory: Path, 
    num_processes: Optional[int] = None
) -> List[str]:
    """
    Transform all CSV files using multiprocessing for faster processing.
    
    Args:
        data_directory: Path to the data directory containing CSV files
        output_directory: Path to save transformed CSV files
        num_processes: Number of processes to use (default: number of CPU cores)
        
    Returns:
        List[str]: List of successfully transformed file paths
    """
    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    
    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files recursively
    csv_files = list(data_directory.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_directory}")
        return []
    
    # Set number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), len(csv_files))  # Don't use more processes than files
    
    print(f"🚀 Found {len(csv_files)} CSV files to transform using {num_processes} processes")
    print(f"📁 Output directory: {output_directory}")
    print()
    
    # Prepare arguments for multiprocessing
    process_args = [(csv_file, data_directory, output_directory) for csv_file in csv_files]
    
    successfully_transformed = []
    failed_files = []
    
    try:
        # Use multiprocessing Pool
        with Pool(processes=num_processes) as pool:
            # Process files in parallel
            results = pool.map(_process_single_file, process_args)
            
            # Process results
            for success, file_path, message in results:
                print(message)
                if success:
                    successfully_transformed.append(file_path)
                else:
                    failed_files.append(file_path)
    
    except Exception as e:
        print(f"❌ Multiprocessing error: {e}")
        return []
    
    print(f"\n🎉 Multiprocessing complete!")
    print(f"✅ Successfully transformed: {len(successfully_transformed)}/{len(csv_files)} files")
    if failed_files:
        print(f"❌ Failed files: {len(failed_files)}")
        for failed_file in failed_files[:5]:  # Show first 5 failed files
            print(f"   - {failed_file}")
        if len(failed_files) > 5:
            print(f"   ... and {len(failed_files) - 5} more")
    
    return successfully_transformed


def transform_all_csv_files(data_directory: Path, output_directory: Path) -> List[str]:
    """
    Transform all CSV files in the data directory using the specified logic.
    Processes files one at a time to handle large datasets efficiently.
    
    Args:
        data_directory: Path to the data directory containing CSV files
        output_directory: Path to save transformed CSV files (required for large data handling)
        
    Returns:
        List[str]: List of successfully transformed file paths
    """
    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    
    if not output_directory:
        raise ValueError("output_directory is required when handling large datasets")
    
    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files recursively
    csv_files = list(data_directory.glob("**/*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_directory}")
        return []
    
    print(f"Found {len(csv_files)} CSV files to transform")
    print(f"Output directory: {output_directory}")
    
    successfully_transformed = []
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            print(f"[{i}/{len(csv_files)}] Transforming: {csv_file.name}")
            
            # Transform the CSV file
            df_transformed = transform_single_csv(csv_file)
            
            # Create output file path maintaining directory structure
            relative_path = csv_file.relative_to(data_directory)
            output_file = output_directory / relative_path.parent / f"{csv_file.stem}_bis.csv"
            
            # Create subdirectories if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save transformed data
            df_transformed.to_csv(output_file, index=False)
            print(f"✓ Saved: {output_file} ({len(df_transformed):,} rows)")
            
            successfully_transformed.append(str(relative_path))
            
            # Clear dataframe from memory immediately
            del df_transformed
            
        except Exception as e:
            print(f"✗ Error processing {csv_file.name}: {e}")
            continue
    
    print(f"\n🎉 Successfully transformed {len(successfully_transformed)}/{len(csv_files)} files")
    return successfully_transformed


def get_transformation_summary(output_directory: Path) -> Dict[str, Any]:
    """
    Get summary statistics of transformed files without loading them into memory.
    
    Args:
        output_directory: Path to directory containing transformed CSV files
        
    Returns:
        Dict with summary statistics
    """
    if not output_directory.exists():
        return {"error": f"Output directory not found: {output_directory}"}
    
    transformed_files = list(output_directory.rglob("*_transformed.csv"))
    
    if not transformed_files:
        return {"total_files": 0, "message": "No transformed files found"}
    
    total_rows = 0
    file_info = []
    
    print(f"Analyzing {len(transformed_files)} transformed files...")
    
    for csv_file in transformed_files:
        try:
            # Count rows efficiently without loading full DataFrame
            with open(csv_file, 'r') as f:
                row_count = sum(1 for line in f) - 1  # Subtract header
            
            file_size = csv_file.stat().st_size / (1024 * 1024)  # MB
            
            file_info.append({
                "file": csv_file.name,
                "rows": row_count,
                "size_mb": round(file_size, 2)
            })
            
            total_rows += row_count
            
        except Exception as e:
            print(f"Error analyzing {csv_file.name}: {e}")
            continue
    
    return {
        "total_files": len(file_info),
        "total_rows": total_rows,
        "files": file_info,
        "average_rows_per_file": round(total_rows / len(file_info)) if file_info else 0,
        "total_size_mb": round(sum(f["size_mb"] for f in file_info), 2)
    }


def process_csv_files_in_batches(data_directory: Path, output_directory: Path, batch_size: int = 10) -> List[str]:
    """
    Process CSV files in batches for even better memory management with very large datasets.
    
    Args:
        data_directory: Path to the data directory containing CSV files
        output_directory: Path to save transformed CSV files
        batch_size: Number of files to process in each batch
        
    Returns:
        List[str]: List of successfully transformed file paths
    """
    if not data_directory.exists():
        raise FileNotFoundError(f"Data directory not found: {data_directory}")
    
    # Create output directory
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files recursively
    csv_files = list(data_directory.rglob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {data_directory}")
        return []
    
    print(f"Found {len(csv_files)} CSV files to transform in batches of {batch_size}")
    
    successfully_transformed = []
    
    # Process files in batches
    for batch_start in range(0, len(csv_files), batch_size):
        batch_end = min(batch_start + batch_size, len(csv_files))
        batch_files = csv_files[batch_start:batch_end]
        
        print(f"\n--- Processing batch {batch_start//batch_size + 1} ({batch_start + 1}-{batch_end} of {len(csv_files)}) ---")
        
        for i, csv_file in enumerate(batch_files):
            try:
                file_num = batch_start + i + 1
                print(f"[{file_num}/{len(csv_files)}] Transforming: {csv_file.name}")
                
                # Transform the CSV file
                df_transformed = transform_single_csv(csv_file)
                
                # Create output file path maintaining directory structure
                relative_path = csv_file.relative_to(data_directory)
                output_file = output_directory / relative_path.parent / f"{csv_file.stem}_transformed.csv"
                
                # Create subdirectories if needed
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Save transformed data
                df_transformed.to_csv(output_file, index=False)
                print(f"✓ Saved: {output_file} ({len(df_transformed):,} rows)")
                
                successfully_transformed.append(str(relative_path))
                
                # Clear dataframe from memory immediately
                del df_transformed
                
            except Exception as e:
                print(f"✗ Error processing {csv_file.name}: {e}")
                continue
        
        print(f"Batch complete. Processed {len([f for f in batch_files if str(f.relative_to(data_directory)) in successfully_transformed[-len(batch_files):]])}/{len(batch_files)} files successfully")
    
    print(f"\n🎉 Successfully transformed {len(successfully_transformed)}/{len(csv_files)} files total")
    return successfully_transformed


def transform_csv(input_file: Path, output_file: Path) -> None:
    """
    Legacy function - transforms a single CSV file and saves to output file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    df_transformed = transform_single_csv(input_file)
    df_transformed.to_csv(output_file, index=False)
    print(f"Transformed {input_file} -> {output_file}")


if __name__ == "__main__":
    # Example usage for large dataset handling
    data_dir = Path(__file__).parent.parent.parent.parent / "data" / "heavy_truck_data"
    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "transformed"
    
    print("🚛 Heavy Truck Data CSV Transformation")
    print("=" * 50)
    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"💻 Available CPU cores: {cpu_count()}")
    print()
    
    # Choose processing method
    use_multiprocessing = True  # Set to False for sequential processing
    
    if use_multiprocessing:
        print("🚀 Using multiprocessing with progress tracking...")
        # Transform all CSV files using multiprocessing with batching (ROBUST)
        transformed_files = transform_all_csv_files_multiprocessing_with_progress(
            data_dir, 
            output_dir, 
            num_processes=min(cpu_count(), 8),  # Limit processes to avoid overwhelming
            batch_size=50  # Process in smaller batches
        )
    else:
        print("🐌 Using sequential processing...")
        # Transform all CSV files sequentially (memory-efficient but slower)
        transformed_files = transform_all_csv_files(data_dir, output_dir)
    
    print(f"\n✅ Transformation complete!")
    print(f"📁 Transformed files are saved in: {output_dir}")
    print(f"📊 Total files processed: {len(transformed_files)}")
    
    # Get summary statistics without loading data into memory
    if transformed_files:
        print("\n📈 Getting transformation summary...")
        summary = get_transformation_summary(output_dir)
        if 'total_rows' in summary:
            print(f"📋 Total rows across all files: {summary['total_rows']:,}")
            print(f"💾 Total size: {summary['total_size_mb']} MB")
            print(f"📊 Average rows per file: {summary['average_rows_per_file']:,}")
    
    # Alternative options:
    # 1. Use fewer processes for multiprocessing:
    # transformed_files = transform_all_csv_files_multiprocessing(data_dir, output_dir, num_processes=4)
    # 
    # 2. Use batch processing for extremely large datasets:
    # transformed_files = process_csv_files_in_batches(data_dir, output_dir, batch_size=5)