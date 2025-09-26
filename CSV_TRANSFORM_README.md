# CSV Transformation Functions

This module provides functions to transform CSV files from the heavy truck data format into a standardized pandas DataFrame format.

## Usage Examples

### Transform a single CSV file

```python
from pathlib import Path
from canlock.data.csv_transform_data import transform_single_csv

# Transform one file
input_file = Path("data/heavy_truck_data/part_1/part_1/20201123075441304067.csv")
df = transform_single_csv(input_file)

print(f"Transformed data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())
```

### Transform all CSV files in the data directory

```python
from pathlib import Path
from canlock.data.csv_transform_data import transform_all_csv_files

# Transform all files and save to output directory
data_dir = Path("data/heavy_truck_data")
output_dir = Path("data/transformed")

transformed_data = transform_all_csv_files(data_dir, output_dir)

print(f"Transformed {len(transformed_data)} files")
```

### Combine all transformed data

```python
from canlock.data.csv_transform_data import (
    transform_all_csv_files, 
    combine_all_transformed_data
)

# Transform and combine all data
data_dir = Path("data/heavy_truck_data")
transformed_data = transform_all_csv_files(data_dir)
combined_df = combine_all_transformed_data(transformed_data)

print(f"Combined dataframe shape: {combined_df.shape}")
print(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
```

## Data Format

The transformation converts CSV files from this format:
```
timestamp;id;dlc;data
2020-11-23 08:03:31.985194;0xcf003e6;8;255;255;255;255;255;255;255;255
```

To a pandas DataFrame with columns:
- `timestamp`: datetime64[ns] - Converted to pandas datetime
- `id`: object - CAN ID (hex string)
- `dlc`: object - Data Length Code
- `data`: object - List of data bytes

## Functions

- `transform_single_csv(input_file_path)`: Transform one CSV file
- `transform_all_csv_files(data_directory, output_directory=None)`: Transform all CSV files
- `combine_all_transformed_data(transformed_data)`: Combine multiple DataFrames
- `transform_csv(input_file, output_file)`: Legacy function for single file transformation