import zipfile
import os

zip_path = r"C:\Users\renoi\Downloads\archive (3).zip"
extract_path = r"c:\XAI\Pitié\chexpert_sample"

# Create extraction directory
os.makedirs(extract_path, exist_ok=True)

print("Opening ZIP file...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # List all files in the archive
    all_files = zip_ref.namelist()
    print(f"Total files in archive: {len(all_files)}")
    
    # Extract CSV files
    csv_files = [f for f in all_files if f.lower().endswith('.csv')]
    print(f"\nExtracting {len(csv_files)} CSV files...")
    for csv_file in csv_files:
        zip_ref.extract(csv_file, extract_path)
        print(f"  ✓ {csv_file}")
    
    # Filter for train images
    train_images = [f for f in all_files if 'train/' in f and f.lower().endswith(('.jpg', '.jpeg', '.png', '.dcm'))]
    print(f"\nFound {len(train_images)} train images")
    
    # Filter for valid images
    valid_images = [f for f in all_files if 'valid/' in f and f.lower().endswith(('.jpg', '.jpeg', '.png', '.dcm'))]
    print(f"Found {len(valid_images)} valid images")
    
    # Extract 500 samples from train
    sample_size = 500
    train_sample = train_images[:sample_size]
    
    print(f"\nExtracting {len(train_sample)} train images...")
    for i, file in enumerate(train_sample, 1):
        zip_ref.extract(file, extract_path)
        if i % 50 == 0:
            print(f"  Extracted {i}/{len(train_sample)} train files...")
    
    # Extract 500 samples from valid
    valid_sample = valid_images[:sample_size]
    
    print(f"\nExtracting {len(valid_sample)} valid images...")
    for i, file in enumerate(valid_sample, 1):
        zip_ref.extract(file, extract_path)
        if i % 50 == 0:
            print(f"  Extracted {i}/{len(valid_sample)} valid files...")
    
    print(f"\n✓ Successfully extracted:")
    print(f"  - {len(csv_files)} CSV files")
    print(f"  - {len(train_sample)} train images")
    print(f"  - {len(valid_sample)} valid images")
    print(f"\nLocation: {extract_path}")
