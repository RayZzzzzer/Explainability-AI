"""
Convert audio files (.wav) to mel-spectrogram images
Preserves directory structure for use with image_dataset_from_directory()
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm


def create_clean_spectrogram(audio_path, output_path, figsize=(10, 4)):
    """
    Generate clean mel-spectrogram image without axes, labels, or decorations.
    
    Args:
        audio_path: Path to input .wav file
        output_path: Path to save spectrogram image (.png)
        figsize: Figure size (width, height)
    """
    # Load audio
    y, sr = librosa.load(audio_path)
    
    # Create mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Create figure without decorations
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    # Display spectrogram
    ax.imshow(
        mel_spec_db,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        interpolation='bilinear'
    )
    
    # Remove all margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save without borders
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def convert_dataset(input_root, output_root):
    """
    Convert all .wav files in directory structure to spectrograms.
    Preserves folder structure: training/testing/validation with fake/real subfolders.
    
    Args:
        input_root: Root directory containing audio files
        output_root: Root directory for spectrogram output
    
    Example structure:
        input_root/
        ├── training/
        │   ├── fake/*.wav
        │   └── real/*.wav
        ├── testing/
        │   ├── fake/*.wav
        │   └── real/*.wav
        └── validation/
            ├── fake/*.wav
            └── real/*.wav
    """
    input_path = Path(input_root)
    output_path = Path(output_root)
    
    # Find all .wav files
    wav_files = list(input_path.rglob("*.wav"))
    
    if not wav_files:
        print(f"⚠️ No .wav files found in {input_root}")
        return
    
    print(f"Found {len(wav_files)} .wav files")
    print(f"Converting to spectrograms...\n")
    
    # Process each audio file
    success_count = 0
    error_count = 0
    
    for wav_file in tqdm(wav_files, desc="Processing"):
        try:
            # Get relative path from input root
            rel_path = wav_file.relative_to(input_path)
            
            # Create corresponding output path
            output_file = output_path / rel_path.with_suffix('.png')
            
            # Create output directory if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert to spectrogram
            create_clean_spectrogram(str(wav_file), str(output_file))
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ Error processing {wav_file}: {e}")
            error_count += 1
    
    print(f"\n✅ Conversion complete!")
    print(f"   Success: {success_count}")
    print(f"   Errors: {error_count}")
    print(f"\nSpectrograms saved to: {output_root}")


def verify_structure(root_dir):
    """
    Verify the dataset structure and print statistics.
    
    Args:
        root_dir: Root directory to verify
    """
    root = Path(root_dir)
    
    if not root.exists():
        print(f"⚠️ Directory not found: {root_dir}")
        return
    
    print(f"\n📊 Dataset Structure: {root_dir}")
    print("=" * 60)
    
    for split in ['training', 'testing', 'validation']:
        split_path = root / split
        if split_path.exists():
            print(f"\n{split.upper()}:")
            for label in ['fake', 'real']:
                label_path = split_path / label
                if label_path.exists():
                    count = len(list(label_path.glob("*.png")))
                    print(f"  - {label}: {count} images")
                else:
                    print(f"  - {label}: directory not found")
        else:
            print(f"\n{split.upper()}: directory not found")


if __name__ == "__main__":
    # Configuration
    INPUT_DIR = "D:/WORK/xai/for-norm/for-norm"     # Change this to your audio directory
    OUTPUT_DIR = "D:/WORK/xai/spectrograms"  # Change this to output directory
    
    # Example usage:
    # INPUT_DIR = "D:/WORK/audio_data"
    # OUTPUT_DIR = "D:/WORK/spectrograms"
    
    print("=" * 60)
    print("Audio to Spectrogram Converter")
    print("=" * 60)
    
    # Check if paths are configured
    if "path/to" in INPUT_DIR or "path/to" in OUTPUT_DIR:
        print("\n⚠️ Please configure INPUT_DIR and OUTPUT_DIR in the script!")
        print("\nExample:")
        print('  INPUT_DIR = "D:/WORK/audio_data"')
        print('  OUTPUT_DIR = "D:/WORK/spectrograms"')
        exit(1)
    
    # Verify input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"\n❌ Input directory not found: {INPUT_DIR}")
        exit(1)
    
    # Convert dataset
    convert_dataset(INPUT_DIR, OUTPUT_DIR)
    
    # Verify output
    verify_structure(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ Ready to use with image_dataset_from_directory()!")
    print("=" * 60)
    print("\nExample code:")
    print(f"""
import tensorflow.keras as K
import numpy as np

root = r'{OUTPUT_DIR}'
batch_size = 32
img_height = img_width = 224

train_ds = K.utils.image_dataset_from_directory(
    str(root),
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

test_ds = K.utils.image_dataset_from_directory(
    str(root),
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = np.array(train_ds.class_names)
print(class_names)
""")
