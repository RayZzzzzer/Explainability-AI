# Data Directory

This directory stores uploaded files and temporary data.

## Structure

```
data/
├── audio_uploads/    # Uploaded audio files (.wav)
└── image_uploads/    # Uploaded images (.png, .jpg, etc.)
```

## Usage

- **audio_uploads/**: Temporary storage for uploaded audio files
- **image_uploads/**: Temporary storage for uploaded images

## Notes

- Files in these directories are temporary
- Old files are automatically cleaned up
- These directories are excluded from git (see `.gitignore`)
- The application automatically creates these directories if they don't exist

## Sample Data

To test the platform, you can:

### For Audio:
1. Copy sample files from `../Audio_real/` or `../Audio_fake/`
2. Or use your own .wav audio files

### For Images:
1. Copy sample files from `../LungCancerDetection/images/`
2. Or use your own chest X-ray images

## Privacy Note

Do not commit actual patient data or sensitive audio recordings to version control. This directory is git-ignored for privacy reasons.
