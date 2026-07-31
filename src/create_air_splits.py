import argparse
import shutil
from pathlib import Path


def create_air_splits(air_source_dir, output_dir):
    source_path = Path(air_source_dir)
    out_path = Path(output_dir)

    # 1. Strict definition of the split based on rooms
    splits = {
        "train": ["booth", "office", "kitchen", "stairway"],
        "validation": ["meeting", "bathroom"],
        "test": [
            "lecture",
            "lecutre",
        ],  # The typo is intentional since the data actually has this typo in some filenames
    }

    # 2. Output folders creation
    for split_name in splits:
        (out_path / split_name).mkdir(parents=True, exist_ok=True)

    copied_counts = {"train": 0, "validation": 0, "test": 0}

    # 3. Files scanning and copying
    print(f"Reading RIR files from: {source_path}")

    # Search for all .wav files (including subdirectories if present)
    found_files = 0
    for wav_path in source_path.glob("**/*.wav"):
        found_files += 1
        filename_lower = wav_path.name.lower()

        # Search for the keyword in the filename to identify the room
        for split_name, keywords in splits.items():
            if any(keyword in filename_lower for keyword in keywords):
                # Copy the file to the corresponding folder
                dest_path = out_path / split_name / wav_path.name
                shutil.copy2(wav_path, dest_path)
                copied_counts[split_name] += 1
                break  # Move to the next file once the split is found

    # 4. Final report
    print("\n--- Operation Completed ---")
    print(f"Total files analyzed: {found_files}")
    print(f"Files copied to Train: {copied_counts['train']}")
    print(f"Files copied to Validation:   {copied_counts['validation']}")
    print(f"Files copied to Test:  {copied_counts['test']}")

    # Safety check: report any ignored files
    total_copied = sum(copied_counts.values())
    if total_copied < found_files:
        print(
            f"\nWARNING: {found_files - total_copied} files were not assigned to any split."
        )
        print("They might have names that do not contain any of the keywords.")

    print(f"\nYour split RIRs are ready at: {out_path.absolute()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split AIR dataset RIRs into train, validation, and test sets."
    )

    # Define command line arguments
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="The directory where the AIR dataset was originally extracted",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The directory where the 3 splits will be created",
    )

    # Parse the arguments
    args = parser.parse_args()

    create_air_splits(args.source_dir, args.output_dir)
