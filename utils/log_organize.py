#!/usr/bin/env python3
import os
import shutil
from datetime import datetime
import argparse


def organize_files_by_timestamp(directory):
    # Ensure the directory exists
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    for entry in os.listdir(directory):

        if not entry.endswith(".log"):
            continue

        file_path = os.path.join(directory, entry)

        if os.path.isfile(file_path):
            timestamp = os.path.getmtime(file_path)
            # Convert timestamp to a date string (e.g., "2025-03-11")
            date_folder = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")
            target_folder = os.path.join(directory, "log-" + date_folder[2:])

            # Create target folder if it doesn't exist
            os.makedirs(target_folder, exist_ok=True)

            # Define the target file path
            target_path = os.path.join(target_folder, entry)
            print(f"Moving '{file_path}' to '{target_path}'")
            shutil.move(file_path, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize files in a directory into folders by their last modification date.")
    parser.add_argument("directory", help="Path to the directory to organize")
    args = parser.parse_args()

    organize_files_by_timestamp(args.directory)
