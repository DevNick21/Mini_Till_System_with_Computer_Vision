from pathlib import Path

from dotenv import load_dotenv
import os

# current directory
parent_dir = Path.cwd()

load_dotenv()
old_names = os.getenv("OLD_NAMES")

start_index = 3  # first anon index

for i, old_name in enumerate(old_names, start=start_index):
    new_name = f"anon{i}"
    child_dir = parent_dir / old_name
    if child_dir.exists() and child_dir.is_dir():
        for file in child_dir.iterdir():
            if file.is_file() and file.name.startswith(old_name):
                new_file = file.with_name(
                    file.name.replace(old_name, new_name, 1))
                file.rename(new_file)
                print(f"Renamed file {file} -> {new_file}")

        new_child_dir = child_dir.with_name(new_name)
        child_dir.rename(new_child_dir)
        print(f"Renamed directory {child_dir} -> {new_child_dir}")
