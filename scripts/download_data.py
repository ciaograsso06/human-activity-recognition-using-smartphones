from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones.zip"
DATA_DIR = Path("data")
TARGET = DATA_DIR / "UCI HAR Dataset"
ARCHIVE = DATA_DIR / "UCI_HAR_Dataset.zip"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if TARGET.exists():
        print(f"Dataset already exists at: {TARGET}")
        return

    print("Downloading UCI HAR Dataset from the official UCI repository...")
    urllib.request.urlretrieve(URL, ARCHIVE)

    extract_dir = DATA_DIR / "_uci_har_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    with zipfile.ZipFile(ARCHIVE, "r") as archive:
        archive.extractall(extract_dir)

    direct = extract_dir / "UCI HAR Dataset"
    if direct.exists():
        shutil.move(str(direct), str(TARGET))
    else:
        nested_archives = list(extract_dir.rglob("UCI HAR Dataset.zip"))
        if not nested_archives:
            raise RuntimeError("Could not locate 'UCI HAR Dataset' in downloaded archive.")
        with zipfile.ZipFile(nested_archives[0], "r") as nested:
            nested.extractall(DATA_DIR)

    shutil.rmtree(extract_dir, ignore_errors=True)
    ARCHIVE.unlink(missing_ok=True)
    print(f"Dataset ready at: {TARGET}")


if __name__ == "__main__":
    main()
