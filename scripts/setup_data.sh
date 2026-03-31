#!/bin/bash
# Download UCF101 video dataset (~7GB)
# Run this ONCE on the VM before training

set -e

DATA_DIR="./data/ucf101"
mkdir -p "$DATA_DIR"

echo "Downloading UCF101..."
cd "$DATA_DIR"

# Download from official mirror
if [ ! -f "UCF101.rar" ]; then
    wget -q --show-progress "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar" -O UCF101.rar
    echo "Extracting..."
    sudo apt-get install -y unrar -qq
    unrar x -o+ UCF101.rar .
    echo "Extracted to $DATA_DIR/UCF-101/"
else
    echo "UCF101.rar already exists, skipping download"
fi

# Count videos
N_VIDEOS=$(find "$DATA_DIR/UCF-101" -name "*.avi" | wc -l)
echo "UCF101 ready: $N_VIDEOS videos in $DATA_DIR/UCF-101/"
