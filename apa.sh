#!/bin/bash

# Directory containing the Python files
DIR="part2"

# Output file
OUTPUT_FILE="output.txt"

# Check if the output file already exists and remove it to start fresh
if [ -f "$OUTPUT_FILE" ]; then
    rm "$OUTPUT_FILE"
fi

# Loop through all Python files in the directory
for file in "$DIR"/*.py; do
    echo "Filename: $(basename "$file")" >> "$OUTPUT_FILE"
    echo "----------------------------------------" >> "$OUTPUT_FILE"
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "\n" >> "$OUTPUT_FILE"
done

echo "Content of all Python files has been written to $OUTPUT_FILE"
