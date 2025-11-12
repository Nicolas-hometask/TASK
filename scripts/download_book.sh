#!/usr/bin/env bash
set -e

# Default: Pride and Prejudice by Jane Austen (Gutenberg ID 1342)
BOOK_ID=${1:-1342}
OUTPUT_FILE="data/book.txt"

mkdir -p data
echo "** Downloading book ${BOOK_ID} (Pride and Prejudice) from Project Gutenberg..."
curl -sL "https://www.gutenberg.org/files/${BOOK_ID}/${BOOK_ID}-0.txt" -o "$OUTPUT_FILE" \
  || curl -sL "https://www.gutenberg.org/files/${BOOK_ID}/${BOOK_ID}.txt.utf-8" -o "$OUTPUT_FILE"

echo "++ Saved to $OUTPUT_FILE"
