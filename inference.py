import argparse
from transformers import pipeline

ocr = pipeline("image-to-text", model="razhan/trocr-base-ckb")


def main():
    parser = argparse.ArgumentParser(
        description="Perform OCR inference on one or more files."
    )

    # Add a positional argument for the file paths
    parser.add_argument("--files", nargs="+", help="File(s) to perform OCR on.")
    args = parser.parse_args()

    # Perform OCR on each file
    for file_path in args.files:
        print(f"Extracting text from {file_path}:")
        results = ocr(file_path)
        print(results[0]["generated_text"])


if __name__ == "__main__":
    main()
