import os
import multiprocessing
import shutil
import csv
import tempfile

num_cores = multiprocessing.cpu_count()
dataset_dir = './data/ocr_data'
num_samples = 0
font_dir = './data/fonts'
corpus = "./data/ckb_corpus.txt"
lang = "ckb"


NO_SETUPS = 7
with open(corpus, "r") as f:
    lines = f.read().splitlines()
    num_samples = int(len(lines) / NO_SETUPS)
    remainder_samples = int(len(lines) % NO_SETUPS)




output_dirs = []
for i in range(NO_SETUPS + 1):
    output_dir = tempfile.mkdtemp(prefix="batch_")
    output_dirs.append(output_dir)


# Normal white background skew angle 0-2, random blue 0-1 )
os.system(f"python3 ./trdg/run.py -i {corpus} -t {num_cores} -f 64 -l {lang} -fd {font_dir} \
    -c {num_samples} -na 2 --output_dir {output_dirs[0]} -b 1 -rk -k 2 -rbl -bl 1")

skip_lines = num_samples 
# Normal white background skew angle 1, random blur 0-1 sine wave distortion )
os.system(f"python3 ./trdg/run.py -i {corpus} -t {num_cores} -f 64 -l {lang} -fd {font_dir} \
    -c {num_samples} -na 2 --output_dir {output_dirs[1]} -b 1 -rk -k 2 -rbl -bl 1 -d 1 -do 2 -sl {skip_lines}")

skip_lines += num_samples 
# Normal white background skew angle 1, random blur 0-1 random distortion ) (Lots of noise)
os.system(f"python3 ./trdg/run.py -i {corpus} -t {num_cores} -f 64 -l {lang} -fd {font_dir} \
    -c {num_samples} -na 2 --output_dir {output_dirs[2]} -b 1 -rk -k 2 -rbl -bl 1 -d 3 -do 2 -sl {skip_lines}")

skip_lines += num_samples 
# Quasicrystal background skew angle 1, Cosine wave distortion 
os.system(f"python3 ./trdg/run.py -i {corpus}  -t {num_cores} -f 64 -l {lang} -fd {font_dir} \
    -c {num_samples} -na 2 --output_dir {output_dirs[3]} -b 2 -k 1 -rbl -bl 0 -d 2 -do 2 -sl {skip_lines}")

skip_lines += num_samples 
# Image background skew angle 1, Cosine wave random )
os.system(f"python3 ./trdg/run.py -i {corpus}  -t {num_cores} -f 64 -l {lang} -fd {font_dir} \
    -c {num_samples} -na 2 --output_dir {output_dirs[4]} -b 3 -k 1 -rbl -bl 0 -d 2 -do -sl {skip_lines}")

skip_lines += num_samples 
# Gaussian blur background skew angle 1, random blur 0-1 sine wave distortion 
os.system(f"python3 ./trdg/run.py -i {corpus} -t {num_cores} -f 64 -l {lang} -fd {font_dir} \
    -c {num_samples} -na 2 --output_dir {output_dirs[5]} -b 0 -rk -k 2 -rbl -bl 1 -d 1 -do 2 -sl {skip_lines}")

skip_lines += num_samples
# Gaussian background skew angle 1, random blur 0-1 random distortion  (Lots of noise)
os.system(f"python3 ./trdg/run.py -i {corpus} -t {num_cores} -f 64 -l {lang} -fd {font_dir} \
    -c {num_samples} -na 2 --output_dir {output_dirs[6]} -b 0 -rk -k 2 -rbl -bl 1 -d 3 -do 2 -sl {skip_lines}")


skip_lines += num_samples
# Gaussian background skew angle 1, random blur 0-1 random distortion  (Lots of noise)
os.system(f"python3 ./trdg/run.py -i {corpus} -t {num_cores} -f 64 -l {lang} -fd {font_dir} \
    -c {remainder_samples} -na 2 --output_dir {output_dirs[7]} -b 0 -rk -k 2 -rbl -bl 1 -d 3 -do 2 -sl {skip_lines}")


def merge_folders(source_folders, destination_folder):
    """Merge multiple folders with labeled files into a single destination folder,
    renaming the files starting from 0 and incrementing, and writing metadata to a CSV file."""
    counter = 0

    # Create destination folder
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(os.path.join(destination_folder, "train.csv"), "w", newline="") as metadata_file:
        writer = csv.writer(metadata_file)
        writer.writerow(["file_name", "text"])
        for source_folder in source_folders:
            with open(os.path.join(source_folder, "labels.txt"), "r") as labels_file:
                for line in labels_file:
                    if line.strip() == "":
                        continue
                    try:
                        filename, label = line.strip().split(" ", 1)
                    except ValueError as e:
                        print(e)
                        print("Error in line: {}'".format(line))
                    # if filename doesn't exist contine
                    if not os.path.exists(os.path.join(source_folder, filename)):
                        continue
                    source_path = os.path.join(source_folder, filename)
                    destination_filename = f"{counter:08d}.jpg"  # Use a zero-padded four-digit number as the new filename
                    destination_path = os.path.join(destination_folder, destination_filename)
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    shutil.copy(source_path, destination_path)
                    writer.writerow([destination_filename, label])
                    counter += 1



merge_folders(output_dirs, dataset_dir)

# Delete the tmp dirs
for output_dir in output_dirs:
    shutil.rmtree(output_dir)