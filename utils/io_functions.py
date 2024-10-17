import os
import requests
from tqdm import tqdm
from pathlib import Path

def download_file(url, path):
    if os.path.exists(path):
        return

    print(f"Downloading File {url} to {path}")

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(path, "wb") as file:
            for data in r.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)



def find_all_images_in_directory(search_dir):
    search_dir = Path(search_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    image_files = [str(file) for file in search_dir.rglob('*') if file.suffix.lower() in image_extensions]
    return image_files


def mirror_data_folder_structure(data, output_dir):
    try:
        root_dataset_path = os.path.commonpath(data)
        for image_file in data:
            # Convert to Path object for easier manipulation
            image_file_path = Path(image_file)
            
            # Extract relative path with respect to the root dataset path
            relative_path = image_file_path.relative_to(root_dataset_path)
            
            # Define the output path maintaining the relative path structure
            output_image_path = output_dir / relative_path
            
            # Create the necessary directories in the output path
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
    except:
        pass
    return output_dir
