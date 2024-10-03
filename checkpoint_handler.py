import os
import re
import json
import time
from huggingface_hub import HfApi, upload_file, upload_folder, hf_hub_download
import shutil

hf_token = "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"
repo_name = "anonymguy/ehm"

def find_latest_checkpoint(log_folder):
    done_files = [f for f in os.listdir(log_folder) if f.startswith("DONE_")]
    if not done_files:
        return None
    latest_done = max(done_files)
    step = int(re.search(r'_(\d{8})', latest_done).group(1))
    return step

def parse_loss_from_log(log_folder, step):
    log_file = os.path.join(log_folder, "main.log")
    with open(log_file, "r") as f:
        for line in f:
            if f"step {step}" in line:
                match = re.search(r"loss: ([\d.]+)", line)
                if match:
                    return float(match.group(1))
    return None

def upload_checkpoint(log_folder, step, api):
    files_to_upload = [
        f for f in os.listdir(log_folder)
        if f.startswith(f"DONE_{step:08d}") or
           f.startswith(f"model_{step:08d}") or
           f.startswith(f"state_{step:08d}")
    ]
    
    for file in files_to_upload:
        file_path = os.path.join(log_folder, file)
        upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"checkpoints/{file}",
            repo_id=repo_name,
            token=hf_token
        )
    
    # Upload main.log and wandb_checkpoint.json
    upload_file(
        path_or_fileobj=os.path.join(log_folder, "main.log"),
        path_in_repo="main.log",
        repo_id=repo_name,
        token=hf_token
    )
    upload_file(
        path_or_fileobj=os.path.join(log_folder, "wandb_checkpoint.json"),
        path_in_repo="wandb_checkpoint.json",
        repo_id=repo_name,
        token=hf_token
    )

def delete_old_checkpoints(log_folder, latest_step):
    for file in os.listdir(log_folder):
        if file.startswith(("DONE_", "model_", "state_")):
            # Extract step number using regex
            match = re.search(r'_(\d{8})', file)
            if match:
                step = int(match.group(1))
                if step != latest_step:
                    os.remove(os.path.join(log_folder, file))
                    print(f"Deleted old checkpoint file: {file}")
            else:
                print(f"Skipping file with unexpected format: {file}")

def upload_checkpoints():
    api = HfApi()
    log_folder = "./log124M"  # Adjust this path if needed
    
    if not os.path.exists(log_folder):
        print(f"Log folder not found: {log_folder}")
        return
    
    latest_step = find_latest_checkpoint(log_folder)
    if latest_step is None:
        print("No checkpoints found in the log folder.")
        return
    
    latest_loss = parse_loss_from_log(log_folder, latest_step)
    
    print(f"Found {len([f for f in os.listdir(log_folder) if f.startswith('DONE_')])} checkpoints in the log folder.")
    print(f"Latest checkpoint: Step {latest_step}, Loss: {latest_loss}")
    
    choice = input("1. Upload all checkpoints\n2. Upload only the latest checkpoint\n3. Switch to download section\nEnter your choice (1/2/3): ")
    
    if choice == "1":
        continuous = input("Should I continuously upload all checkpoints? (y/n): ").lower() == 'y'
        if continuous:
            while True:
                checkpoints = sorted([int(re.search(r'_(\d{8})', f).group(1)) for f in os.listdir(log_folder) if f.startswith("DONE_") and re.search(r'_(\d{8})', f)])
                for step in checkpoints:
                    upload_checkpoint(log_folder, step, api)
                    if step != max(checkpoints):
                        delete_old_checkpoints(log_folder, max(checkpoints))
                print(f"Uploaded {len(checkpoints)} checkpoints. Waiting for new checkpoints...")
                time.sleep(300)  # Wait for 5 minutes
        else:
            checkpoints = sorted([int(re.search(r'_(\d{8})', f).group(1)) for f in os.listdir(log_folder) if f.startswith("DONE_") and re.search(r'_(\d{8})', f)])
            for step in checkpoints:
                upload_checkpoint(log_folder, step, api)
                if step != max(checkpoints):
                    delete_old_checkpoints(log_folder, max(checkpoints))
            print(f"Uploaded {len(checkpoints)} checkpoints.")
    
    elif choice == "2":
        continuous = input("Should I continuously upload only the latest checkpoint? (y/n): ").lower() == 'y'
        if continuous:
            last_uploaded_step = None
            while True:
                latest_step = find_latest_checkpoint(log_folder)
                if latest_step != last_uploaded_step:
                    if last_uploaded_step:
                        api.delete_folder(repo_id=repo_name, folder_path=f"checkpoints", token=hf_token)
                        delete_old_checkpoints(log_folder, latest_step)
                    
                    upload_checkpoint(log_folder, latest_step, api)
                    last_uploaded_step = latest_step
                    print(f"Uploaded latest checkpoint: Step {latest_step}")
                
                print("Waiting for new checkpoints...")
                time.sleep(300)  # Wait for 5 minutes
        else:
            upload_checkpoint(log_folder, latest_step, api)
            delete_old_checkpoints(log_folder, latest_step)
            print(f"Uploaded latest checkpoint: Step {latest_step}")
    
    elif choice == "3":
        download_checkpoints()
    
    else:
        print("Invalid choice. Please run the script again and select a valid option.")

def download_checkpoints():
    api = HfApi()
    
    # List available checkpoints
    files = api.list_repo_files(repo_id=repo_name, token=hf_token)
    checkpoints = [f for f in files if f.startswith("checkpoints/DONE_")]
    
    if not checkpoints:
        print("No checkpoints available for download.")
        return
    
    print("Available checkpoints:")
    for checkpoint in checkpoints:
        step = int(re.search(r'_(\d{8})', checkpoint).group(1))
        print(f"Step: {step}")
    
    step_to_download = input("Enter the step number of the checkpoint you want to download: ")
    
    try:
        step = int(step_to_download)
    except ValueError:
        print("Invalid step number. Please enter a valid integer.")
        return
    
    download_folder = "./log124M"
    if os.path.exists(download_folder):
        overwrite = input(f"The folder {download_folder} already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Download cancelled.")
            return
        shutil.rmtree(download_folder)
    os.makedirs(download_folder)
    
    # Download checkpoint files
    checkpoint_files = [f for f in files if f.startswith(f"checkpoints/") and f"{step:08d}" in f]
    for file in checkpoint_files:
        local_file = os.path.join(download_folder, os.path.basename(file))
        hf_hub_download(repo_id=repo_name, filename=file, local_dir=download_folder, token=hf_token)
        # Move file from nested 'checkpoints' folder to main download folder
        nested_file = os.path.join(download_folder, file)
        if os.path.exists(nested_file):
            shutil.move(nested_file, local_file)
        print(f"Downloaded: {os.path.basename(file)}")
    
    # Download main.log and wandb_checkpoint.json
    for file in ["main.log", "wandb_checkpoint.json"]:
        local_file = os.path.join(download_folder, file)
        hf_hub_download(repo_id=repo_name, filename=file, local_dir=download_folder, token=hf_token)
        print(f"Downloaded: {file}")
    
    # Remove the cache folder
    cache_folder = os.path.join(download_folder, '.cache')
    if os.path.exists(cache_folder):
        shutil.rmtree(cache_folder)
    
    # Remove the empty 'checkpoints' folder if it exists
    checkpoints_folder = os.path.join(download_folder, 'checkpoints')
    if os.path.exists(checkpoints_folder) and not os.listdir(checkpoints_folder):
        os.rmdir(checkpoints_folder)
    
    print(f"Downloaded checkpoint and log files to {download_folder}")

if __name__ == "__main__":
    while True:
        choice = input("1. Upload checkpoints\n2. Download checkpoints\n3. Exit\nEnter your choice (1/2/3): ")
        if choice == "1":
            upload_checkpoints()
        elif choice == "2":
            download_checkpoints()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please try again.")
