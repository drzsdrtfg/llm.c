# Should be placed in cd ~
import os
import shutil
from huggingface_hub import snapshot_download
from datasets import load_dataset

def download_dataset(repo_id, base_dir):
    # Create the fineweb10B folder within the base directory
    dataset_dir = os.path.join(base_dir, "fineweb10B")
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"Downloading dataset '{repo_id}' to '{dataset_dir}'...")
    
    try:
        # Download the dataset files
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dataset_dir,
            ignore_patterns=["*.gitattributes", "README.md"]
        )
        print(f"Dataset files downloaded successfully to {dataset_dir}")
        
        # List the contents of the downloaded folder
        print("\nContents of the downloaded folder:")
        for root, dirs, files in os.walk(dataset_dir):
            level = root.replace(dataset_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
            sub_indent = ' ' * 4 * (level + 1)
            for file in files:
                print(f"{sub_indent}{file}")
        
        # Attempt to load the dataset to check its structure
        try:
            dataset = load_dataset(dataset_dir, use_auth_token=False)
            print("\nDataset structure:")
            print(dataset)
            
            # Print information about the first few examples
            print("\nFirst few examples:")
            for split in dataset.keys():
                print(f"\nSplit: {split}")
                for i, example in enumerate(dataset[split].select(range(min(3, len(dataset[split])))), 1):
                    print(f"Example {i}:")
                    for key, value in example.items():
                        if isinstance(value, (list, str)):
                            print(f"  {key}: {value[:100]}{'...' if len(str(value)) > 100 else ''}")
                        else:
                            print(f"  {key}: {value}")
        except Exception as e:
            print(f"\nNote: Unable to load the dataset using datasets library. This is expected if the files are in a custom format.")
            print(f"Error details: {e}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")

def delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted successfully.")
    else:
        print(f"Folder '{folder_path}' does not exist.")

if __name__ == "__main__":
    # Set the Hugging Face dataset repository ID
    repo_id = "anonymguy/ehm"
    
    # Set the base directory path where you want to create the fineweb10B folder
    base_dir = "/root/llm.c/dev/data"

     # Path to the folder you want to delete
    folder_to_delete = "/root/llm.c/dev/data/fineweb10B/.cache"

    # Download the dataset
    download_dataset(repo_id, base_dir)

    # Call the function to delete the folder
    delete_folder(folder_to_delete)
