# Uploads the pretokenized dataset to huggingface
#Should be placed in cd ~
from huggingface_hub import HfApi, create_repo

def upload_folder_to_huggingface(local_folder_path, repo_name, token):
    # Initialize Hugging Face API
    api = HfApi()
    
    try:
        # Create a new repository on Hugging Face
        create_repo(repo_name, token=token, repo_type="dataset")
        print(f"Repository '{repo_name}' created successfully.")
    except Exception as e:
        print(f"Error creating repository: {e}")
        print("Continuing with upload as the repository might already exist.")
    
    try:
        # Upload the entire folder to Hugging Face
        api.upload_folder(
            folder_path=local_folder_path,
            repo_id=repo_name,
            repo_type="dataset",
            token=token
        )
        print(f"Folder '{local_folder_path}' uploaded successfully to {repo_name}")
    except Exception as e:
        print(f"Error uploading folder: {e}")

if __name__ == "__main__":
    # Set your Hugging Face token
    hf_token = ""
    
    # Set the local folder path
    local_folder_path = "dev/data/fineweb10B"
    
    # Set the desired repository name on Hugging Face
    repo_name = "anonymguy/ehm"
    
    # Upload the folder
    upload_folder_to_huggingface(local_folder_path, repo_name, hf_token)
