# Create the data_registration file

# Import libraries to upload the data using huggingface_hub
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# define the huggingface space and type of data being uploaded
repo_id = "papsofts/tourism-project"
repo_type = "dataset"

# Initialize API client using Huggingface token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
  # Create the new space
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# upload the data folder contents to the Repository
api.upload_folder(
    folder_path="mlops_tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
