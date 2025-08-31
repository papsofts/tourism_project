from huggingface_hub import HfApi
import os

# Login to huggingface space
api = HfApi(token=os.getenv("HF_TOKEN"))

# upload the deployment directory contents to huggingface space
api.upload_folder(
    folder_path="mlops_tourism_project/deployment",     # the local folder containing your files
    repo_id="papsofts/tourism-project",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
