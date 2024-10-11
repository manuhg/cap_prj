from huggingface_hub import hf_hub_download
def download_model(model_repo):
    hf_hub_download(repo_id=model_repo, filename="config.json")

download_model("wave-on-discord/gemini-nano")
