from huggingface_hub import snapshot_download

if __name__ == "__main__":
    repo_id = "Qwen/Qwen2.5-Math-1.5B"
    local_dir = snapshot_download(repo_id=repo_id, local_dir="models/Qwen2.5-Math-1.5B")
