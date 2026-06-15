import requests
import os
import zipfile
import shutil
import re
import subprocess
from pathlib import Path

def get_current_version():
    try:
        result = subprocess.run(["llama-server", "--version"], capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
        match = re.search(r"version: (\d+)", output)
        if match:
            return match.group(1)
    except Exception:
        pass
    return None

def main():
    repo = "ggml-org/llama.cpp"
    api_url = f"https://api.github.com/repos/{repo}/releases/latest"
    
    print("Checking current version...")
    current_version = get_current_version()
    if current_version:
        print(f"Current version: {current_version}")
    else:
        print("llama-server not found. Proceeding with fresh installation.")

    print("Fetching latest release information...")
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        release = response.json()
    except Exception as e:
        print(f"Failed to fetch release information: {e}")
        return

    # Find the latest CUDA 13 binary asset
    cuda_binary_asset = next((a for a in release["assets"] if a["name"].startswith("llama-b") and "-bin-win-cuda-13" in a["name"] and a["name"].endswith(".zip")), None)

    if not cuda_binary_asset:
        print("Could not find a CUDA 13 binary asset in the latest release.")
        return

    # Extract version from the binary asset name (e.g., llama-b9632-bin-win-cuda-13.3-x64.zip -> 9632)
    match = re.search(r"llama-b(\d+)-bin", cuda_binary_asset["name"])
    if not match:
        print(f"Could not determine latest version from asset name: {cuda_binary_asset['name']}")
        return
    
    latest_version = match.group(1)

    if current_version and current_version == latest_version:
        print(f"Already up to date (version {current_version}). No update needed.")
        return

    print(f"Update available: {current_version or 'None'} -> {latest_version}")

    # Extract CUDA version from the binary asset name (e.g., llama-b9632-bin-win-cuda-13.3-x64.zip -> 13.3)
    cuda_ver_match = re.search(r"cuda-([\d\.]+)-x64", cuda_binary_asset["name"])
    if not cuda_ver_match:
        print(f"Could not determine CUDA version from asset name: {cuda_binary_asset['name']}")
        return
    
    cuda_version = cuda_ver_match.group(1)
    print(f"Detected CUDA version: {cuda_version}")

    # Find the corresponding CUDA runtime (DLLs) asset
    runtime_asset_name = f"cudart-llama-bin-win-cuda-{cuda_version}-x64.zip"
    cuda_runtime_asset = next((a for a in release["assets"] if a["name"] == runtime_asset_name), None)

    if not cuda_runtime_asset:
        print(f"Could not find the corresponding CUDA runtime asset for version {cuda_version}.")
        return

    binary_url = cuda_binary_asset["browser_download_url"]
    runtime_url = cuda_runtime_asset["browser_download_url"]

    print(f"Latest CUDA Binary URL: {binary_url}")
    print(f"Latest CUDA Runtime URL: {runtime_url}")

    # Find where llama-server is located to determine the unpacking directory
    server_path = shutil.which("llama-server")
    if server_path:
        target_dir = Path(server_path).parent
        print(f"Target directory for unpacking: {target_dir}")
    else:
        target_dir = Path.cwd()
        print(f"llama-server not found, using current directory: {target_dir}")

    def download_and_unpack(url, filename):
        print(f"Downloading and unpacking {filename}...")
        temp_zip = Path(os.getenv("TEMP", ".")) / filename
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(temp_zip, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        
        with zipfile.ZipFile(temp_zip, "r") as zip_ref:
            zip_ref.extractall(target_dir)
        
        temp_zip.unlink()

    try:
        download_and_unpack(binary_url, "llama-bin.zip")
        download_and_unpack(runtime_url, "llama-rt.zip")
        print("Update complete!")
    except Exception as e:
        print(f"An error occurred during update: {e}")

if __name__ == "__main__":
    main()
