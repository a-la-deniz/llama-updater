import csv
import io
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import requests

REPO = "ggml-org/llama.cpp"
API_TIMEOUT = 30
# (connect, read) timeout for downloads; read timeout applies per chunk, not the whole file
DOWNLOAD_TIMEOUT = (10, 60)


def github_headers():
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_current_version():
    try:
        result = subprocess.run(
            ["llama-server", "--version"], capture_output=True, text=True, check=False
        )
        output = result.stdout + result.stderr
        # Newer builds: "version: 0.7.0-dev (build 11756, commit ...)"
        # Older builds: "version: 4567 (abc1234)"
        match = re.search(r"\(build (\d+)", output) or re.search(
            r"version: (\d+) \(", output
        )
        if match:
            return int(match.group(1))
    except Exception:
        pass
    return None


def find_cuda_assets(release):
    """Return (binary_asset, runtime_asset, build, cuda_version) if the release has a
    complete Windows CUDA 13 x64 build, otherwise None."""
    for asset in release["assets"]:
        # e.g. llama-b9632-bin-win-cuda-13.3-x64.zip
        match = re.fullmatch(r"llama-b(\d+)-bin-win-cuda-(13[\d.]*)-x64\.zip", asset["name"])
        if not match:
            continue
        build, cuda_version = int(match.group(1)), match.group(2)
        runtime_name = f"cudart-llama-bin-win-cuda-{cuda_version}-x64.zip"
        runtime = next((a for a in release["assets"] if a["name"] == runtime_name), None)
        if runtime:
            return asset, runtime, build, cuda_version
    return None


def running_llama_processes(target_dir):
    """Return names of running processes whose executable also exists in target_dir."""
    try:
        result = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=True
        )
    except Exception:
        return []
    names = {row[0] for row in csv.reader(io.StringIO(result.stdout)) if row}
    return sorted(n for n in names if (target_dir / n).exists() and n.lower().startswith("llama"))


def download(url, dest):
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)


def main():
    api_url = f"https://api.github.com/repos/{REPO}/releases"

    print("Checking current version...")
    current_version = get_current_version()
    if current_version:
        print(f"Current version: {current_version}")
    else:
        print("llama-server not found. Proceeding with fresh installation.")

    print("Fetching latest release information...")
    try:
        response = requests.get(api_url, headers=github_headers(), timeout=API_TIMEOUT)
        response.raise_for_status()
        releases = response.json()
    except Exception as e:
        print(f"Failed to fetch release information: {e}")
        return

    # CI uploads platform builds at different times, so the newest release may not
    # have the Windows CUDA assets yet. Take the first one that does.
    found = None
    for release in releases:
        found = find_cuda_assets(release)
        if found:
            break

    if not found:
        print("Could not find a recent release with CUDA 13 Windows binaries.")
        return

    binary_asset, runtime_asset, latest_version, cuda_version = found

    if current_version and current_version >= latest_version:
        if current_version == latest_version:
            print(f"Already up to date (version {current_version}). No update needed.")
        else:
            print(
                f"Local build {current_version} is newer than latest release "
                f"{latest_version}. Skipping."
            )
        return

    print(f"Update available: {current_version or 'None'} -> {latest_version}")
    print(f"Detected CUDA version: {cuda_version}")

    binary_url = binary_asset["browser_download_url"]
    runtime_url = runtime_asset["browser_download_url"]
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

    # Windows can't overwrite executables or DLLs that are in use
    running = running_llama_processes(target_dir)
    if running:
        print(f"Close these running processes before updating: {', '.join(running)}")
        return

    try:
        with tempfile.TemporaryDirectory(prefix="llama-update-") as tmp:
            tmp = Path(tmp)
            staging = tmp / "staging"
            for url, filename in [
                (binary_url, binary_asset["name"]),
                (runtime_url, runtime_asset["name"]),
            ]:
                print(f"Downloading {filename}...")
                zip_path = tmp / filename
                download(url, zip_path)
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(staging)

            # Only touch the install once everything downloaded and unpacked cleanly
            print(f"Installing to {target_dir}...")
            shutil.copytree(staging, target_dir, dirs_exist_ok=True)
        print("Update complete!")
    except Exception as e:
        print(f"An error occurred during update: {e}")


if __name__ == "__main__":
    main()
