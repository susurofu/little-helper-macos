"""Start the project's local services using the selected Python interpreter."""
import importlib.util
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROJECT_DIR = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://127.0.0.1:11434/api/tags"


def ollama_environment():
    """Use the local server while preserving Ollama's storage configuration."""
    return dict(os.environ, OLLAMA_HOST="127.0.0.1:11434")


def ollama_ready():
    try:
        with urllib.request.urlopen(OLLAMA_URL, timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def stop_process(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def main():
    os.chdir(PROJECT_DIR)
    packages = {"uvicorn": "uvicorn", "fastapi": "fastapi", "ollama": "ollama",
                "transformers": "transformers", "chromadb": "chromadb", "docx": "python-docx"}
    missing = [package for module, package in packages.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        print(f"Missing Python packages in {sys.executable}: {', '.join(missing)}", flush=True)
        print("Install them with your selected interpreter: python -m pip install " + " ".join(missing))
        return 1
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", 8000)) == 0:
            print("Port 8000 is already in use. Stop the existing server before launching.")
            return 1

    ollama_process = None
    server_process = None
    try:
        if ollama_ready():
            print("Using the running Ollama service.", flush=True)
        else:
            executable = shutil.which("ollama")
            if not executable:
                print("Ollama is not on PATH. Install Ollama, then reopen your terminal.")
                return 1
            print("Starting Ollama at http://127.0.0.1:11434 …", flush=True)
            env = ollama_environment()
            ollama_process = subprocess.Popen([executable, "serve"], env=env)
            deadline = time.monotonic() + 30
            while not ollama_ready():
                if ollama_process.poll() is not None:
                    print("Ollama exited before it was ready. Check its output above.")
                    return 1
                if time.monotonic() >= deadline:
                    print("Ollama did not become ready within 30 seconds.")
                    return 1
                time.sleep(0.5)
        print("Starting the local API. Press Ctrl+C to stop.\nhttp://127.0.0.1:8000/docs", flush=True)
        server_process = subprocess.Popen(
            [sys.executable, "-m", "backend.local_server"], env=ollama_environment())
        return server_process.wait()
    except KeyboardInterrupt:
        print("\nStopping services started by this launcher…", flush=True)
        return 0
    except OSError as exc:
        print(f"Could not start local services: {exc}")
        return 1
    finally:
        stop_process(server_process)
        # Leave a pre-existing Ollama service running.
        stop_process(ollama_process)


if __name__ == "__main__":
    sys.exit(main())
