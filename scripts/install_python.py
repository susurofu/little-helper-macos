"""Create a local venv, install dependencies, and prepare configured models."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import venv

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from scripts.start_local import ollama_environment, ollama_ready, stop_process


def run(command, env=None):
    print("Running: " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_DIR, env=env)


def install(with_tests=False, skip_models=False):
    if not (3, 10) <= sys.version_info[:2] < (3, 15):
        raise RuntimeError("Use Python 3.10–3.14 (Python 3.12 recommended).")
    environment = PROJECT_DIR / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        if environment.exists():
            raise RuntimeError(".venv exists but has no Python executable. Rename it and rerun setup.")
        print("Creating project .venv…", flush=True)
        venv.EnvBuilder(with_pip=True).create(environment)
    run([str(python), "-m", "pip", "install", "--upgrade", "pip"])
    requirements = "requirements-test.txt" if with_tests else "requirements.txt"
    run([str(python), "-m", "pip", "install", "-r", str(PROJECT_DIR / requirements)])
    run([str(python), "-m", "pip", "check"])
    if skip_models:
        return
    executable = shutil.which("ollama")
    if not executable:
        raise RuntimeError("Ollama is not on PATH. Reopen the terminal and rerun setup.")
    config = json.loads((PROJECT_DIR / "global_config.json").read_text(encoding="utf-8"))
    models = list(dict.fromkeys([config["local_llm_model"], "qllama/multilingual-e5-small:latest"]))
    env = ollama_environment()
    print("Preparing models in the shared local Ollama service (outside .venv).", flush=True)
    owned = None
    try:
        if not ollama_ready():
            owned = subprocess.Popen([executable, "serve"], env=env)
            deadline = time.monotonic() + 30
            while not ollama_ready():
                if owned.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Ollama did not start. Check its output above.")
                time.sleep(0.5)
        for model in models:
            # Pull is safe to repeat: Ollama reuses already downloaded layers.
            run([executable, "pull", model], env=env)
    finally:
        stop_process(owned)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-tests", action="store_true", help="Also install Python test dependencies")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    args = parser.parse_args()
    try:
        install(args.with_tests, args.skip_models)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nSetup interrupted. Rerun to resume.", file=sys.stderr)
        return 130
    print("Setup complete. Run start.bat on Windows or ./start.sh on macOS/Linux.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
