# macOS backend setup

Keep this entire backend folder wherever you want to run it. It is separate from
`manifest.json` and `extention/`, which make up the browser extension.

Double-click `install.command`, or run `./install.sh` from Terminal. Run as your
normal user; Homebrew may request an administrator password during its setup.
The installer installs Homebrew if missing, installs Python 3.12 if no compatible
Python (3.10–3.14 with venv/pip support) is available, and installs the global
Ollama executable through Homebrew if needed. An existing official Ollama app is
also recognized. Homebrew may require Xcode Command Line Tools and a supported
macOS version; follow any prompts from its installer.

Python dependencies, including the Ollama Python client, go into `.venv` inside
this backend folder. The Ollama server itself is installed outside `.venv`.
Models are pulled through the local service at `127.0.0.1:11434`. Ollama decides where to store models using its defaults or the user's existing
configuration. These scripts do not set or override `OLLAMA_MODELS`. Downloads
reuse existing model layers.

Run `./start.sh` (or double-click `start.command`) after setup. There is no need
to activate `.venv` manually. If you move an already installed backend folder,
recreate its `.venv` by removing that environment and rerunning setup, since
Python virtual environments are not portable.

Optional installer flags: `--skip-models` skips downloads; `--with-tests` also
installs test dependencies. Set `LOCAL_HELPER_PYTHON` to select a compatible
Python executable explicitly.
