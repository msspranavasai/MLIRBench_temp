"""Interactive first-run configuration for MLIRBench."""

import getpass
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".env"

PROVIDERS = {
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Google Gemini": "GOOGLE_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
}


def prompt(label, default=""):
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value or default


def setup():
    print("\n=== MLIRBench First-Run Setup ===\n")

    if ENV_FILE.exists():
        print("Existing .env detected.")
        if input("Reconfigure? [y/N]: ").strip().lower() != "y":
            print("Existing configuration preserved.")
            return

    config = {}

    print("\n[1/3] Compiler configuration")

    config["MLIR_OPT"] = prompt("Path to mlir-opt", "mlir-opt")

    mode = prompt(
        "Torch-MLIR mode (local/remote)",
        "local",
    ).lower()

    if mode not in {"local", "remote"}:
        raise ValueError("Torch-MLIR mode must be local or remote.")

    config["TORCH_MLIR_MODE"] = mode

    if mode == "local":
        config["TORCH_MLIR_OPT"] = prompt(
            "Path to torch-mlir-opt",
            "torch-mlir-opt",
        )
    else:
        endpoint = prompt("Torch-MLIR validation API URL")
        if not endpoint.startswith(("http://", "https://")):
            raise ValueError("A valid HTTP(S) endpoint is required.")
        config["TORCH_MLIR_API"] = endpoint.rstrip("/")

    print("\n[2/3] Model provider credentials")
    print("Press Enter to skip providers you do not use.")

    for name, variable in PROVIDERS.items():
        key = getpass.getpass(f"{name} API key (optional): ").strip()
        if key:
            config[variable] = key

    print("\n[3/3] Save configuration")

    # Preserve unrelated existing variables during reconfiguration.
    if ENV_FILE.exists():
        existing = {}
        for line in ENV_FILE.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
        existing.update(config)
        config = existing

    # Write a new file with restrictive permissions.
    fd = os.open(
        ENV_FILE,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )

    with os.fdopen(fd, "w") as f:
        for key, value in config.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"Invalid multiline value for {key}")
            f.write(f'{key}="{value}"\n')

    os.chmod(ENV_FILE, 0o600)

    print("\nConfiguration saved to .env")
    print("API keys will not be displayed.")

    print("\nCompiler discovery:")
    for name in ("MLIR_OPT", "TORCH_MLIR_OPT"):
        if name in config:
            path = config[name]
            found = shutil.which(path) or (
                str(Path(path).resolve()) if Path(path).is_file() else None
            )
            print(f"  {name}: {'FOUND' if found else 'NOT FOUND'}")

    print("\nNext: python run.py check")


if __name__ == "__main__":
    setup()
