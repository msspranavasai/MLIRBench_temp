
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

class ValidatorInfrastructureError(RuntimeError):
    """The validation environment is unavailable or misconfigured."""

MLIR_OPT = os.getenv("MLIR_OPT", "mlir-opt")
TORCH_MLIR_OPT = os.getenv("TORCH_MLIR_OPT", "torch-mlir-opt")

# Supported values: local, remote
TORCH_MLIR_MODE = os.getenv("TORCH_MLIR_MODE", "local").lower()

# Required only when TORCH_MLIR_MODE=remote
TORCH_MLIR_API = os.getenv("TORCH_MLIR_API", "").rstrip("/")

VALIDATOR_TIMEOUT = int(os.getenv("MLIR_VALIDATOR_TIMEOUT", "60"))


def _run_validator(command, mlir_text):
    """Validate MLIR using a local compiler executable."""

    executable = command[0]

    if not shutil.which(executable) and not Path(executable).is_file():
        raise ValidatorInfrastructureError(
            f"Required validator '{executable}' was not found. "
            "Install it or configure its path during setup."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".mlir",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(mlir_text)
            temp_path = f.name

        result = subprocess.run(
            command + [temp_path],
            capture_output=True,
            text=True,
            timeout=VALIDATOR_TIMEOUT,
            check=False,
        )

        location = r'(?m)^' + re.escape(temp_path) + r':\d+:\d+: '
        diagnostic_kind = ('verification_error' if re.search(location + r'note: see current operation:', result.stderr)
                           else 'ssa_error' if re.search(location + r'error: use of undeclared SSA value name', result.stderr)
                           else 'other_error' if result.returncode else 'none')
        return {
            "diagnostic_kind": diagnostic_kind,
            "valid": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except subprocess.TimeoutExpired as exc:
        raise ValidatorInfrastructureError(
            f"Validator '{executable}' timed out after "
            f"{VALIDATOR_TIMEOUT} seconds."
        ) from exc

    except OSError as exc:
        raise ValidatorInfrastructureError(
            f"Failed to execute validator '{executable}': {exc}"
        ) from exc

    finally:
        if temp_path is not None:
            Path(temp_path).unlink(missing_ok=True)


def validate_mutation(input_mlir: str, mutated_mlir: str):
    """Validate a mutation in the MLIR linalg dialect."""

    return _run_validator(
        [MLIR_OPT],
        mutated_mlir,
    )


def validate_with_local_torch_mlir(mutated_mlir: str):
    """Validate Torch dialect MLIR using a local executable."""

    return _run_validator(
        [TORCH_MLIR_OPT],
        mutated_mlir,
    )


def validate_with_remote_api(mutated_mlir: str):
    """Validate Torch dialect MLIR through an external API."""

    if not TORCH_MLIR_API:
        raise ValidatorInfrastructureError(
            "TORCH_MLIR_API is not configured. "
            "Set the validator endpoint in your .env file."
        )

    try:
        response = requests.post(
            f"{TORCH_MLIR_API}/validate",
            json={"mlir": mutated_mlir},
            timeout=VALIDATOR_TIMEOUT,
        )

        response.raise_for_status()
        result = response.json()

    except (requests.RequestException, ValueError) as exc:
        raise ValidatorInfrastructureError(
            f"Torch-MLIR validation service failed: {exc}"
        ) from exc

    if not isinstance(result, dict):
        raise ValidatorInfrastructureError(
            "Torch-MLIR validation service returned an invalid response."
        )

    required_fields = {
        "valid",
        "stdout",
        "stderr",
        "returncode",
    }

    if not required_fields.issubset(result):
        raise ValidatorInfrastructureError(
            "Torch-MLIR validation service returned an incomplete response."
        )

    if (
        not isinstance(result["valid"], bool)
        or not isinstance(result["stdout"], str)
        or not isinstance(result["stderr"], str)
        or type(result["returncode"]) is not int
    ):
        raise ValidatorInfrastructureError(
            "Torch-MLIR validation service returned invalid field types."
        )

    return {
        "valid": result["valid"],
        "stdout": result["stdout"],
        "stderr": result["stderr"],
        "returncode": result["returncode"],
    }


def validate_torch_mutation(input_mlir: str, mutated_mlir: str):
    """Validate Torch dialect MLIR using the configured backend."""

    if TORCH_MLIR_MODE == "local":
        return validate_with_local_torch_mlir(mutated_mlir)

    if TORCH_MLIR_MODE == "remote":
        return validate_with_remote_api(mutated_mlir)

    raise ValidatorInfrastructureError(
        f"Unsupported TORCH_MLIR_MODE: {TORCH_MLIR_MODE}. "
        "Expected 'local' or 'remote'."
    )
