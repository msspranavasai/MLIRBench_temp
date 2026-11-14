# Validation API

MLIRBench uses a lightweight FastAPI service for validating Torch dialect MLIR programs. The service wraps `torch-mlir-opt` and is used by the benchmark pipeline for Torch mutation validation.

---

## Configuration

The base URL is configured through the following environment variable:

```bash
TORCH_MLIR_API=http://<validator-host>:8000
```

---

## Endpoints

### 1. Health Check

Verify service availability.

- **Method & Path:** `GET /health`
- **Example Request:**

```bash
curl http://<validator-host>:8000/health
```

---

### 2. Validate MLIR

Validate a Torch dialect MLIR program.

- **Method & Path:** `POST /validate`
- **Request Header:** `Content-Type: application/json`
- **Request Body:**

```json
{
  "mlir": "module { ... }"
}
```

- **Response Body:**

```json
{
  "valid": true,
  "returncode": 0,
  "stdout": "...",
  "stderr": ""
}
```
