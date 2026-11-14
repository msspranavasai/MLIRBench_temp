# Google Cloud Setup

This document describes how to recreate the Google Cloud infrastructure used by MLIRBench for Torch-MLIR validation.

The validator service is hosted on a dedicated Google Cloud VM and exposes a lightweight API that wraps `torch-mlir-opt`.

---

# Reference Configuration

The MLIRBench validator was deployed using the following configuration.

| Setting          | Value                 |
| ---------------- | --------------------- |
| Cloud Provider   | Google Cloud Platform |
| Service          | Compute Engine        |
| Machine Type     | e2-medium             |
| Region           | us-west1              |
| Zone             | us-west1-b            |
| Operating System | Ubuntu 22.04 LTS      |
| Internal IP      | 10.138.0.2            |
| External IP      | Static IP             |
| Public Endpoint  | Port 8000             |

---

# Step 1: Create a Google Cloud Project

1. Open Google Cloud Console.
2. Create a new project.
3. Enable billing for the project.

---

# Step 2: Enable Compute Engine

Navigate to:

```text
Google Cloud Console
→ Compute Engine
```

Click:

```text
Enable Compute Engine API
```

Wait for the service to finish provisioning.

---

# Step 3: Create the VM

Navigate to:

```text
Compute Engine
→ VM Instances
→ Create Instance
```

Configure:

| Setting      | Value            |
| ------------ | ---------------- |
| Machine Type | e2-medium        |
| Region       | us-west1         |
| Zone         | us-west1-b       |
| Boot Disk    | Ubuntu 22.04 LTS |
| Disk Size    | 30–50 GB         |

Create the instance.

---

# Step 4: Reserve a Static IP Address

Navigate to:

```text
VPC Network
→ IP Addresses
```

Locate the VM's external IP address.

Click:

```text
Reserve Static Address
```

Assign the static address to the VM.

This ensures the validator endpoint remains unchanged across VM restarts.

---

# Step 5: Connect to the VM

Use either:

```bash
gcloud compute ssh <instance-name> --zone us-west1-b
```

or the browser SSH option in the Google Cloud Console.

---

# Step 6: Install Required Packages

Update the system:

```bash
sudo apt update
sudo apt upgrade -y
```

Install required utilities:

```bash
sudo apt install -y \
python3 \
python3-pip \
python3-venv \
git \
curl \
wget
```

---

# Step 7: Copy Validator Files

Copy the validator deployment files to the VM.

Example:

```bash
scp -r validator/ USER@VM_IP:~
```

Replace this step with the actual files used in your deployment.

Typical files include:

```text
app.py
requirements.txt
torchmlir-api.service
```

---

# Step 8: Create Python Environment

```bash
python3 -m venv mlir_venv
source mlir_venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Step 9: Install Torch-MLIR

Install Torch-MLIR following the version used by MLIRBench.

Verify:

```bash
torch-mlir-opt --version
```

Expected:

```text
LLVM version ...
```

---

# Step 10: Verify Validator Application

Start locally:

```bash
uvicorn app:app \
  --host 0.0.0.0 \
  --port 8000
```

Verify:

```bash
curl http://127.0.0.1:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "tool": "torch-mlir-opt",
  "available": true
}
```

---

# Step 11: Configure Firewall

Navigate to:

```text
VPC Network
→ Firewall
→ Create Firewall Rule
```

Configuration:

| Setting     | Value                 |
| ----------- | --------------------- |
| Direction   | Ingress               |
| Targets     | Specified Target Tags |
| Network Tag | torchmlir-api         |
| Protocol    | TCP                   |
| Port        | 8000                  |

Save the rule.

---

# Step 12: Add VM Network Tag

Navigate to:

```text
Compute Engine
→ VM Instances
→ Edit
```

Add:

```text
torchmlir-api
```

under Network Tags.

Save.

---

# Step 13: Create a Systemd Service

Create:

```bash
sudo nano /etc/systemd/system/torchmlir-api.service
```

Copy the service definition provided with MLIRBench.

Reload services:

```bash
sudo systemctl daemon-reload
```

Enable startup:

```bash
sudo systemctl enable torchmlir-api
```

Start:

```bash
sudo systemctl start torchmlir-api
```

Verify:

```bash
sudo systemctl status torchmlir-api
```

---

# Step 14: Verify Public Access

From any machine:

```bash
curl http://<PUBLIC_IP>:8000/health
```

Expected:

```json
{
  "status": "ok",
  "tool": "torch-mlir-opt",
  "available": true
}
```

---

# Step 15: Configure MLIRBench

Add to `.env`:

```text
TORCH_MLIR_API=http://<PUBLIC_IP>:8000
```

Example:

```text
TORCH_MLIR_API=http://136.118.170.255:8000
```

The benchmark pipeline will automatically route Torch validation requests to this endpoint.

---

# VM Lifecycle

Stopping a VM preserves:

- Installed software
- Files
- Python environments
- Torch-MLIR installation
- Systemd configuration

Stop:

```bash
gcloud compute instances stop <instance-name> --zone us-west1-b
```

Start:

```bash
gcloud compute instances start <instance-name> --zone us-west1-b
```

When using a static IP address, the validator endpoint remains unchanged after restart.

---

# Verification Checklist

Before running benchmark experiments:

- Compute Engine enabled
- VM created
- Static IP assigned
- Torch-MLIR installed
- Validator service running
- Firewall rule created
- Network tag assigned
- Health endpoint reachable
- `TORCH_MLIR_API` configured

All checks should pass before using the validator service with MLIRBench.
