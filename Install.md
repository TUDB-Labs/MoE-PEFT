# Install MoE PEFT Factory

## Table of Contents

- [Docker](./Install.md#docker)
- [Linux](./Install.md#linux-ubuntu-debian-fedora-etc)
- [Windows](./Install.md#microsoft-windows)
- [macOS](./Install.md#apple-macos)

## Docker

### Requirements

- One or more NVIDIA GPUs
  - At least 16GB VRAM per card
  - Cards with Ampere or newer architecture will perform faster
- Installation of the [Docker Engine](https://docs.docker.com/get-docker/)
- Installation of the [latest graphics driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) and [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)

### Steps

```bash
docker run --gpus all -it --rm mikecovlee/moe_peft
```

You can check all available tags from: [mikecovlee/moe_peft/tags](https://hub.docker.com/r/mikecovlee/moe_peft/tags). Currently, we only provide a Linux image for the x86_64 (amd64) architecture.

## Verification

From the command line, type:

```bash
python
```

then enter the following code:

```python
import moe_peft
moe_peft.setup_logging("INFO")
moe_peft.backend.check_available()
```

Expected output:

```
MoE PEFT Factory: NVIDIA CUDA initialized successfully.
```

## Linux (Ubuntu, Debian, Fedora, etc.)

### Requirements

- One or more NVIDIA GPUs
  - At least 16GB VRAM per card
  - Cards with Ampere or newer architecture will perform faster
- Installation of the [latest graphics driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) and [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
- `conda` installed and configured
- Internet connection for automated tasks

### Steps

```bash
# Clone Repository
git clone https://github.com/TUDB-Labs/MoE-PEFT
cd moe_peft
# Optional but recommended
conda create -n moe_peft python=3.11
conda activate moe_peft
# Install requirements
pip3 install -r requirements.txt --upgrade
# Install extra requirements
pip3 install ninja
pip3 install bitsandbytes==0.43.1
pip3 install flash-attn==2.5.8 --no-build-isolation
```

## Verification

From the command line, type:

```bash
python
```

then enter the following code:

```python
import moe_peft
moe_peft.setup_logging("INFO")
moe_peft.backend.check_available()
```

Expected output:

```
MoE PEFT Factory: NVIDIA CUDA initialized successfully.
```

## Microsoft Windows

### Requirements

- One or more NVIDIA GPUs
  - At least 16GB VRAM per card
  - Cards with Ampere or newer architecture will perform faster
- Windows 10 or later
- Installation of the [latest graphics driver](https://www.nvidia.com/Download/index.aspx?lang=en-us) and [CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
- `conda` installed and configured
- Internet connection for automated tasks

### Steps

```bash
# Clone Repository
git clone https://github.com/TUDB-Labs/MoE-PEFT
cd moe_peft
# Optional but recommended
conda create -n moe_peft python=3.11
conda activate moe_peft
# Install requirements (CUDA 12.1)
pip3 install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip3 install -r requirements.txt --upgrade
# Install extra requirements
pip3 install bitsandbytes==0.43.1
```

## Verification

From the command line, type:

```bash
python
```

then enter the following code:

```python
import moe_peft
moe_peft.setup_logging("INFO")
moe_peft.backend.check_available()
```

Expected output:

```
MoE PEFT Factory: NVIDIA CUDA initialized successfully.
```

## Apple macOS

### Requirements

- Macintosh with Apple Silicon (recommended) or AMD GPUs
- macOS 12.3 or later
- Xcode command-line tools: `xcode-select --install`
- `conda` installed and configured
- Internet connection for automated tasks

## Steps

```bash
# Clone Repository
git clone https://github.com/TUDB-Labs/MoE-PEFT
cd moe_peft
# Optional but recommended
conda create -n moe_peft python=3.11
conda activate moe_peft
# Install requirements
pip3 install -r requirements.txt --upgrade
```

## Verification

From the command line, type:

```bash
python
```

then enter the following code:

```python
import moe_peft
moe_peft.setup_logging("INFO")
moe_peft.backend.check_available()
```

Expected output:

```
MoE PEFT Factory: APPLE MPS initialized successfully.
```