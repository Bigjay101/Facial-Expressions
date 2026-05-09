# Facial Expression Recognition

A CNN-based image classification project for recognising 9 facial expressions: *angry, contempt, disgust, fear, happy, natural, sad, sleepy, surprised*.

---

## Project Structure

```
project/
├── data/                   # gitignored — pulled via kagglehub at runtime
├── notebooks/
│   ├── 01_eda.ipynb        # exploratory data analysis & sanity checks
│   └── 02_preprocessing.ipynb  # pipeline verification
├── src/
│   └── dataset.py          # dataset class, transforms, sampler, dataloaders
├── outputs/                # gitignored — checkpoints & logs
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Prerequisites

- Windows 11
- NVIDIA GPU with up-to-date drivers ([download here](https://www.nvidia.com/drivers))
- WSL2 with Ubuntu installed

### Install WSL2 (if not already set up)

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

Restart your machine when prompted. Ubuntu will open after restart and ask you to create a username and password.

---

## Setup

### 1. Install Miniconda (inside Ubuntu)

Open Ubuntu and run:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 2. Create the Environment

```bash
conda create -n imgclass python=3.10 -y
conda activate imgclass
```

### 3. Install Dependencies

Navigate to the project folder and install from `requirements.txt`:

```bash
cd ~/projects/<repo-name>
pip install -r requirements.txt
```

### 4. Verify GPU Access

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

This should print `True` and your GPU name. If it prints `False`, make sure your NVIDIA drivers are up to date on the Windows side.

---

## Dataset

The dataset is sourced automatically from Kaggle at runtime via `kagglehub` — no manual download needed. On first run you will be prompted to authenticate:

```python
import kagglehub
kagglehub.login()
```

You will need your Kaggle API key, found at kaggle.com → Account → API → Create New Token.

---

## Running the Pipeline

### Verify preprocessing

```bash
conda activate imgclass
jupyter notebook
```

Open `notebooks/02_preprocessing.ipynb` and run all cells. This verifies that the dataset loads correctly, transforms are applied, and the sampler is balancing classes as expected.

### Test the dataset module directly

```bash
python src/dataset.py
```

This will print class weights and a sample batch shape. Expected output:

```
Batch shape  : torch.Size([32, 3, 224, 224])
Dtype        : torch.float32
Pixel range  : -2.xx to 2.xx
```

---

## Daily Usage

Every time you open a fresh Ubuntu terminal, activate the environment before running any code:

```bash
conda activate imgclass
```

---

## Troubleshooting

**CondaToSNonInteractiveError (Terms of Service):**
```bash
conda config --set auto_activate_base false
conda tos accept --override-channels --channel defaults
```

**Slow file I/O:** Make sure your project files are inside the WSL filesystem (`~/projects/...`) and not on the Windows mount (`/mnt/c/...`).

**WSL running out of memory:** Create `C:\Users\<YourName>\.wslconfig` on Windows with:
```ini
[wsl2]
memory=16GB
processors=8
gpuSupport=true
```
Then restart WSL with `wsl --shutdown` in PowerShell.