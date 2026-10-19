# 🌸 Multi-Class Flower Detection with YOLOv26 Nano

Hello! This repository contains the dataset preparation pipeline, training scripts, and trained model weights from my summer research internship at **Universitat Politècnica de Catalunya (UPC)** under the supervision of **Prof. Antoni Grau**.

The goal of this project is to detect and classify **5 common flower species** (`Daisy`, `Sunflower`, `Rose`, `Tulip`, and `Dandelion`) in real time using the lightweight **YOLOv26 Nano (`yolo26n`)** architecture. 

I wrote this guide step-by-step so that anyone—even if you are completely new to YOLO or cloud GPUs—can easily understand what we did, reproduce the dataset, and train or test the model from scratch.

---

## 🎯 What Does This Model Do?
Given an image or video, the model draws bounding boxes around flowers and predicts their species across 5 classes:
* `0: daisy` (includes common daisies and chamomile)
* `1: sunflower`
* `2: rose` (unified from 86 different rose breeds)
* `3: tulip`
* `4: dandelion`

Why **YOLOv26 Nano**? Because it is an NMS-free (end-to-end) lightweight model with only **~2.39M parameters (5.4 GFLOPs)**. It is small enough (only `5.5 MB`) to run on edge devices, drones, or mobile robots while still achieving **94.0% mAP50** accuracy.

---

## 🛠️ Challenges We Faced & How We Solved Them

During the internship, we didn't just run `model.train()` on a ready-made dataset. We had to solve several real-world engineering problems first:

### 1. Messy Multi-Source Datasets & Duplicate Files
We gathered 22 dataset archives from Roboflow/Google Drive. When we inspected the files using MD5 checksums, we discovered:
* **10 archives were exact duplicates** (`Copy of...`), which would cause severe data leakage if mixed into train and validation splits.
* **Different class names & IDs:** One dataset labeled daisies as `chamomile`, another used Turkish labels (`karahindiba` for dandelion), and another dataset had **86 separate classes for different rose breeds**.
* **Mixed formats:** Over 13,000 annotations were segmentation polygons instead of bounding boxes.
* **Extreme class imbalance:** Some field images had tiny background weeds labeled, and there were 70x more sunflower boxes than dandelion/tulip boxes.

**Our Solution (`prepare_dataset.py`):** 
We wrote an automated Python script that removes duplicate archives, converts all polygons into standard YOLO bounding boxes, maps all synonyms and the 86 rose breeds into our 5 main classes, filters out microscopic noise boxes (`area < 0.25%`), and balances every class equally (`2,000` images per class -> `10,000` total clean images).

### 2. Cloud GPU Crashes (`Kernel Died` / Shared Memory Error)
When we first tried training on a cloud server with `imgsz=1024`, `batch=128`, and `workers=8`, JupyterLab kept crashing with `Kernel Died` errors because Docker containers have limited shared RAM (`/dev/shm`), and upscaling small images to 1024p caused catastrophic forgetting.

**Our Solution (`train.py`):**
* Locked resolution to **`imgsz=640`** (the natural resolution of our dataset).
* Set **`batch=32`** and **`workers=2`**, which dropped VRAM usage to a super stable **6.21 GB** on an NVIDIA A40 GPU and completely eliminated RAM crashes.

---

## 📊 Final Training Results (100 Epochs)

After cleaning and balancing the dataset, we trained `yolo26n.pt` for 100 epochs (`1.186 hours` on an NVIDIA A40 GPU) using the `SGD` optimizer.

### Validation Metrics (`weights/best.pt`)

| Class | Validation Images | Bounding Boxes | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Classes (Average)** | **1,000** | **4,368** | **0.906 (90.6%)** | **0.901 (90.1%)** | **0.940 (94.0%)** | **0.673 (67.3%)** |
| **Daisy** | 202 | 613 | 0.900 | 0.869 | **0.935** | 0.568 |
| **Sunflower** | 200 | 1,643 | 0.819 | 0.893 | **0.925** | 0.660 |
| **Rose** | 200 | 821 | 0.869 | 0.900 | **0.928** | 0.634 |
| **Tulip** | 200 | 956 | 0.985 | 0.995 | **0.995** | 0.877 |
| **Dandelion** | 203 | 335 | 0.956 | 0.848 | **0.916** | 0.625 |

* **Inference Speed:** `5.2 ms` per image (~192 FPS on NVIDIA A40)
* **Model Size:** `5.5 MB` (`weights/best.pt`)

---

## ☁️ Cloud Deployment Guide (RunPod Setup)

If you want to reproduce this project from scratch, you will need a cloud GPU because extracting 22 ZIP files and training YOLO on 10,000 images requires good hardware. We used **RunPod.io** for this project. 

Here is exactly how you can set up the same environment:

### Step 1: Renting the GPU
1. Go to RunPod.io and create an account.
2. Go to **Pods** -> **Deploy**.
3. Select **Secure Cloud** (or Community Cloud for cheaper rates).
4. Choose an **NVIDIA A40** or **RTX 3090 / 4090** (We need at least 12GB+ VRAM, A40 has 48GB which is perfect).
5. **CRITICAL:** Click `Customize Deploy` before starting the pod:
   * **Template:** Select the official `RunPod PyTorch 2.x` template (it comes with JupyterLab and Python pre-installed).
   * **Container Disk:** Set to `50 GB` (to hold the Docker image).
   * **Volume Disk:** Set to `100 GB` (We need this space to download 22 ZIP files, extract them, and save the curated dataset).
6. Click **Deploy**.

### Step 2: Accessing the Workspace
1. Once your Pod is running, click the **Connect** button.
2. Select **Connect to Jupyter Lab** (Port 8888).
3. Open a new **Terminal** inside JupyterLab. Your working directory should be `/workspace`.

### Step 3: Downloading the Code and Preparing the Environment
Run these commands in your RunPod JupyterLab terminal:

```bash
# 1. Clone this repository
git clone https://github.com/nasabaxx/Flower-Detection-YOLOv26.git
cd Flower-Detection-YOLOv26

# 2. Install required Python libraries
pip install -r requirements.txt
```

### Step 4: Fetching and Cleaning the Dataset
We use `gdown` to download the raw dataset archives directly from our Google Drive to the RunPod server. Run these commands inside the `Flower-Detection-YOLOv26` directory:

```bash
# Download raw ZIP archives into the total_dataset folder
gdown --folder https://drive.google.com/drive/folders/1xpbi9BJMLVrpwgXfZe90wDCOpa0gNmiN?usp=sharing -O /workspace/total_dataset

# Run the automated merger, deduplicator, and class balancer
python prepare_dataset.py
```
*This script will take a few minutes. It will extract everything and create a clean, balanced dataset inside `/workspace/curated_dataset`.*

### Step 5: Training the Model
Now that the data is ready, start the hardware-aware YOLOv26 Nano training by running:

```bash
python train.py
```
*(Training takes about 1.2 hours on an A40 GPU. Once finished, your weights will be saved as `best.pt` in the `runs/` folder.)*

---

## 💻 Running Inference Locally (Testing the Model)

If you don't want to train the model and just want to test our pre-trained weights (`weights/best.pt`) on your own computer:

1. Clone this repo and install requirements:
```bash
git clone https://github.com/nasabaxx/Flower-Detection-YOLOv26.git
cd Flower-Detection-YOLOv26
pip install -r requirements.txt
```

2. Run the prediction script on any image:
```bash
python predict.py --source path/to/your_flower_photo.jpg --weights weights/best.pt
```
The model will output the detected flowers with bounding boxes and save the image in the `runs/detect/predict/` folder!


