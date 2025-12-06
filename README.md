# 🤖 Hand Gesture Recognition using Deep Learning

This repository contains a **Hand Gesture Recognition System** built using **Python, OpenCV, TensorFlow**, and **Transfer Learning on the InceptionV3 model**.  
It recognizes **A–Z gestures**, including **Nothing**, **Space**, and **Delete**.

---

## ✨ Features

### ✔ 1. Image-Based Prediction
Upload any hand gesture image and get:
- Top-5 prediction scores  
- Best-matched gesture  
- Automatic sequence updating  

### ✔ 2. Real-Time Webcam Gesture Recognition
- Live gesture detection inside a defined ROI  
- Stable predictions (using debounce + voting logic)  
- Automatic sentence formation  

### ✔ 3. ROI Capture Mode
- Captures only the selected box region  
- Predicts and appends results to sentence  

### ✔ 4. Tkinter GUI Interface
Includes:
- Image upload  
- Prediction window  
- Live webcam feed  
- ROI capture  
- Delete-last and Clear buttons  
- Status logs  

### ✔ 5. Deep Learning Training Support
- Uses **Transfer Learning on InceptionV3**  
- Trained on a **1GB ASL Dataset**  
- Achieves **≈95% accuracy**

---

## 📂 Dataset

The project uses a **1GB ASL Alphabet Dataset** sourced from Kaggle.

Dataset contains:
- 26 folders for **A–Z**
- Additional folders: **nothing**, **space**, **del**
- Each folder includes **hundreds of gesture samples**

This dataset is used to train a custom classifier using transfer learning.

---

## ⚙️ System Requirements

### Software Requirements
- Python **3.5 – 3.10**
- TensorFlow **1.15** (supports frozen graphs)
- OpenCV-Python
- Tkinter
- NumPy
- Matplotlib
- Pillow (PIL)
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```
---

## 🏗️ Training the Model
To train the InceptionV3-based classifier, run:

```bash
python train.py \
  --bottleneck_dir=logs/bottlenecks \
  --how_many_training_steps=2000 \
  --model_dir=inception \
  --summaries_dir=logs/training_summaries/basic \
  --output_graph=logs/output_graph.pb \
  --output_labels=logs/output_labels.txt \
  --image_dir=./dataset
```
Training with the full dataset may take 2–3 hours depending on system performance.

---

## 🧪 Classifying a Single Image
Use this command to classify any image:

```bash
python classify.py path/to/image.jpg
```

---
## 📊 Evaluating the Entire Dataset
To compute prediction accuracy for all classes:

```bash
python batch_eval_fixed.py
```
Outputs include:
- Per-class accuracy
- Total accuracy
- Confidence distribution
  
![WhatsApp Image 2025-12-03 at 23 28 49_0e0e9297](https://github.com/user-attachments/assets/53b4e51c-71f7-4817-9f09-ee0bdfa8c8e7)

---

## 🎥 Using the Webcam (Live Demo)
Start the GUI with:

```bash
python asl_tkinter_gui_capture.py
```
GUI Features:
- Upload and predict images
- Open webcam
- Capture ROI
- Real-time prediction
- Sentence formation
- Delete & Clear controls

Your hand must be inside the blue ROI rectangle for recognition.

---
## 🖼️ Screenshots
### 🔹 GUI Home Screen

<img width="1493" height="869" alt="Screenshot 2025-12-04 193000" src="https://github.com/user-attachments/assets/c3dd382e-956d-4468-8680-cf28dc3ba5e6" />


### 🔹 Webcam Activated

<img width="1495" height="950" alt="Screenshot 2025-12-04 193054" src="https://github.com/user-attachments/assets/c00fde30-4d85-448e-9ab1-902d609a0ad5" />


### 🔹 Uploading an Image for Prediction

<img width="1494" height="959" alt="Screenshot 2025-12-04 193135" src="https://github.com/user-attachments/assets/3867eb99-f3fc-4b34-982d-11f624ec6d34" />

### 🔹 Prediction of Uploaded Image with top 5 Predictions

<img width="1490" height="953" alt="Screenshot 2025-12-04 193211" src="https://github.com/user-attachments/assets/bc24bfac-12ef-4c17-8941-bdba938fb7c7" />


### 🔹 Real-Time Prediction

<img width="1491" height="959" alt="Screenshot 2025-12-04 193413" src="https://github.com/user-attachments/assets/cea9f7e0-e842-429f-af6b-79855d76f421" />


### 🔹 ROI Capture

<img width="1488" height="952" alt="Screenshot 2025-12-04 193520" src="https://github.com/user-attachments/assets/f614a43f-6f17-4d67-9110-7483020c3d7e" />


