# CODE-ALPHA-MACHINE-LEARNING-INTERNSHIP-TASK-2
```markdown
# 🎤 Emotion Recognition from Speech  
**CodeAlpha Machine Learning Internship – Task 2**

[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Notion](https://img.shields.io/badge/Notebook-RAVDESS-important)](#)

A deep learning solution that classifies **8 human emotions** from raw speech audio.  
Extracts **MFCC features** with delta and delta-delta coefficients, then trains a  
2D Convolutional Neural Network (CNN) to recognise emotions such as happy, angry, sad, etc.

> 📢 **Live Demo** available inside the Colab notebook – upload your own `.wav` file and get a real‑time prediction.

---

## 📖 Table of Contents

- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Installation & Usage](#-installation--usage)
- [Model Architecture](#-model-architecture)
- [Performance](#-performance)
- [Interactive Demo](#-interactive-demo)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

---

## ✨ Features

- **8‑class emotion recognition**: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- **Actor‑independent split** – model tested on *unseen speakers*, simulating real‑world use
- **MFCC + derivatives**: 40 static coefficients, delta (velocity), and delta‑delta (acceleration)
- **Two architectures**: 2‑D CNN (primary) and optional LSTM for comparison
- **Live testing** – upload any `.wav` file and see the predicted emotion + probability bars
- **Rich visualisation**: waveforms, mel‑spectrograms, confusion matrix, training curves, Grad‑CAM (planned)
- **Google Colab ready** – runs on free GPU (T4) with all dependencies pre‑installed

---

## 🧠 Tech Stack

| Category       | Tools / Libraries                                      |
|----------------|--------------------------------------------------------|
| **Language**   | Python 3                                                |
| **Deep Learning** | TensorFlow / Keras                                    |
| **Audio**      | librosa, soundfile, IPython.display.Audio               |
| **Data**       | pandas, numpy                                           |
| **Visualisation** | matplotlib, seaborn                                   |
| **Environment**| Google Colab, Jupyter Notebook, GitHub                  |

---

## 📂 Project Structure

```
CodeAlpha_EmotionRecognition/
│
├── Emotion_Recognition_from_Speech.ipynb   # Main notebook
├── best_emotion_model.keras                # Saved model (after training)
├── README.md
└── assets/                                 # (optional) screenshots, GIFs
    ├── confusion_matrix.png
    ├── training_curves.png
    └── live_demo.gif
```

---

## 🎧 Dataset

**RAVDESS – The Ryerson Audio-Visual Database of Emotional Speech and Song**

- 24 professional actors (12 male, 12 female)
- 1440 speech files (.wav)
- 8 emotions, two intensity levels (except neutral)
- Sampling rate: 48 kHz (downsampled to 22.05 kHz for processing)
- Labels parsed from file names – no manual annotation required

📥 Download: [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) (or directly via `kagglehub` in the notebook)

---

## 🚀 Installation & Usage

### 1. Open in Google Colab  
Click the Colab badge at the top of this README or  
[use this link](https://colab.research.google.com/) to open the notebook.

### 2. Enable GPU  
Runtime → Change runtime type → Hardware accelerator = **T4 GPU**

### 3. Upload the dataset  
Run the “Environment Setup” cell – it will ask you to upload the `RAVDESS.zip` file.  
*(If you don’t have the zip, the notebook can also download it automatically via `kagglehub`.)*

### 4. Run all cells  
Execute each cell step‑by‑step:  
- Feature extraction (MFCCs)  
- Train/validation/test split (actor‑independent)  
- CNN model training  
- Evaluation (accuracy, confusion matrix, classification report)  
- Live emotion prediction test

### 5. Test your own voice!  
Scroll to the “Live Testing” section, upload a `.wav` file, and see the emotion prediction immediately.

> No local installation required – everything runs in the cloud.

---

## 🏗 Model Architecture

The primary model is a **2‑D Convolutional Neural Network**:

```
Input (time_steps × n_mfcc × 3)  
│
├─ Conv2D(64) → BatchNorm → MaxPool → Dropout  
├─ Conv2D(128) → BatchNorm → MaxPool → Dropout  
├─ Conv2D(256) → BatchNorm → MaxPool → Dropout  
├─ Flatten  
├─ Dense(128) → BatchNorm → Dropout  
└─ Dense(8, softmax)
```

- Input shape: `(130, 40, 3)` – time steps, MFCC coefficients, channels (static, Δ, Δ²)
- Optimizer: Adam (lr = 1e‑4)
- Loss: categorical crossentropy
- Regularisation: dropout (0.3‑0.5) + early stopping

An optional **LSTM** model is also provided in the notebook for comparison.

---

## 📊 Performance

| Metric        | Value       |
|---------------|-------------|
| **Test Accuracy** | 72–78% (actor‑independent) |
| **Best F1‑score**  | ~0.75 (angry, happy)      |
| **Hardest emotions**| calm ↔ neutral, sad       |
| **Training time**   | ~15 min on T4 GPU         |

*Detailed per‑class metrics and confusion matrix are shown inside the notebook.*

---

## 🎥 Interactive Demo

![Live Demo](assets/live_demo.gif) *← replace with a real GIF*

- Upload any speech file (2‑4 seconds, clear emotional content)
- The model outputs the predicted emotion, confidence score, and a probability bar chart
- The processed MFCCs and mel‑spectrogram are visualised as well

---

## 🔮 Future Improvements

- [ ] Combine RAVDESS with TESS/CREMA‑D to increase data diversity
- [ ] Data augmentation: pitch shift, noise injection, time stretching
- [ ] Attention‑based models (Transformer, CRNN)
- [ ] Real‑time microphone input in a web app (Streamlit / Gradio)
- [ ] Export to TensorFlow Lite for on‑device inference
- [ ] Add Grad‑CAM for audio (explainability)

---

## 🤝 Contributing

Contributions are welcome!  
Please open an issue to discuss what you’d like to change, then submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- **RAVDESS** dataset creators: Livingstone SR, Russo FA (2018)
- [**CodeAlpha**](https://www.codealpha.tech) for the internship opportunity and mentorship
- The open‑source community behind TensorFlow, librosa, and scikit‑learn

---

## 📬 Contact

**DAIYAAN SHAIKH** – [LinkedIn](https://www.linkedin.com/in/daiyaan-shaikh-159909377?utm_source=share_via&utm_content=profile&utm_medium=member_android) – [GitHub](https://github.com/shaikhdaiyaan251-cloud)

Project Link: [[https://github.com/shaikhdaiyaan251-cloud/CODE-ALPHA-MACHINE-LEARNING-INTERNSHIP-TASK-](https://github.com/shaikhdaiyaan251-cloud/CODE-ALPHA-MACHINE-LEARNING-INTERNSHIP-TASK-2)]

---

<p align="center">Made with ❤️ as part of the CodeAlpha ML Internship</p>
```
