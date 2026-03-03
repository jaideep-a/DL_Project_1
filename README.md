# Pneumonia Detection from Chest X-Rays

**Live Website:** [https://web-production-5e63f.up.railway.app/](https://web-production-5e63f.up.railway.app/)

A deep learning project to train and test a model for detecting pneumonia from chest X-ray images. This repository contains scripts for training various models (including a deep learning ResNet18 model) and a simple web application for demonstrating predictions.

---

## Project Overview

This project focuses on the end-to-end pipeline of a deep learning solution for medical image classification:
- **Dataset Preparation:** Loading and transforming chest X-ray images.
- **Model Training:** Fine-tuning a pre-trained ResNet18 model on the dataset, alongside baseline naive and classical models.
- **Evaluation:** Testing model accuracy and per-class metrics.
- **Web App Demonstration:** A Flask-based web application to upload an X-ray and receive a pneumonia prediction.

---

## Web Application

The project includes a simple web interface ([available here](https://web-production-5e63f.up.railway.app/)) that allows you to easily test the model. 
- **Upload Scans:** You can drag and drop chest X-ray images into the portal.
- **Get Predictions:** The images are instantly processed by the deep learning model API.
- **View Results:** The model returns the prediction (PNEUMONIA or NORMAL) along with its confidence score directly on the screen.

---

## Project Structure

```text
├── README.md               <- description of project and how to set up and run it
├── requirements.txt        <- requirements file to document dependencies
├── Makefile                <- setup and run project from command line
├── setup.py                <- script to set up project (get data, build features, train model)
├── main.py                 <- main script to run project / web interface
├── scripts/                <- directory for pipeline scripts or utility scripts
│   ├── make_dataset.py     <- script to get data
│   ├── build_features.py   <- script to run pipeline to generate features
│   ├── model.py            <- script to train model and predict
│   ├── train_deep.py       <- train deep learning model (ResNet18)
│   ├── train_naive.py      <- train baseline models
│   └── setup_database.py   <- script to initialize database
├── models/                 <- directory for trained models and analytics
├── data/                   <- directory for project data
│   ├── raw/                <- directory for raw data or script to download
│   ├── processed/          <- directory to store processed data
│   └── outputs/            <- directory to store any output data
├── notebooks/              <- directory to store any exploration notebooks used
├── website/                <- secondary directory for the Flask web application
├── deployed_model_api/     <- backend code for the deployed on-render ML API
├── templates/              <- HTML templates for the web interface
├── static/                 <- static assets (CSS, JS) for the web interface
├── tests/                  <- unit tests for the application
└── .gitignore              <- git ignore file
```

---

## Technology Stack

- **Deep Learning / ML:** PyTorch, torchvision, scikit-learn, pandas, numpy
- **Web Application:** Python 3.11, Flask
- **Deployment:** Render (for web app to demonstrate predictions)

---

## Setup & Installation

### Prerequisites

- Python 3.11+
- pip

### Installation

**1. Clone the repository:**
```bash
git clone https://github.com/jaideep-a/DL_Project_1.git
cd DL_Project_1
```

**2. Create a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

**3. Install dependencies:**
```bash
make install
# or pip install -r requirements.txt
```

**4. Prepare the Dataset:**
The chest X-ray dataset used for model training is the [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) dataset.

Place it under `data/raw/chest_xray/` with the following structure:
```
data/raw/chest_xray/
    train/   NORMAL/   PNEUMONIA/
    val/     NORMAL/   PNEUMONIA/
    test/    NORMAL/   PNEUMONIA/
```

Verify the dataset layout:
```bash
python scripts/make_dataset.py
```

---

## Training and Evaluation

### Deep Learning Model (ResNet18)
To fine-tune a pre-trained ResNet18 model on the chest X-ray dataset, run:
```bash
python scripts/train_deep.py
```
This script will apply data augmentation, handle class imbalance using weighted cross-entropy loss, and evaluate the model's accuracy on the test set.

### Baseline Model
To run and evaluate a naive baseline model (which predicts the most frequent class):
```bash
python scripts/train_naive.py
```

### General Training / Prediction Interface
```bash
# Train models
python scripts/model.py --train

# Run offline prediction on a single image
python scripts/model.py --predict path/to/image.jpg
```

---

## Running the Web App

The project includes a web application for demonstrating the model.

1. Configure environment variables (if required for the prediction API):
```bash
cp .env.example .env
```
Ensure your `.env` contains the required `MODEL_API_URL` pointing to the deployed model API.

2. Run the application:
```bash
make run
# or python main.py
```

Open **http://localhost:5000** in your browser.

---

## Testing

Run unit tests:
```bash
make test
# or python -m pytest tests/ -v
```

---

## Team Members

- **Jaideep Aher**
- **Hanfu**
- **Keming**

---

## Medical Disclaimer

This application and deep learning model are for **educational and research purposes only**. It is not intended for clinical diagnosis. Always consult qualified healthcare professionals for medical advice.

## License

MIT License
