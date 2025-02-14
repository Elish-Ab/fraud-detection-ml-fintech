# Fraud Detection Model API

This project provides a machine learning-based fraud detection model with a Flask API for serving predictions. The model identifies fraudulent transactions based on financial transaction data.

## 📌 Project Structure

```
WEEK8/
│── data/                      # Raw and processed datasets
│   ├── creditcard.csv
│   ├── Fraud_Data.csv
│   ├── IpAddress_to_Country.csv
│   ├── Preprocessed_Data.csv
│── logs/                      # Logging output
│── mlruns/                    # MLflow experiment tracking
│── notebooks/                 # Jupyter notebooks for data cleaning & analysis
│   ├── data_clean.ipynb
│── project/                   # Model deployment
│   ├── mlruns
│   ├── mlflow.db
│   ├── serve_model.py         # Flask API for model inference
│── scripts/                   # Model training and preprocessing scripts
│   ├── data_cleaning.py
│   ├── model_training.py
│── .gitignore
│── Dockerfile                 # Docker setup for API deployment
│── requirements.txt           # Python dependencies
│── README.md                  # Documentation
```

## 🚀 Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/fraud-detection-api.git
cd fraud-detection-api
```

### 2️⃣ Create a Virtual Environment & Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Train the Model

Run the model training script to generate the trained model:

```bash
python scripts/model_training.py
```

### 4️⃣ Run the Flask API

Start the API server for model inference:

```bash
python project/serve_model.py
```

The API will be available at:

```
http://127.0.0.1:5000
```

## 🔥 API Endpoints

### 1️⃣ Health Check

```http
GET /health
```

**Response:**

```json
{"status": "API is running"}
```

### 2️⃣ Predict Fraud

```http
POST /predict
```

**Request Body (JSON):**

```json
{
  "feature1": value,
  "feature2": value,
  "feature3": value
}
```

**Response:**

```json
{"prediction": 1}   # 1 = Fraud, 0 = Legitimate
```

## 🐳 Docker Deployment

### 1️⃣ Build the Docker Image

```bash
docker build -t fraud-detection-model .
```

### 2️⃣ Run the Container

```bash
docker run -p 5000:5000 fraud-detection-model
```

## 📜 License

This project is licensed under the MIT License.

you can contact me: +251909492011
