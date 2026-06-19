# 💳 Credit Card Fraud Detection

XGBoost-powered fraud detection system with a **FastAPI** REST backend and **Streamlit** dashboard.

---

## 📁 Project Structure

```
credit-fraud-detection/
├── train_model.py      # Model training aur evaluation
├── app.py              # FastAPI REST API
├── streamlit_app.py    # Streamlit web dashboard
├── model/              # Trained model (.pkl) yahan save hota hai
├── requirements.txt    # Python dependencies
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Dependencies Install Karo

```bash
pip install -r requirements.txt
```

### 2. Model Train Karo

```bash
# creditcard.csv apni machine par rakho (Kaggle se download karo)
python train_model.py --data creditcard.csv
```

Model `model/best_model.pkl` mein save ho jayega.

### 3. FastAPI Server Chalao

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
uvicorn app:app --reload
```

API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Streamlit Dashboard Chalao

```bash
# Naye terminal mein
streamlit run streamlit_app.py
```
#streamlit run streamlit_app.py
Dashboard: [http://localhost:8501](http://localhost:8501)

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/predict` | Single transaction predict karo |
| POST | `/predict-batch` | Multiple transactions batch mein |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Time": 406.0,
    "Amount": 149.62,
    "V1": -1.359, "V2": -0.072, "V3": 2.536,
    "V4": 1.378, "V5": -0.338, "V6": 0.462,
    "V7": 0.239, "V8": 0.098, "V9": 0.363,
    "V10": 0.090, "V11": -0.551, "V12": -0.617,
    "V13": -0.991, "V14": -0.311, "V15": 1.468,
    "V16": -0.470, "V17": 0.207, "V18": 0.025,
    "V19": 0.403, "V20": 0.251, "V21": -0.018,
    "V22": 0.277, "V23": -0.110, "V24": 0.066,
    "V25": 0.128, "V26": -0.189, "V27": 0.133,
    "V28": -0.021
  }'
```

### Example Response

```json
{
  "prediction": 0,
  "label": "Legit",
  "fraud_probability": 0.0023,
  "confidence": "High"
}
```

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | XGBoost Classifier |
| Imbalance Handling | `scale_pos_weight` |
| Preprocessing | StandardScaler (Time + Amount + V features) |
| Main Metric | AUPRC (Area Under Precision-Recall Curve) |
| Dataset | [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |

---

## 📊 Dataset

Dataset GitHub par **upload nahi ki** (284K rows, 150MB). Kaggle se download karo:

👉 [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 📸 Screenshots

### Dashboard
Single transaction aur batch CSV upload dono support karta hai.

### API Docs
FastAPI auto-generated Swagger UI → `http://localhost:8000/docs`

---

## 🤝 Contributing

Pull requests welcome hain. Koi issue ho to GitHub Issues mein report karo.

---

## 📄 License

MIT License
