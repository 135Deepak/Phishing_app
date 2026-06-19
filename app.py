import os
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd
import joblib
import csv
from datetime import datetime
import numpy as np
from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ---------------- CONFIG ----------------
DATA_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_FILE = "phish_model.pkl"
VECT_FILE = "vectorizer.pkl"

# Load env
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

app = Flask(__name__)
CORS(app)

# ---------------- TELEGRAM ----------------
def send_telegram_alert(chat_id, text):
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False, "Telegram not configured"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }

    try:
        r = requests.post(url, data=payload)
        return (r.status_code == 200, r.text)
    except Exception as e:
        return False, str(e)

# ---------------- FEATURE ENGINEERING ----------------
def extract_features(url):
    url = url.lower()

    suspicious_keywords = [
        "login", "verify", "bank", "secure", "account",
        "update", "free", "gift", "card", "password"
    ]

    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        url.count('//'),
        1 if 'https' in url else 0,
        sum(1 for word in suspicious_keywords if word in url)
    ]

# ---------------- DATA LOAD ----------------
def is_suspicious_domain(url):
    try:
        domain = urlparse(url).netloc.lower()

        # suspicious patterns
        if '-' in domain:
            return True
        
        if domain.endswith(('.xyz', '.ru', '.tk', '.ml', '.ga')):
            return True

        if len(domain.split('.')) < 2:
            return True

        # fake-like names
        if any(word in domain for word in ["account", "secure", "update"]):
            return True

        return False
    except:
        return False
    
def detect_label_and_load_dataset(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns.tolist()]

    url_col = next((c for c in df.columns if c.lower() == "url"), None)
    label_col = next((c for c in df.columns if c.lower() in ["label","status","result","type","class"]), None)

    return df, url_col, label_col

# ---------------- TRAIN MODEL ----------------
def train_and_save_model():
    df, url_col, label_col = detect_label_and_load_dataset(DATA_PATH)

    X_text = df[url_col].astype(str)
    y = df[label_col].astype(str)

    vectorizer = TfidfVectorizer(max_features=5000, analyzer='char')
    X_vec = vectorizer.fit_transform(X_text)

    X_extra = np.array([extract_features(url) for url in X_text])
    X_combined = np.hstack((X_vec.toarray(), X_extra))

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_combined, y)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECT_FILE)

    return model, vectorizer

# ---------------- LOAD MODEL ----------------
def load_or_train():
    return joblib.load(MODEL_FILE), joblib.load(VECT_FILE)

model, vectorizer = load_or_train()

# ---------------- LOGGING ----------------
def log_scan(url, prediction, probability):
    file_exists = os.path.exists("scan_log.csv")

    with open("scan_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["url","prediction","probability","time"])
        writer.writerow([url, prediction, probability, datetime.now()])

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/history")
def history():
    data = []
    if os.path.exists("scan_log.csv"):
        with open("scan_log.csv", "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    return jsonify(data[::-1])

@app.route("/stats")
def stats():
    total = 0
    unsafe = 0

    if os.path.exists("scan_log.csv"):
        with open("scan_log.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total += 1
                if row["prediction"].lower() == "unsafe":
                    unsafe += 1
                    

    return jsonify({
        "total": total,
        "unsafe": unsafe,
        "safe": total - unsafe
    })

@app.route("/check_url", methods=["POST"])
def check_url():
    url = request.form.get("url", "").strip()
    telegram_chat_id = request.form.get("telegram_chat_id", "").strip() or TELEGRAM_CHAT_ID

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    try:
        # VECTOR + FEATURES
        vec = vectorizer.transform([url])
        extra = np.array([extract_features(url)])

        combined = np.hstack((vec.toarray(), extra))

        pred = model.predict(combined)[0]

        # PROBABILITY
        try:
            prob = model.predict_proba(combined)[0].max() * 100
            probability = round(float(prob), 2)
        except:
            probability = None

        pred_str = str(pred).lower()

        # ML decision
        is_unsafe_ml = any(k in pred_str for k in ["phish","malicious","bad","unsafe"]) or pred_str in ("1","true")

        # RULE decision
        suspicious_words = ["free","gift","verify","bank"]
        keyword_count = sum(1 for w in suspicious_words if w in url.lower())

        is_unsafe_rule = keyword_count >= 2

        # FINAL
        domain_flag = is_suspicious_domain(url)

        is_unsafe = is_unsafe_ml or is_unsafe_rule or domain_flag
        result = {
            "url": url,
            "prediction": str(pred),
            "probability": probability,
            "is_unsafe": is_unsafe
        }

        status = "unsafe" if is_unsafe else "safe"
        log_scan(url, status, probability)

        # TELEGRAM
        if is_unsafe and telegram_chat_id:
            msg = f"🚨 PhishGuard Alert\nUnsafe URL:\n{url}"
            ok, msg_res = send_telegram_alert(telegram_chat_id, msg)
            result["telegram_sent"] = ok
            result["telegram_message"] = msg_res
        else:
            result["telegram_sent"] = False
            result["telegram_message"] = "No alert sent"

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear_history", methods=["POST"])
def clear_history():
    try:
        with open("scan_log.csv", "w") as f:
            f.write("url,prediction,probability,time\n")
        return jsonify({"message": "History cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("Running on http://127.0.0.1:5000")
    app.run(debug=True)