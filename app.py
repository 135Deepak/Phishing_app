# app.py
import os
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load environment variables from .env
load_dotenv()

# Config
DATA_PATH = "PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_FILE = "phish_model.pkl"
VECT_FILE = "vectorizer.pkl"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")   # set in .env
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")       # set in .env (string or number)

app = Flask(__name__)
CORS(app)

# ------------------- Telegram helper -------------------
def send_telegram_alert(chat_id: str, text: str) -> (bool, str):
    """
    Send a Telegram message via the Bot API.
    Returns (success, message)
    """
    if not TELEGRAM_BOT_TOKEN:
        return False, "Telegram bot token not configured."

    if not chat_id:
        return False, "Telegram chat_id not configured."

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        if r.status_code == 200:
            return True, "Telegram alert sent."
        else:
            return False, f"Telegram API error {r.status_code}: {r.text}"
    except Exception as e:
        return False, f"Exception sending Telegram: {e}"

# ------------------- Model load/train -------------------
def detect_label_and_load_dataset(path):
    df = pd.read_csv(path)
    # normalize column names
    df.columns = [c.strip() for c in df.columns.tolist()]
    # find URL column and label column
    url_col = next((c for c in df.columns if c.lower() == "url"), None)
    possible_labels = [c for c in df.columns if c.lower() in ("label", "status", "result", "type", "class")]
    label_col = possible_labels[0] if possible_labels else None
    return df, url_col, label_col

def train_and_save_model(data_path=DATA_PATH, model_file=MODEL_FILE, vect_file=VECT_FILE, limit_rows=10000):
    df, url_col, label_col = detect_label_and_load_dataset(data_path)
    if url_col is None or label_col is None:
        raise KeyError(f"Dataset must contain URL and label-like column. Found: {df.columns.tolist()}")

    # Optionally limit rows for quicker dev training
    if limit_rows:
        df = df.head(limit_rows)

    X = df[url_col].astype(str)
    y = df[label_col].astype(str)

    vectorizer = TfidfVectorizer(max_features=5000, analyzer='char')
    X_vec = vectorizer.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_vec, y)

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vect_file)
    return model, vectorizer

def load_or_train():
    if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECT_FILE)
        # detect label col for interpretation
        df, url_col, label_col = detect_label_and_load_dataset(DATA_PATH)
        return model, vectorizer, label_col
    else:
        model, vectorizer = train_and_save_model()
        df, url_col, label_col = detect_label_and_load_dataset(DATA_PATH)
        return model, vectorizer, label_col

model, vectorizer, LABEL_COL = load_or_train()

# ------------------- Routes -------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check_url", methods=["POST"])
def check_url():
    """
    Expects form-data:
      - url (required)
      - telegram_chat_id (optional; if not provided server will use TELEGRAM_CHAT_ID env)
    """
    url = request.form.get("url", "").strip()
    telegram_chat_id = request.form.get("telegram_chat_id", "").strip() or TELEGRAM_CHAT_ID

    if not url:
        return jsonify({"error": "No URL provided."}), 400

    try:
        # vectorize & predict
        vec = vectorizer.transform([url])
        pred = model.predict(vec)[0]
        # try probability if available
        probability = None
        try:
            prob = model.predict_proba(vec)[0].max() * 100
            probability = round(float(prob), 2)
        except Exception:
            probability = None

        # Interpret unsafe labels generically
        pred_str = str(pred).lower()
        is_unsafe = any(k in pred_str for k in ["phish", "malicious", "bad", "unsafe"]) or pred_str in ("1", "true")

        result = {
            "url": url,
            "prediction": str(pred),
            "probability": probability,
            "is_unsafe": is_unsafe
        }

        # If unsafe -> send telegram alert (if chat id available)
        if is_unsafe and telegram_chat_id:
            msg_text = (
                f"🚨 *PhishGuard Alert*\n\n"
                f"Detected an unsafe link:\n`{url}`\n\n"
                f"Prediction: *{pred}*\n\n"
                f"Do NOT open this link unless you expect it."
            )
            ok, msg = send_telegram_alert(telegram_chat_id, msg_text)
            result["telegram_sent"] = ok
            result["telegram_message"] = msg
        else:
            result["telegram_sent"] = False
            result["telegram_message"] = "No telegram alert sent."

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# ------------------- Run -------------------
if __name__ == "__main__":
    print("Starting PhishGuard server at http://127.0.0.1:5000")
    app.run(debug=True)
