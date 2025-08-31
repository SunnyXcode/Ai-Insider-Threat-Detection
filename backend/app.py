# backend/app.py
from flask import Flask, jsonify, request, render_template_string
from model import InsiderThreatModel
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model/insider_iforest.pkl"

# -------------------- Initialize or Load Model --------------------
if os.path.exists(MODEL_PATH):
    print(f"🔹 Loading existing model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
else:
    print("⚡ No saved model found, training new model...")
    model = InsiderThreatModel(contamination=0.03)
    model.load_data("data")   # expects data/logon.csv, device.csv, email.csv, file.csv
    model.train(save=True)
    print(f"✅ Model trained and saved to {MODEL_PATH}")

# -------------------- API Endpoints --------------------

@app.route("/risky_users", methods=["GET"])
def risky_users():
    """Return top risky users as JSON"""
    top_n = int(request.args.get("top_n", 20))
    return jsonify(model.get_risky_users(top_n=top_n))

@app.route("/risky_users/table", methods=["GET"])
def risky_users_table():
    """Return risky users as HTML table"""
    top_n = int(request.args.get("top_n", 20))
    records = model.get_risky_users(top_n=top_n)
    if not records:
        return "<h3>No data available</h3>"

    table_html = "<table border='1' style='border-collapse:collapse;padding:6px;'>"
    table_html += "<tr>" + "".join(f"<th>{col}</th>" for col in records[0].keys()) + "</tr>"
    for row in records:
        table_html += "<tr>" + "".join(f"<td>{row[col]}</td>" for col in row.keys()) + "</tr>"
    table_html += "</table>"

    return render_template_string("""
    <html>
      <head>
        <title>Risky Users</title>
        <style>
          body { font-family: Arial, sans-serif; padding: 20px; }
          table { border: 1px solid #ccc; width: 100%; }
          th { background: #f2f2f2; text-align: left; padding: 6px; }
          td { padding: 6px; }
        </style>
      </head>
      <body>
        <h2>Top Risky Users (Ranked)</h2>
        {{table|safe}}
      </body>
    </html>
    """, table=table_html)

@app.route("/user/features", methods=["GET"])
def user_features():
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "pass user param"}), 400

    if not hasattr(model, "get_user_features"):
        return jsonify({"error": "Model method get_user_features missing"}), 500

    try:
        feats = model.get_user_features(user)
        return jsonify(feats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/user/raw", methods=["GET"])
def user_raw():
    user = request.args.get("user")
    if not user:
        return jsonify({"error": "pass user param"}), 400

    if not hasattr(model, "get_user_raw"):
        return jsonify({"error": "Model method get_user_raw missing"}), 500

    try:
        raw = model.get_user_raw(user)
        return jsonify(raw)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/refresh", methods=["POST"])
def refresh():
    try:
        print("♻ Refreshing model...")
        model.load_data("data")
        model.train(save=True)
        joblib.dump(model, MODEL_PATH)
        print(f"✅ Model refreshed and saved to {MODEL_PATH}")
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
