from flask import Flask, render_template_string, request, redirect, session, send_file
from flask_bcrypt import Bcrypt
import pandas as pd
import pickle
import os
from dotenv import load_dotenv
from datetime import datetime
from werkzeug.utils import secure_filename
from supabase import create_client, Client


# --- FLASK APP SETUP ---
app = Flask(__name__)
app.secret_key = "loke2005@."
bcrypt = Bcrypt(app)

# --- SUPABASE CONFIG ---
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- LOAD TRAINED MODEL + METADATA ---
MODEL_PATH = "model.pkl"
TARGET_COLS_PATH = "target_cols.pkl"
FEATURE_COLS_PATH = "feature_cols.pkl"

model = None
target_cols = None
feature_cols = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(TARGET_COLS_PATH, "rb") as f:
        target_cols = pickle.load(f)
    if os.path.exists(FEATURE_COLS_PATH):
        with open(FEATURE_COLS_PATH, "rb") as f:
            feature_cols = pickle.load(f)
    print("✅ Model and columns loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- FILE UPLOAD CONFIG ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {".csv"}

def allowed_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS

# --- ROUTES ---
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATES['index.html'])

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password_hash = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")

        try:
            supabase.table("users").insert({
                "username": username,
                "email": email,
                "password": password_hash
            }).execute()
        except Exception as e:
            return f"❌ Signup failed: {e}"
        return redirect("/login")
    return render_template_string(HTML_TEMPLATES['signup.html'])

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        response = supabase.table("users").select("*").eq("email", email).execute()
        user_data = response.data
        if user_data and bcrypt.check_password_hash(user_data[0]["password"], password):
            session["user_id"] = user_data[0]["id"]
            session["username"] = user_data[0]["username"]
            return redirect("/dashboard")
        return "❌ Invalid credentials"
    return render_template_string(HTML_TEMPLATES['login.html'])

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")
    model_info = None
    try:
        expected_features = getattr(model, "n_features_in_", None)
        model_info = {
            "targets": len(target_cols) if target_cols else None,
            "expected_features": expected_features
        }
    except Exception:
        model_info = None
    return render_template_string(
        HTML_TEMPLATES['dashboard.html'],
        username=session["username"],
        model_info=model_info
    )

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template_string(HTML_TEMPLATES['predict.html'])

    if "file" not in request.files:
        return render_template_string(HTML_TEMPLATES['predict.html'], prediction_result="❌ No file uploaded.")
    file = request.files["file"]
    if file.filename == "" or not allowed_file(file.filename):
        return render_template_string(HTML_TEMPLATES['predict.html'], prediction_result="❌ Invalid file.")

    if model is None or target_cols is None:
        return render_template_string(HTML_TEMPLATES['predict.html'], prediction_result="❌ Model not loaded.")

    try:
        safe_name = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
        file.save(filepath)

        df = pd.read_csv(filepath)
        ids = df["ID"] if "ID" in df.columns else None
        X = df.drop(columns=["ID"], errors="ignore")

        if feature_cols:
            missing = [c for c in feature_cols if c not in X.columns]
            if missing:
                return render_template_string(HTML_TEMPLATES['predict.html'], prediction_result=f"❌ Missing features: {missing[:10]}{'...' if len(missing)>10 else ''}")
            X = X[feature_cols]
        else:
            expected = getattr(model, "n_features_in_", None)
            if expected and X.shape[1] != expected:
                return render_template_string(HTML_TEMPLATES['predict.html'], prediction_result=f"❌ Model expects {expected} features, got {X.shape[1]}")

        preds = model.predict(X)
        pred_df = pd.DataFrame(preds, columns=target_cols)
        if ids is not None:
            pred_df.insert(0, "ID", ids.values)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"predictions_{ts}.csv"
        result_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
        pred_df.to_csv(result_path, index=False)
        session["last_prediction_csv"] = out_name

        table_html = pred_df.head(10).to_html(classes="table-auto border", index=False)
        return render_template_string(HTML_TEMPLATES['predict.html'], prediction_result=f"✅ Prediction done! {len(pred_df)} rows processed. <a href='/download'>📥 Download CSV</a><br><br>{table_html}")

    except Exception as e:
        return render_template_string(HTML_TEMPLATES['predict.html'], prediction_result=f"❌ Error: {e}")

@app.route("/download")
def download():
    out_name = session.get("last_prediction_csv")
    if not out_name:
        return "❌ No predictions available.", 400
    result_path = os.path.join(app.config["UPLOAD_FOLDER"], out_name)
    if not os.path.exists(result_path):
        return "❌ Prediction file not found.", 404
    return send_file(result_path, mimetype="text/csv", as_attachment=True, download_name=out_name)


# --- HTML TEMPLATES ---
HTML_TEMPLATES = {
    'index.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Home</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-md text-center space-y-6">
        <h1 class="text-4xl font-bold text-gray-800">Welcome</h1>
        <p class="text-gray-600">Please choose an option to continue.</p>
        <div class="space-y-4">
            <a href="/login" class="block w-full bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-blue-700 transition-colors">
                Log In
            </a>
            <a href="/signup" class="block w-full bg-green-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-green-700 transition-colors">
                Sign Up
            </a>
            <a href="/predict" class="block w-full bg-purple-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-purple-700 transition-colors">
                Make a Prediction
            </a>
        </div>
    </div>
</body>
</html>
""",
    'signup.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Up</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-md">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Create an Account</h1>
        <form action="/signup" method="post" class="space-y-4">
            <div>
                <label for="username" class="block text-sm font-medium text-gray-700">Username</label>
                <input type="text" id="username" name="username" required
                       class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            <div>
                <label for="email" class="block text-sm font-medium text-gray-700">Email</label>
                <input type="email" id="email" name="email" required
                       class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            <div>
                <label for="password" class="block text-sm font-medium text-gray-700">Password</label>
                <input type="password" id="password" name="password" required
                       class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            <button type="submit"
                    class="w-full bg-green-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-green-700 transition-colors">
                Sign Up
            </button>
        </form>
        <p class="mt-4 text-center text-gray-600">
            Already have an account? <a href="/login" class="text-blue-600 hover:underline">Log In</a>
        </p>
    </div>
</body>
</html>
""",
    'login.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Log In</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-md">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Log In</h1>
        <form action="/login" method="post" class="space-y-4">
            <div>
                <label for="email" class="block text-sm font-medium text-gray-700">Email</label>
                <input type="email" id="email" name="email" required
                       class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            <div>
                <label for="password" class="block text-sm font-medium text-gray-700">Password</label>
                <input type="password" id="password" name="password" required
                       class="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-blue-500 focus:border-blue-500">
            </div>
            <button type="submit"
                    class="w-full bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-blue-700 transition-colors">
                Log In
            </button>
        </form>
        <p class="mt-4 text-center text-gray-600">
            Don't have an account? <a href="/signup" class="text-green-600 hover:underline">Sign Up</a>
        </p>
    </div>
</body>
</html>
""",
    'dashboard.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>body { font-family: 'Inter', sans-serif; }</style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-lg text-center space-y-6">
        <h1 class="text-4xl font-bold text-gray-800">Hello, {{ username }}!</h1>
        <p class="text-gray-600">Welcome to your dashboard.</p>

        {% if model_info %}
        <div class="rounded-xl p-4 bg-purple-50 text-purple-800 text-left">
            <div class="font-semibold mb-1">Model summary</div>
            <div>Expected features: {{ model_info.expected_features }}</div>
            <div>Outputs (targets): {{ model_info.targets }}</div>
        </div>
        {% endif %}

        <div class="space-y-4">
            <a href="/predict" class="block w-full bg-purple-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-purple-700 transition-colors">
                Go to Prediction Page
            </a>
            <a href="/logout" class="block w-full bg-red-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-red-700 transition-colors">
                Log Out
            </a>
        </div>
    </div>
</body>
</html>
""",
    'predict.html': """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Upload File for Prediction</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap">
    <style>
        body { font-family: 'Inter', sans-serif; }
        .drag-area {
            border: 2px dashed #7e22ce;
            border-radius: 12px;
            background: #faf5ff;
            padding: 40px;
            text-align: center;
            transition: 0.3s;
        }
        .drag-area.dragover {
            background: #ede9fe;
            border-color: #5b21b6;
        }
        .preview { max-height: 420px; overflow: auto; }
    </style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
    <div class="bg-white p-8 rounded-xl shadow-lg w-full max-w-md">
        <h1 class="text-3xl font-bold text-center text-gray-800 mb-6">Upload File for Prediction</h1>
        <form action="/predict" method="post" enctype="multipart/form-data" class="space-y-4">
            <div id="drop-area" class="drag-area">
                <p class="text-gray-600">Drag & Drop your .csv here<br>or click to select</p>
                <input type="file" id="fileInput" name="file" accept=".csv" class="hidden" required>
                <button type="button" id="browseBtn"
                        class="mt-3 px-4 py-2 bg-purple-600 text-white rounded-lg shadow hover:bg-purple-700">
                    Browse File
                </button>
                <p id="fileName" class="mt-2 text-sm text-gray-700"></p>
            </div>
            <button type="submit"
                    class="w-full bg-purple-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-purple-700 transition-colors">
                Predict
            </button>
        </form>

        {% if prediction_result %}
            <div class="mt-6 p-4 bg-purple-100 text-purple-800 rounded-lg shadow-inner text-center font-bold preview">
                {{ prediction_result | safe }}
            </div>
        {% endif %}

        <p class="mt-4 text-center text-gray-600">
            <a href="/dashboard" class="text-blue-600 hover:underline">Go to Dashboard</a>
        </p>
    </div>

    <script>
        const dropArea = document.getElementById("drop-area");
        const fileInput = document.getElementById("fileInput");
        const browseBtn = document.getElementById("browseBtn");
        const fileName = document.getElementById("fileName");

        browseBtn.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", () => {
            fileName.textContent = fileInput.files[0]?.name || "";
        });

        dropArea.addEventListener("dragover", (e) => {
            e.preventDefault();
            dropArea.classList.add("dragover");
        });
        dropArea.addEventListener("dragleave", () => {
            dropArea.classList.remove("dragover");
        });
        dropArea.addEventListener("drop", (e) => {
            e.preventDefault();
            dropArea.classList.remove("dragover");
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                fileName.textContent = e.dataTransfer.files[0].name;
            }
        });
    </script>
</body>
</html>
"""
}

if __name__ == "__main__":
    app.run(debug=True)
