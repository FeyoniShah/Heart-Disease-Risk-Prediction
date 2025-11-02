# app.py
from flask import Flask, request, jsonify, session
from prediction import CHDRiskPredictor
from database import Database
import os
import traceback

# ---------- Configuration ----------
# Set these via environment variables in production
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_USER = os.getenv("DB_USER", "root")
DB_PASS = os.getenv("DB_PASS", "Feyoni@1819")
DB_NAME = os.getenv("DB_NAME", "heart_app")
MODEL_PATH = os.getenv("MODEL_PATH", "trained_models/best_model_logisticregression.pkl")
FLASK_SECRET = os.getenv("FLASK_SECRET", "dev-secret-key")  # change for production

# ---------- App init ----------
app = Flask(__name__)
app.secret_key = FLASK_SECRET

# Initialize DB and predictor once at startup
db = Database(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME)

try:
    predictor = CHDRiskPredictor(model_path=MODEL_PATH)
except Exception as e:
    # If model loading fails, print stack trace and continue (endpoints will return error)
    print("Model load error:", str(e))
    traceback.print_exc()
    predictor = None


# ---------- Helpers ----------
def current_user_id():
    """Return logged-in user's id or None if anonymous."""
    return session.get("user_id")


# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "CHD Prediction API", "status": "ok"})


# Signup: POST /signup { email, password, name (optional) }
@app.route("/signup", methods=["POST"])
def signup():
    payload = request.json or {}
    email = payload.get("email", "").strip()
    password = payload.get("password", "")
    name = payload.get("name", None)

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    try:
        user_id = db.create_user(email=email, password=password, name=name)
        return jsonify({"message": "user created", "user_id": user_id}), 201
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "internal server error", "details": str(e)}), 500


# Login: POST /login { email, password }
@app.route("/login", methods=["POST"])
def login():
    payload = request.json or {}
    email = payload.get("email", "").strip()
    password = payload.get("password", "")

    if not email or not password:
        return jsonify({"error": "email and password required"}), 400

    user = db.verify_user(email, password)
    if user:
        # set minimal session
        session["user_id"] = user["id"]
        session["email"] = user["email"]
        return jsonify({"message": "login successful", "user": {"id": user["id"], "email": user["email"], "name": user.get("name")}}), 200
    else:
        return jsonify({"error": "invalid credentials"}), 401


# Logout: POST /logout
@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "logged out"}), 200






# Add these imports at top of app.py (if not already imported)
from werkzeug.utils import secure_filename
from ocr import MedicalOCRPipeline
import uuid

# Allowed extensions (adjust if you need more)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "tiff", "bmp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# New route: POST /ocr_predict - accepts multipart/form-data with file field 'image'
# @app.route("/ocr_predict", methods=["POST"])
# def ocr_predict():
#     # Ensure predictor is available
#     if predictor is None:
#         return jsonify({"error": "model not loaded on server"}), 500

#     # Expect file in request.files['image']
#     if "image" not in request.files:
#         return jsonify({"error": "no file part 'image' in request"}), 400

#     file = request.files["image"]
#     if file.filename == "":
#         return jsonify({"error": "no file selected"}), 400

#     if not allowed_file(file.filename):
#         return jsonify({"error": f"file type not allowed. Allowed: {ALLOWED_EXTENSIONS}"}), 400

#     # save to temp directory
#     try:
#         filename = secure_filename(file.filename)
#         # create unique temp filename to avoid collisions
#         tmp_name = f"upload_{uuid.uuid4().hex}_{filename}"
#         upload_dir = "uploads"
#         os.makedirs(upload_dir, exist_ok=True)
#         file_path = os.path.join(upload_dir, tmp_name)
#         file.save(file_path)
#     except Exception as e:
#         return jsonify({"error": "failed to save uploaded file", "details": str(e)}), 500

#     # Run OCR pipeline
#     try:
#         ocr_pipeline = MedicalOCRPipeline()
#         ocr_result = ocr_pipeline.process_uploaded_file(file_path)
#         if not ocr_result.get("success"):
#             # process_uploaded_file already returns cleaned structure
#             return jsonify({"error": "ocr_failed", "details": ocr_result.get("error"), "raw_text": ocr_result.get("raw_text", "")}), 400

#         # ocr_result contains chd_format_data mapping (ready for predictor)
#         chd_input = ocr_result.get("chd_format_data", {})

#         if not chd_input:
#             return jsonify({"error": "no usable fields extracted from OCR", "parsed_data": ocr_result.get("parsed_data", {}), "raw_text": ocr_result.get("raw_text", "")}), 400

#         # Some predictors require all features; predictor.validate_input will raise ValueError if missing features
#         results = predictor.predict_single(chd_input)

#         # Save to DB like /predict
#         uid = current_user_id()
#         saved_id = db.insert_prediction(user_id=uid, input_data=chd_input, result_data=results.get("prediction", {}), model_name=predictor.model_package.get("model_name") if predictor.model_package else None)
#         results["prediction"]["db_id"] = saved_id

#         # Return combined response: OCR details + prediction
#         if len(chd_input) < len(predictor.required_features):
#             return jsonify({
#                 "success": True,
#                 "ocr": {
#                     "raw_text": ocr_result.get("raw_text"),
#                     "parsed_data": ocr_result.get("parsed_data"),
#                     "chd_format_data": chd_input
#                 },
#                 "warning": "Some fields missing. Please review and complete form manually."
#             }), 200

#     except ValueError as ve:
#         # from predictor.validate_input (missing features etc)
#         return jsonify({"error": "invalid input for prediction", "details": str(ve), "ocr_parsed": ocr_result.get("parsed_data", {})}), 400
#     except Exception as e:
#         # Keep stack trace in server logs but return friendly message
#         import traceback as tb
#         tb.print_exc()
#         return jsonify({"error": "ocr_predict_failed", "details": str(e)}), 500






@app.route("/ocr_predict", methods=["POST"])
def ocr_predict():
    if predictor is None:
        return jsonify({"error": "model not loaded on server"}), 500

    if "image" not in request.files:
        return jsonify({"error": "no file part 'image' in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "no file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"file type not allowed. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    try:
        filename = secure_filename(file.filename)
        tmp_name = f"upload_{uuid.uuid4().hex}_{filename}"
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, tmp_name)
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": "failed to save uploaded file", "details": str(e)}), 500

    try:
        # Run OCR
        from ocr import process_medical_document
        ocr_result = process_medical_document(file_path)
        
        if not ocr_result.get("success"):
            return jsonify({
                "success": False,
                "error": "OCR failed", 
                "details": ocr_result.get("error"),
                "raw_text": ocr_result.get("raw_text", "")
            }), 400

        # Get extracted data
        chd_input = ocr_result.get("chd_format_data", {})
        
        # Return OCR results without prediction if some fields are missing
        # Let user fill missing fields manually
        return jsonify({
            "success": True,
            "ocr": {
                "raw_text": ocr_result.get("raw_text"),
                "parsed_data": ocr_result.get("parsed_data"),
                "chd_format_data": chd_input,
                "fields_extracted": len(chd_input)
            }
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "OCR processing failed", "details": str(e)}), 500





# Predict: POST /predict { patient data JSON }
# If logged-in, the prediction will be stored under that user.
# Returns the predictor result JSON.
@app.route("/predict", methods=["POST"])
def predict():
    if predictor is None:
        return jsonify({"error": "model not loaded on server"}), 500

    patient_data = request.json
    if not isinstance(patient_data, dict):
        return jsonify({"error": "send patient data as JSON object"}), 400

    try:
        # Make prediction
        results = predictor.predict_single(patient_data)

        # Save to DB (store input + prediction['prediction'])
        uid = current_user_id()
        # result_json store the 'prediction' dictionary
        saved_id = db.insert_prediction(user_id=uid, input_data=patient_data, result_data=results.get("prediction", {}), model_name=predictor.model_package.get("model_name") if predictor.model_package else None)

        # attach saved id for client convenience
        results["prediction"]["db_id"] = saved_id

        return jsonify(results), 200

    except ValueError as ve:
        # validation errors from predictor.validate_input
        return jsonify({"error": "invalid input", "details": str(ve)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "prediction failed", "details": str(e)}), 500


# History: GET /history  -> Returns prediction history for logged-in user
@app.route("/history", methods=["GET"])
def history():
    uid = current_user_id()
    if not uid:
        return jsonify({"error": "not authenticated"}), 401

    try:
        rows = db.get_predictions(uid)
        return jsonify({"predictions": rows}), 200
    except Exception as e:
        return jsonify({"error": "failed to fetch history", "details": str(e)}), 500


# Optional: Admin / recent predictions (no auth here — you can add admin checks)
@app.route("/recent", methods=["GET"])
def recent():
    try:
        rows = db.get_predictions(None) if False else db.get_recent_predictions(limit=50)  # keep safe default
        return jsonify({"recent": rows}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# from flask import render_template  # add this at the top with other imports

# # Route to render the prediction form page
# @app.route('/predict', methods=['GET'])
# def predict_page():
#     return render_template('predict.html')


# @app.route('/login', methods=['GET'])
# def login_page():
#     return render_template('login.html')

# @app.route('/signup', methods=['GET'])
# def signup_page():
#     return render_template('signup.html')



# @app.route("/dashboard")
# def dashboard():
#     if not current_user_id():
#         return "Not authenticated", 401
#     return render_template("dashboard.html")

# @app.route("/result/<int:pred_id>")
# def result(pred_id):
#     if not current_user_id():
#         return "Not authenticated", 401
#     return render_template("result.html")



from flask import render_template, redirect, url_for, session  # add at the top with other imports

# ---------- Page Routes (GET) ----------

# Landing page
@app.route("/index", methods=["GET"])
def index_page():
    return render_template("index.html")

# Prediction form page
@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")

# Login page
@app.route("/login", methods=["GET"])
def login_page():
    return render_template("login.html")

# Signup page
@app.route("/signup", methods=["GET"])
def signup_page():
    return render_template("signup.html")

# Dashboard page (requires authentication)
@app.route("/dashboard", methods=["GET"])
def dashboard_page():
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return render_template("dashboard.html")

# Prediction result page (requires authentication)
@app.route("/result/<int:pred_id>", methods=["GET"])
def result_page(pred_id):
    if not session.get("user_id"):
        return redirect(url_for("login"))
    return render_template("result.html")




# ---------- Run ----------
if __name__ == "__main__":
    # Development server
    app.run(host="0.0.0.0", port=5000, debug=True)
