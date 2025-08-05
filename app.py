import os
import base64
import json
import requests
import cv2  
import threading  
from flask import Flask, request, jsonify, render_template, Response  # Add Response
from flask_cors import CORS
import cloudinary
import cloudinary.uploader
import firebase_admin
from firebase_admin import credentials, firestore

# --- App Initialization ---
app = Flask(__name__)
CORS(app)

# --- 1. INITIALIZE SERVICES ---

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firestore initialized successfully.")
except Exception as e:
    print(f"Error initializing Firestore: {e}. Make sure 'serviceAccountKey.json' is present.")
    db = None

# --- API KEYS ---
# 🔴 SECURITY WARNING: Use environment variables in a real application!
GEMINI_API_KEY = "AIzaSyCHmH1920F-an1HgnhyL-CYUUIvatOGD8g"
OPENWEATHER_API_KEY = "8e3b9efee494694b0fad4ccdbf429603"
NEWS_API_KEY = "0730872634df40fd9f8e10acf68f261c"

# Initialize Cloudinary
cloudinary.config(
    cloud_name="dzldp0nc9",
    api_key="327779188319912",
    api_secret="gbhqfWpihlGAJ4FlGkyMqKC0MKk",
    secure=True
)

# --- 2. API & MODEL CONFIGURATION ---
MODEL_NAME = "gemini-1.5-flash-latest"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"

# --- 3. LIVE CAMERA SETUP ---
# Global variable for the camera and a lock for thread-safety
camera = None
camera_lock = threading.Lock()

# --- 4. API ENDPOINTS (Routes) ---

@app.route("/")
def index():
    """Renders the main page."""
    return render_template("index.html")

@app.route('/buyer')
def buyer_page():
    """Renders the buyer marketplace page."""
    return render_template('index2.html')

# --- LIVE CAMERA & PREDICTION ROUTES (NEW) ---

def generate_frames():
    """Generator function to read frames from the camera and stream them."""
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)  # Use the default webcam (0)
            if not camera.isOpened():
                print("Error: Could not open video stream.")
                camera = None
                return

    while True:
        with camera_lock:
            if camera is None:
                break
            success, frame = camera.read()
            if not success:
                break
        
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Streams video from the webcam."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video', methods=['POST'])
def stop_video():
    """Releases the camera hardware."""
    global camera
    with camera_lock:
        if camera:
            camera.release()
            camera = None
    return jsonify({"status": "camera released"})

@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    """Predicts disease from a base64 encoded frame from the live stream."""
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image data in the request"}), 400
    try:
        image_b64 = data['image'].split(',')[1] # Strip the "data:image/jpeg;base64," header
        prompt_text = """
        You are an expert agricultural scientist. Analyze this crop leaf image. Identify any disease. 
        If a disease is found, provide its name and suggest 3 practical remedies. If healthy, state that.
        Your response MUST be a single, valid JSON object with no markdown.
        Use this structure:
        {"is_healthy": <true_or_false>, "disease_name": "...", "remedy_suggestion": ["cure 1", "cure 2", "cure 3"]}
        """
        gemini_payload = {
            "contents": [{"parts": [{"inlineData": {"mime_type": "image/jpeg", "data": image_b64}}, {"text": prompt_text}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        response = requests.post(GEMINI_API_URL, json=gemini_payload, timeout=60)
        response.raise_for_status()
        prediction_data = json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])
        return jsonify({
            "healthy": prediction_data.get("is_healthy", False),
            "disease": prediction_data.get("disease_name", "Unknown"),
            "recommendations": prediction_data.get("remedy_suggestion", [])
        })
    except Exception as e:
        print(f"FRAME PREDICTION ERROR: {e}")
        return jsonify({"error": f"An unexpected error occurred during prediction: {e}"}), 500

# --- PREDICTION FROM FILE UPLOAD ---
@app.route("/predict_upload", methods=["POST"])
def predict_upload():
    """Predicts crop disease from an uploaded leaf image using Gemini."""
    if 'leaf' not in request.files:
        return jsonify({"error": "No 'leaf' file part in the request"}), 400

    file = request.files['leaf']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        image_bytes = file.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt_text = """
        You are an expert agricultural scientist specializing in Indian farming conditions.
        Analyze this crop leaf image. Identify any disease.
        - If healthy, state that.
        - If a disease is found, provide its common name and suggest 3 practical, step-by-step remedies.
        Your entire response MUST be a single, valid JSON object with no markdown formatting.
        Use this exact structure:
        {"is_healthy": <true_or_false>, "disease_name": "...", "remedy_suggestion": ["cure 1", "cure 2", "cure 3"]}
        """
        gemini_payload = {
            "contents": [{"parts": [{"inlineData": {"mime_type": "image/jpeg", "data": image_b64}}, {"text": prompt_text}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        response = requests.post(GEMINI_API_URL, json=gemini_payload, timeout=60)
        response.raise_for_status()
        prediction_data = json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])
        return jsonify({
            "healthy": prediction_data.get("is_healthy", False),
            "disease": prediction_data.get("disease_name", "Unknown"),
            "recommendations": prediction_data.get("remedy_suggestion", [])
        })
    except Exception as e:
        print(f"UPLOAD PREDICTION ERROR: {e}")
        return jsonify({"error": f"An unexpected error occurred during prediction: {e}"}), 500

# --- OTHER ENDPOINTS ---

@app.route("/ask-agro-assistant", methods=["POST"])
def ask_agro_assistant():
    """Handles chatbot queries using the Gemini API."""
    try:
        data = request.get_json()
        user_question = data.get("question", "").strip()
        if not user_question: return jsonify({"error": "No question provided."}), 400
        system_prompt = "You are 'Agro Assistant', a friendly and helpful AI chatbot for a web application designed for farmers..." # Shortened for brevity
        gemini_payload = {"contents": [{"parts": [{"text": system_prompt}, {"text": f"User's question: {user_question}"}]}]}
        response = requests.post(GEMINI_API_URL, json=gemini_payload, timeout=45)
        response.raise_for_status()
        result_text = response.json()['candidates'][0]['content']['parts'][0]['text']
        return jsonify({"answer": result_text})
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app.route("/upload-item-image", methods=["POST"])
def upload_item_image():
    """Handles image uploads for marketplace items to Cloudinary."""
    if 'item_image' not in request.files: return jsonify({"error": "No 'item_image' file part"}), 400
    file_to_upload = request.files['item_image']
    if file_to_upload.filename == '': return jsonify({"error": "No file selected"}), 400
    try:
        upload_result = cloudinary.uploader.upload(file_to_upload, folder="agri_assistant_items")
        return jsonify({"imageUrl": upload_result.get('secure_url')})
    except Exception as e:
        return jsonify({"error": f"Failed to upload image: {e}"}), 500

@app.route('/add-item', methods=['POST'])
def add_item():
    """Adds a new product item to the Firestore database."""
    if not db: return jsonify({"error": "Database not initialized"}), 500
    try:
        db.collection('products').add(request.get_json())
        return jsonify({"success": True, "message": "Item added successfully"}), 201
    except Exception as e:
        return jsonify({"error": f"Failed to add item: {e}"}), 500

@app.route('/get-items', methods=['GET'])
def get_items():
    """Retrieves all product items from the Firestore database."""
    if not db: return jsonify({"error": "Database not initialized"}), 500
    try:
        products_ref = db.collection('products').stream()
        products_list = [doc.to_dict() for doc in products_ref]
        return jsonify(products_list)
    except Exception as e:
        return jsonify({"error": f"Failed to get items: {e}"}), 500

@app.route("/agri-news", methods=["GET"])
def agri_news():
    """Fetches agricultural news from News API."""
    search_query = '("agriculture" OR "farming" OR "crops" OR "horticulture") AND "india"'
    url = f"https://newsapi.org/v2/everything?q={search_query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        articles = [a for a in response.json().get("articles", []) if a.get("title") != "[Removed]"]
        return jsonify({"articles": articles[:20]})
    except Exception as e:
        return jsonify({"error": f"Could not retrieve news: {e}"}), 502

@app.route("/weather", methods=["GET"])
def weather():
    """Fetches current weather data from OpenWeatherMap."""
    lat, lon, city = request.args.get("lat"), request.args.get("lon"), request.args.get("city")
    if city: url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={OPENWEATHER_API_KEY}"
    elif lat and lon: url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={OPENWEATHER_API_KEY}"
    else: return jsonify({"error": "City or lat/lon required"}), 400
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return jsonify({ "city": data.get("name"), "temp": data.get("main", {}).get("temp"), "condition": data.get("weather", [{}])[0].get("description", "").title(), "lat": data.get("coord", {}).get("lat"), "lon": data.get("coord", {}).get("lon")})
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 502

@app.route("/nearby-weather", methods=["GET"])
def nearby_weather():
    """Fetches weather for a predefined list of nearby major cities."""
    nearby_cities, weather_data = ["Coimbatore", "Tiruppur", "Erode", "Salem", "Pollachi"], []
    for city in nearby_cities:
        try:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={OPENWEATHER_API_KEY}"
            data = requests.get(url).json()
            weather_data.append({"city": data.get("name"), "temp": data.get("main", {}).get("temp"), "condition": data.get("weather", [{}])[0].get("description", "").title()})
        except Exception as e:
            print(f"Could not fetch weather for {city}: {e}")
    return jsonify(weather_data)

@app.route("/prices", methods=["GET"])
def prices():
    """Returns mock price data for vegetables."""
    location = request.args.get('location', 'coimbatore').lower()
    veg = request.args.get('vegetable', 'tomato').lower()
    price_data = {'coimbatore': {'tomato': 35, 'potato': 25, 'onion': 30}, 'salem': {'brinjal': 22, 'onion': 28}}
    price = price_data.get(location, {}).get(veg, "N/A")
    return jsonify({"prices": [{"name": veg.title(), "location": location.title(), "price": price}]})

@app.route("/planner", methods=["GET"])
def planner():
    """Generates a farming plan using the Gemini API."""
    crop, area, season = request.args.get("crop"), request.args.get("area"), request.args.get("season")
    if not all([crop, area, season]): return jsonify({"error": "Crop, area, and season are required"}), 400
    prompt = f"""Create a farming plan for '{crop}' on {area} acres during '{season}' season in India near Coimbatore. Your response MUST be a valid JSON object with this exact structure: {{"suggestion": "...", "estimated_cost": "...", "tips": "..."}}"""
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"response_mime_type": "application/json"}}
    try:
        response = requests.post(GEMINI_API_URL, json=payload, timeout=45)
        response.raise_for_status()
        plan_data = json.loads(response.json()['candidates'][0]['content']['parts'][0]['text'])
        return jsonify(plan_data)
    except Exception as e:
        return jsonify({"error": f"Failed to generate plan: {e}"}), 500

# --- 5. RUN THE FLASK APPLICATION ---
if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)