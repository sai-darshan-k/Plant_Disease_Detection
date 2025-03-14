from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define image size
IMG_SIZE = (256, 256)

# Load the TFLite model
model_path = 'model_quantized.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Log model input details for debugging
print("Input details:", input_details)
print("Output details:", output_details)

# Original class names
original_class_names = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy", "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight",
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy", "Tomato___Bacterial_spot", "Tomato___Early_blight",
    "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

# Disease information dictionary
disease_info = {
    "Apple_scab": {
        "disease_name": "Apple Scab",
        "symptoms": "Dark, velvety spots on leaves and fruit, leaf drop, deformed fruit.",
        "organic_cure": "Apply neem oil, remove infected leaves, improve air circulation.",
        "prevention_tips": "Prune trees for better airflow, avoid overhead watering, use resistant varieties.",
        "seasonal_risk": "Spring and early summer, especially in wet conditions."
    },
    "Black_rot": {
        "disease_name": "Black Rot",
        "symptoms": "Black spots on leaves and fruit, yellowing leaves, fruit rot.",
        "organic_cure": "Use copper-based fungicides, prune affected areas, maintain hygiene.",
        "prevention_tips": "Remove fallen leaves, apply mulch, sanitize tools.",
        "seasonal_risk": "Late spring to summer, humid weather."
    },
    "Cedar_apple_rust": {
        "disease_name": "Cedar Apple Rust",
        "symptoms": "Yellow-orange spots on leaves, galls on twigs, reduced fruit quality.",
        "organic_cure": "Remove nearby cedar trees, apply sulfur spray, prune infected parts.",
        "prevention_tips": "Plant resistant varieties, maintain distance from cedar trees.",
        "seasonal_risk": "Spring, during wet and warm weather."
    },
    "healthy": {
        "disease_name": "Healthy",
        "symptoms": "No visible symptoms, normal growth and appearance.",
        "organic_cure": "Maintain good watering and nutrient practices to keep plants healthy.",
        "prevention_tips": "Regular monitoring, balanced fertilization, proper irrigation.",
        "seasonal_risk": "Not applicable."
    },
    "Powdery_mildew": {
        "disease_name": "Powdery Mildew",
        "symptoms": "White powdery spots on leaves, stunted growth, leaf curling.",
        "organic_cure": "Spray with milk solution (1:9 milk:water), improve air flow, use neem oil.",
        "prevention_tips": "Avoid overcrowding, prune for ventilation, water at the base.",
        "seasonal_risk": "Summer, in warm and dry conditions."
    },
    "Cercospora_leaf_spot Gray_leaf_spot": {
        "disease_name": "Cercospora Leaf Spot/Gray Leaf Spot",
        "symptoms": "Gray-white spots with dark borders on leaves, leaf yellowing.",
        "organic_cure": "Remove infected leaves, apply compost tea, ensure proper spacing.",
        "prevention_tips": "Crop rotation, remove plant debris, use resistant hybrids.",
        "seasonal_risk": "Summer, in warm and humid conditions."
    },
    "Common_rust": {
        "disease_name": "Common Rust",
        "symptoms": "Orange-brown spots on leaves, reduced photosynthesis.",
        "organic_cure": "Use sulfur dust, improve ventilation, remove affected leaves.",
        "prevention_tips": "Plant resistant varieties, avoid dense planting.",
        "seasonal_risk": "Summer, during warm and wet weather."
    },
    "Northern_Leaf_Blight": {
        "disease_name": "Northern Leaf Blight",
        "symptoms": "Long, gray-green lesions on leaves, leaf death.",
        "organic_cure": "Crop rotation, remove debris, apply garlic extract.",
        "prevention_tips": "Use resistant hybrids, space plants properly, avoid overhead irrigation.",
        "seasonal_risk": "Late summer, in cool and wet conditions."
    },
    "Esca_(Black_Measles)": {
        "disease_name": "Esca (Black Measles)",
        "symptoms": "Dark streaks on wood, leaf discoloration, fruit shriveling.",
        "organic_cure": "Prune infected parts, apply biochar, avoid water stress.",
        "prevention_tips": "Maintain plant health, avoid wounding, use organic mulch.",
        "seasonal_risk": "Summer, in warm climates."
    },
    "Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "disease_name": "Leaf Blight (Isariopsis Leaf Spot)",
        "symptoms": "Dark spots on leaves, premature leaf drop.",
        "organic_cure": "Use copper spray, remove infected leaves, improve drainage.",
        "prevention_tips": "Prune for airflow, avoid wet foliage, monitor plant stress.",
        "seasonal_risk": "Late spring to summer, in humid conditions."
    },
    "Haunglongbing_(Citrus_greening)": {
        "disease_name": "Huanglongbing (Citrus Greening)",
        "symptoms": "Yellowing leaves, misshapen fruit, bitter taste.",
        "organic_cure": "No cure; remove infected trees, use resistant varieties, control psyllids with neem.",
        "prevention_tips": "Monitor for psyllids, use healthy planting material, apply organic insecticides.",
        "seasonal_risk": "Year-round in warm climates."
    },
    "Bacterial_spot": {
        "disease_name": "Bacterial Spot",
        "symptoms": "Water-soaked spots turning dark brown/black, leaf drop.",
        "organic_cure": "Apply copper-based sprays, remove affected parts, avoid overhead watering.",
        "prevention_tips": "Use disease-free seeds, rotate crops, maintain dry foliage.",
        "seasonal_risk": "Summer, in warm and wet conditions."
    },
    "Early_blight": {
        "disease_name": "Early Blight",
        "symptoms": "Concentric rings on leaves, yellowing, leaf drop.",
        "organic_cure": "Use baking soda spray, mulch soil, rotate crops.",
        "prevention_tips": "Stake plants, remove lower leaves, avoid wet conditions.",
        "seasonal_risk": "Early summer, in warm and humid weather."
    },
    "Late_blight": {
        "disease_name": "Late Blight",
        "symptoms": "Dark, water-soaked spots on leaves, white mold, rapid decay.",
        "organic_cure": "Apply copper fungicide, remove infected plants, improve air circulation.",
        "prevention_tips": "Use resistant varieties, avoid overhead watering, monitor weather.",
        "seasonal_risk": "Late summer, in cool and wet conditions."
    },
    "Leaf_scorch": {
        "disease_name": "Leaf Scorch",
        "symptoms": "Brown, scorched leaf edges, leaf drop.",
        "organic_cure": "Ensure proper watering, mulch, apply compost tea.",
        "prevention_tips": "Maintain soil moisture, avoid drought stress, use mulch.",
        "seasonal_risk": "Summer, during hot and dry periods."
    },
    "Leaf_Mold": {
        "disease_name": "Leaf Mold",
        "symptoms": "Yellowing leaves with grayish-white mold underneath.",
        "organic_cure": "Increase ventilation, apply sulfur spray, remove affected leaves.",
        "prevention_tips": "Prune for airflow, avoid high humidity, stake plants.",
        "seasonal_risk": "Summer, in warm and humid conditions."
    },
    "Septoria_leaf_spot": {
        "disease_name": "Septoria Leaf Spot",
        "symptoms": "Small, water-soaked spots turning gray with black dots.",
        "organic_cure": "Remove infected leaves, use neem oil, avoid wet foliage.",
        "prevention_tips": "Rotate crops, mulch soil, water at the base.",
        "seasonal_risk": "Late summer, in wet conditions."
    },
    "Spider_mites Two-spotted_spider_mite": {
        "disease_name": "Spider Mites (Two-spotted)",
        "symptoms": "Tiny yellow spots on leaves, webbing, leaf drop.",
        "organic_cure": "Spray with water, use insecticidal soap, introduce predatory mites.",
        "prevention_tips": "Maintain humidity, monitor plants, avoid dust buildup.",
        "seasonal_risk": "Summer, in hot and dry conditions."
    },
    "Target_Spot": {
        "disease_name": "Target Spot",
        "symptoms": "Concentric rings on leaves, yellowing, defoliation.",
        "organic_cure": "Remove debris, apply compost tea, improve spacing.",
        "prevention_tips": "Stake plants, avoid wet leaves, remove lower foliage.",
        "seasonal_risk": "Summer, in warm and humid weather."
    },
    "Tomato_Yellow_Leaf_Curl_Virus": {
        "disease_name": "Yellow Leaf Curl Virus",
        "symptoms": "Yellowing, curling leaves, stunted growth.",
        "organic_cure": "Remove infected plants, control whiteflies with neem, use reflective mulch.",
        "prevention_tips": "Use row covers, monitor whiteflies, plant resistant varieties.",
        "seasonal_risk": "Summer, in warm climates."
    },
    "Tomato_mosaic_virus": {
        "disease_name": "Mosaic Virus",
        "symptoms": "Mottled leaves, yellowing, distorted growth.",
        "organic_cure": "Remove infected plants, disinfect tools, avoid handling wet plants.",
        "prevention_tips": "Use virus-free seeds, avoid tobacco use near plants, rotate crops.",
        "seasonal_risk": "Year-round, higher in warm weather."
    }
}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_modified_class_name(original_name):
    """Strip crop prefix from class name"""
    return original_name.split("___")[-1].replace(",_", "_")

def preprocess_image(image_file):
    """Preprocess image for prediction"""
    try:
        img = Image.open(image_file).convert('RGB')  # Ensure RGB format
        img = img.resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32)  # Explicitly set to float32
        img_array = np.expand_dims(img_array, 0)     # Add batch dimension
        img_array = img_array / 255.0                # Rescale to [0, 1]
        return img_array
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

def predict_disease(image_file):
    """Make prediction using TFLite model"""
    try:
        input_data = preprocess_image(image_file)
        
        # Debug input shape and type
        print("Input shape:", input_data.shape)
        print("Input dtype:", input_data.dtype)
        
        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Debug raw output
        print("Raw output:", output_data[0])
        
        # Get prediction
        predicted_class = np.argmax(output_data[0])
        confidence = float(np.max(output_data[0]))  # Ensure float for JSON serialization
        
        # Get disease details
        original_predicted_name = original_class_names[predicted_class]
        modified_predicted_name = get_modified_class_name(original_predicted_name)
        disease_details = disease_info.get(modified_predicted_name, {})
        
        return {
            "disease_name": disease_details.get("disease_name", "Unknown"),
            "confidence": confidence,
            "symptoms": disease_details.get("symptoms", "No information available"),
            "organic_cure": disease_details.get("organic_cure", "No information available"),
            "prevention_tips": disease_details.get("prevention_tips", "No information available"),
            "seasonal_risk": disease_details.get("seasonal_risk", "No information available")
        }
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Predict disease
            result = predict_disease(file_path)
            return jsonify({
                "success": True,
                "filename": filename,
                "result": result
            })
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": "An unexpected error occurred"}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
    
    return jsonify({"error": "File type not allowed"}), 400

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)