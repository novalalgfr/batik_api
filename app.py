import io
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ----------------------------------
# 1. INISIALISASI & KONFIGURASI
# ----------------------------------
app = Flask(__name__)
CORS(app)

CLASS_NAMES = ['batik_ceplok', 'batik_kawung', 'batik_nitik', 'batik_parang', 'batik_sidoluhur', 'batik_truntum']
IMG_SIZE = (224, 224)
HUB_MODEL_URL = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

# ----------------------------------
# 2. DEFINISI CLASS MODEL (DIPERBAIKI!)
# ----------------------------------
class BatikLiteModel(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.feature_extractor = hub.KerasLayer(
            HUB_MODEL_URL, 
            trainable=False, 
            name='efficientnet_lite0_base'
        )
        self.dropout_layer = tf.keras.layers.Dropout(0.2)
        self.output_layer = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.feature_extractor(inputs)
        x = self.dropout_layer(x)
        x = self.output_layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ----------------------------------
# 3. PEMUATAN MODEL (ALTERNATIF APPROACH)
# ----------------------------------
model = None

def load_batik_model():
    global model
    try:
        MODEL_PATH = 'batik_lite_model.h5'
        
        # Method 1: Coba load dengan custom objects
        custom_objects = {
            'KerasLayer': hub.KerasLayer,
            'BatikLiteModel': BatikLiteModel 
        }
        model = load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
        print(f"Model '{MODEL_PATH}' berhasil dimuat dengan custom objects.")
        return True
        
    except Exception as e1:
        print(f"Method 1 gagal: {e1}")
        
        try:
            # Method 2: Recreate model dan load weights
            print("Mencoba recreate model dan load weights...")
            model = BatikLiteModel(num_classes=len(CLASS_NAMES))
            
            # Build model dengan dummy input
            dummy_input = tf.zeros((1, 224, 224, 3))
            _ = model(dummy_input)
            
            # Load weights
            model.load_weights(MODEL_PATH)
            print("Model berhasil dibuat ulang dan weights dimuat.")
            return True
            
        except Exception as e2:
            print(f"Method 2 gagal: {e2}")
            
            try:
                # Method 3: Load SavedModel format jika tersedia
                model = tf.keras.models.load_model('batik_lite_model_saved', compile=False)
                print("Model berhasil dimuat dari SavedModel format.")
                return True
                
            except Exception as e3:
                print(f"Method 3 gagal: {e3}")
                print("Semua method loading gagal. Model tidak dapat dimuat.")
                return False

# Panggil fungsi load model
load_batik_model()

# ----------------------------------
# 4. FUNGSI AUTENTIKASI (TIDAK BERUBAH)
# ----------------------------------

# Thresholds untuk autentikasi
TEXTURE_VAR_THRESH = 10000
LINE_LEN_THRESH = 300
COLOR_GRAD_THRESH = 2600
COLOR_DIV_THRESH = 20
MOTIF_CONF_THRESH = 50

def analyze_texture(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_variance = np.var(edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=30)
    avg_line_length = np.mean([np.sqrt((x2-x1)**2 + (y2-y1)**2) for line in lines for x1, y1, x2, y2 in line]) if lines is not None else 0
    return edge_variance, avg_line_length

def analyze_color(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    dx = cv2.Sobel(hsv[:, :, 0], cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(hsv[:, :, 0], cv2.CV_64F, 0, 1, ksize=5)
    gradient = np.mean(np.sqrt(dx**2 + dy**2))
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    color_diversity = np.sum(hist > 0.01 * np.max(hist))
    return gradient, color_diversity

def is_authentic(img_pil, predicted_class, confidence, class_names):
    edge_var, avg_line_len = analyze_texture(img_pil)
    color_grad, color_div = analyze_color(img_pil)
    is_handmade = (edge_var > TEXTURE_VAR_THRESH) and (avg_line_len < LINE_LEN_THRESH)
    has_natural_color = (color_grad < COLOR_GRAD_THRESH) and (color_div >= COLOR_DIV_THRESH)
    is_traditional_motif = (confidence >= MOTIF_CONF_THRESH) and (class_names[predicted_class] in CLASS_NAMES)
    authentic_status = is_handmade and has_natural_color and is_traditional_motif
    reasons = []
    if not is_handmade:
        reasons.append("Kriteria tekstur buatan tangan tidak terpenuhi.")
    if not has_natural_color:
        reasons.append("Kriteria pewarnaan alami tidak terpenuhi.")
    if not is_traditional_motif:
        reasons.append("Motif bukan tradisional atau tingkat keyakinan rendah.")
    return {
        'is_authentic': bool(authentic_status),
        'details': {
            'Tekstur (Handmade)': bool(is_handmade),
            'Pewarnaan (Alami)': bool(has_natural_color),
            'Motif (Tradisional)': bool(is_traditional_motif)
        },
        'debug_values': {
            'texture_variance': f"{edge_var:.2f} (Syarat: >{TEXTURE_VAR_THRESH})",
            'avg_line_length': f"{avg_line_len:.2f} (Syarat: <{LINE_LEN_THRESH})",
            'color_gradient': f"{color_grad:.2f} (Syarat: <{COLOR_GRAD_THRESH})",
            'color_diversity': f"{color_div} (Syarat: >= {COLOR_DIV_THRESH})",
            'motif_confidence': f"{confidence:.2f}% (Syarat: >= {MOTIF_CONF_THRESH}%)"
        },
        'reasons': reasons if not authentic_status else ["Memenuhi semua kriteria batik tulis tradisional."]
    }

# ----------------------------------
# 5. FUNGSI UTAMA UNTUK API
# ----------------------------------
def process_and_predict(image_bytes):
    # Buka gambar dari byte stream menggunakan Pillow
    img_pil = Image.open(image_bytes).resize(IMG_SIZE)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
        
    # Konversi ke array numpy untuk model
    img_array = image.img_to_array(img_pil)
    img_array = img_array / 255.0  # Normalisasi ke [0,1]
    img_array_exp = tf.expand_dims(img_array, 0)
    
    # Prediksi
    predictions = model.predict(img_array_exp)
    
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100

    # Panggil fungsi autentikasi dengan objek gambar PIL
    auth_result = is_authentic(img_pil, predicted_class_idx, confidence, CLASS_NAMES)

    # Gabungkan hasil
    final_result = {
        'classification': {
            'motif': CLASS_NAMES[predicted_class_idx],
            'confidence': f"{confidence:.2f}"
        },
        'authentication': auth_result
    }
    return final_result

# ----------------------------------
# 6. API ENDPOINT
# ----------------------------------
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if model is None:
        return jsonify({'error': 'Model is not loaded'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'Request harus berisi file gambar'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
    try:
        img_bytes = io.BytesIO(file.read())
        results = process_and_predict(img_bytes)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'class_names': CLASS_NAMES
    })

# ----------------------------------
# 7. MENJALANKAN SERVER
# ----------------------------------
if __name__ == '__main__':
    if model is None:
        print("⚠️  WARNING: Model tidak berhasil dimuat. API akan mengembalikan error.")
    else:
        print("✅ Model berhasil dimuat. API siap digunakan.")
    
    app.run(host='0.0.0.0', port=5000, debug=True)