# ----------------------------------
# Import Library yang Dibutuhkan
# ----------------------------------
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
# 1. INISIALISASI & KONFIGURASI APLIKASI
# ----------------------------------
app = Flask(__name__)
CORS(app) # Mengizinkan Cross-Origin Resource Sharing

# --- Konfigurasi Model ---
CLASS_NAMES = ['batik_ceplok', 'batik_kawung', 'batik_nitik', 'batik_parang', 'batik_sidoluhur', 'batik_truntum']
IMG_SIZE = (224, 224)
HUB_MODEL_URL = "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"

# --- Thresholds Autentikasi (NILAI KETAT) ---
TEXTURE_VAR_THRESH = 12000    # Menuntut variasi garis lebih tinggi (khas tulisan tangan)
LINE_LEN_THRESH = 250       # Menuntut garis lebih pendek/terputus (khas canting)
COLOR_GRAD_THRESH = 2200    # Menuntut gradasi warna lebih halus (khas pewarna alami)
COLOR_DIV_THRESH = 25       # Menuntut keragaman warna lebih kaya
MOTIF_CONF_THRESH = 70      # Menuntut keyakinan prediksi model yang tinggi
WAX_DOT_THRESH = 15         # Syarat minimal jumlah tetesan lilin kecil (artefak tulis)

# ----------------------------------
# 2. DEFINISI CLASS MODEL KUSTOM
# ----------------------------------
class BatikLiteModel(tf.keras.Model):
    """Mendefinisikan arsitektur model kustom dengan feature extractor dari TF Hub."""
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
# 3. PEMUATAN MODEL (METODE ANDAL)
# ----------------------------------
model = None

def load_batik_model():
    """
    Memuat model dengan metode "Reconstruct & Load Weights" untuk menghindari error
    pada model Keras subclass yang disimpan dalam format H5.
    """
    global model
    MODEL_H5_PATH = 'batik_lite_model.h5'

    try:
        print(f"Mencoba memuat model: Membuat ulang arsitektur dan memuat weights dari '{MODEL_H5_PATH}'...")
        
        # 1. Buat instance model baru, berikan argumen 'num_classes' secara eksplisit.
        model = BatikLiteModel(num_classes=len(CLASS_NAMES))
        
        # 2. Bangun model dengan input tiruan agar arsitektur terbentuk.
        dummy_input = tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3))
        _ = model(dummy_input)
        
        # 3. Muat HANYA bobot (weights) ke dalam arsitektur yang sudah ada.
        model.load_weights(MODEL_H5_PATH)
        
        print("✅ Model berhasil dibuat ulang dan weights dimuat.")
        return True
    
    except Exception as e:
        print(f"❌ GAGAL MEMUAT MODEL: {e}")
        print("Pastikan file 'batik_lite_model.h5' ada di direktori yang sama dan tidak rusak.")
        model = None
        return False

# Panggil fungsi untuk memuat model saat aplikasi pertama kali dijalankan
load_batik_model()

# ----------------------------------
# 4. FUNGSI-FUNGSI AUTENTIKASI
# ----------------------------------
def analyze_texture(img_pil):
    """Menganalisis ketidakteraturan garis pada gambar."""
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_variance = np.var(edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=30)
    avg_line_length = np.mean([np.sqrt((x2-x1)**2 + (y2-y1)**2) for line in lines for x1, y1, x2, y2 in line]) if lines is not None else 0
    return edge_variance, avg_line_length

def analyze_color(img_pil):
    """Menganalisis gradasi dan keragaman warna."""
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2HSV)
    dx = cv2.Sobel(hsv[:, :, 0], cv2.CV_64F, 1, 0, ksize=5)
    dy = cv2.Sobel(hsv[:, :, 0], cv2.CV_64F, 0, 1, ksize=5)
    gradient = np.mean(np.sqrt(dx**2 + dy**2))
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    color_diversity = np.sum(hist > 0.01 * np.max(hist))
    return gradient, color_diversity

def analyze_wax_dots(img_pil):
    """Menganalisis dan menghitung jumlah titik lilin kecil (artefak batik tulis)."""
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(opening, connectivity=8)
    
    dot_count = 0
    min_dot_area = 2
    max_dot_area = 50
    
    for i in range(1, num_labels): # Mulai dari 1 untuk skip background
        if min_dot_area <= stats[i, cv2.CC_STAT_AREA] <= max_dot_area:
            dot_count += 1
            
    return dot_count

def is_authentic(img_pil, predicted_class_idx, confidence):
    """Menjalankan semua pemeriksaan keaslian dan mengembalikan hasil terperinci."""
    # 1. Analisis Tekstur
    edge_var, avg_line_len = analyze_texture(img_pil)
    is_handmade_texture = (edge_var > TEXTURE_VAR_THRESH) and (avg_line_len < LINE_LEN_THRESH)
    
    # 2. Analisis Pewarnaan
    color_grad, color_div = analyze_color(img_pil)
    has_natural_color = (color_grad < COLOR_GRAD_THRESH) and (color_div >= COLOR_DIV_THRESH)
    
    # 3. Analisis Motif
    is_traditional_motif = (confidence >= MOTIF_CONF_THRESH) and (CLASS_NAMES[predicted_class_idx] in CLASS_NAMES)

    # 4. Analisis Titik Lilin
    wax_dots_count = analyze_wax_dots(img_pil)
    has_wax_dots = wax_dots_count >= WAX_DOT_THRESH

    # Status final (semua kriteria HARUS terpenuhi)
    authentic_status = is_handmade_texture and has_natural_color and is_traditional_motif and has_wax_dots
    
    reasons = []
    if not authentic_status:
        if not is_handmade_texture: reasons.append("Kriteria tekstur buatan tangan tidak terpenuhi (garis terlalu rapi/teratur).")
        if not has_natural_color: reasons.append("Kriteria pewarnaan alami tidak terpenuhi (gradasi warna terlalu tajam atau palet warna terbatas).")
        if not is_traditional_motif: reasons.append("Motif bukan tradisional atau tingkat keyakinan model terlalu rendah.")
        if not has_wax_dots: reasons.append("Tidak terdeteksi artefak titik lilin yang cukup, mengindikasikan proses non-manual.")
    else:
        reasons.append("Memenuhi semua kriteria batik tulis tradisional (tekstur, warna, motif, dan artefak).")

    return {
        'is_authentic': bool(authentic_status),
        'details': {
            'Tekstur (Handmade)': bool(is_handmade_texture),
            'Pewarnaan (Alami)': bool(has_natural_color),
            'Motif (Tradisional)': bool(is_traditional_motif),
            'Artefak (Titik Lilin)': bool(has_wax_dots)
        },
        'debug_values': {
            'texture_variance': f"{edge_var:.2f} (Syarat: >{TEXTURE_VAR_THRESH})",
            'avg_line_length': f"{avg_line_len:.2f} (Syarat: <{LINE_LEN_THRESH})",
            'color_gradient': f"{color_grad:.2f} (Syarat: <{COLOR_GRAD_THRESH})",
            'color_diversity': f"{color_div} (Syarat: >= {COLOR_DIV_THRESH})",
            'motif_confidence': f"{confidence:.2f}% (Syarat: >= {MOTIF_CONF_THRESH}%)",
            'wax_dots_count': f"{wax_dots_count} (Syarat: >= {WAX_DOT_THRESH})"
        },
        'reasons': reasons
    }

# ----------------------------------
# 5. FUNGSI UTAMA UNTUK PROSES API
# ----------------------------------
def process_and_predict(image_bytes):
    """Menerima byte gambar, melakukan prediksi dan autentikasi."""
    img_pil = Image.open(image_bytes).resize(IMG_SIZE)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
        
    img_array = image.img_to_array(img_pil)
    img_array = img_array / 255.0
    img_array_exp = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array_exp)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx] * 100
    
    auth_result = is_authentic(img_pil, predicted_class_idx, confidence)
    
    return {
        'classification': {
            'motif': CLASS_NAMES[predicted_class_idx],
            'confidence': f"{confidence:.2f}"
        },
        'authentication': auth_result
    }

# ----------------------------------
# 6. API ENDPOINTS
# ----------------------------------
@app.route('/predict', methods=['POST'])
def predict_endpoint():
    if model is None:
        return jsonify({'error': 'Model tidak dapat dimuat, server tidak siap.'}), 500
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
    """Endpoint untuk memeriksa status server dan model."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'class_names': CLASS_NAMES
    })

# ----------------------------------
# 7. MENJALANKAN SERVER FLASK
# ----------------------------------
if __name__ == '__main__':
    if model is None:
        print("🔴 FATAL: Model tidak berhasil dimuat. API tidak akan berfungsi dengan benar.")
    else:
        print("🟢 SUCCESS: Model berhasil dimuat. Server API siap menerima permintaan.")
    
    # Jalankan server di semua interface (0.0.0.0) pada port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)