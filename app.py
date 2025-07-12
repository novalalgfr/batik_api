import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from werkzeug.utils import secure_filename

# ----------------------------------
# 1. KONFIGURASI
# ----------------------------------
app = Flask(__name__)
CORS(app)

IMG_SIZE = (224, 224)
CLASS_NAMES = ['batik_ceplok', 'batik_kawung', 'batik_nitik', 'batik_parang', 'batik_sidoluhur', 'batik_truntum']
MODEL_PATH = 'batik_model_jogja_final_2.h5'

# ----------------------------------
# 2. LOAD MODEL
# ----------------------------------
model = None
def load_batik_model():
    global model
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model EfficientNetB0 berhasil dimuat.")
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        model = None

load_batik_model()

# ----------------------------------
# 3. UTILITAS ANALISIS GAMBAR
# ----------------------------------
def analyze_texture(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_variance = np.var(edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=30)
    avg_line_length = np.mean([
        np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        for line in lines for x1, y1, x2, y2 in line
    ]) if lines is not None else 0
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

def analyze_wax_dots(img_pil):
    gray = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(opening, connectivity=8)
    dot_count = sum(1 for i in range(1, num_labels) if 2 <= stats[i, cv2.CC_STAT_AREA] <= 50)
    return dot_count

# ----------------------------------
# 4. THRESHOLD DINAMIS & AUTENTIKASI
# ----------------------------------
def calculate_dynamic_threshold(value, min_val, max_val, mode='mean'):
    if mode == 'scale':
        scale = value / (value + 1e-6)
        return min_val + (max_val - min_val) * scale
    return (min_val + max_val) / 2

def is_authentic(img_pil, predicted_class_idx, confidence):
    # --- Analisis fitur ---
    edge_var, avg_line_len = analyze_texture(img_pil)
    color_grad, color_div = analyze_color(img_pil)
    wax_dots_count = analyze_wax_dots(img_pil)

    # --- Hitung threshold dinamis ---
    texture_thresh    = calculate_dynamic_threshold(edge_var, 5000, 13000)
    color_grad_thresh = calculate_dynamic_threshold(color_grad, 1000, 2500)
    color_div_thresh  = calculate_dynamic_threshold(color_div, 10, 40)
    motif_conf_thresh = calculate_dynamic_threshold(confidence, 40, 75)
    wax_dot_thresh    = calculate_dynamic_threshold(wax_dots_count, 5, 25)

    # --- Evaluasi masing-masing fitur utama ---
    handmade_texture   = (edge_var > texture_thresh) and (avg_line_len < 250)
    natural_color      = (color_grad < color_grad_thresh) and (color_div >= color_div_thresh)
    traditional_motif  = (confidence >= motif_conf_thresh) and (CLASS_NAMES[predicted_class_idx] in CLASS_NAMES)
    wax_dots_present   = wax_dots_count >= wax_dot_thresh

    # --- Skoring total 4 fitur utama ---
    score = sum([
        handmade_texture,
        natural_color,
        traditional_motif,
        wax_dots_present
    ])
    
    # --- Penilaian akhir: semua fitur harus True ---
    is_auth = all([
        handmade_texture,
        natural_color,
        traditional_motif,
        wax_dots_present
    ])

    # --- Penjelasan jika tidak autentik ---
    reasons = []
    if not is_auth:
        if not handmade_texture:
            reasons.append("Tekstur terlalu rapi.")
        if not natural_color:
            reasons.append("Warna tidak alami.")
        if not traditional_motif:
            reasons.append("Motif tidak tradisional atau confidence rendah.")
        if not wax_dots_present:
            reasons.append("Tidak cukup titik lilin.")
    else:
        reasons.append("Memenuhi kriteria batik tulis tradisional.")

    return {
        'is_authentic': bool(is_auth),
        'details': {
            'Tekstur (Handmade)': bool(handmade_texture),
            'Pewarnaan (Alami)': bool(natural_color),
            'Motif (Tradisional)': bool(traditional_motif),
            'Artefak (Titik Lilin)': bool(wax_dots_present)
        },
        'debug_values': {
            'texture_variance': f"{edge_var:.2f} (>{texture_thresh:.2f})",
            'avg_line_length': f"{avg_line_len:.2f} (<250)",
            'color_gradient': f"{color_grad:.2f} (<{color_grad_thresh:.2f})",
            'color_diversity': f"{color_div} (≥{color_div_thresh:.2f})",
            'motif_confidence': f"{confidence:.2f}% (≥{motif_conf_thresh:.2f}%)",
            'wax_dots_count': f"{wax_dots_count} (≥{wax_dot_thresh:.2f})",
            'total_score': f"{score}/4"
        },
        'reasons': reasons
    }

# ----------------------------------
# 5. PROSES PREDIKSI
# ----------------------------------
def process_and_predict_from_path(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array_exp = tf.expand_dims(img_array, 0)
    img_preprocessed = preprocess_input(img_array_exp)

    preds = model.predict(img_preprocessed)
    predicted_idx = np.argmax(preds[0])
    confidence = float(preds[0][predicted_idx]) * 100

    img_pil_for_auth = Image.open(img_path).resize(IMG_SIZE)
    if img_pil_for_auth.mode != "RGB":
        img_pil_for_auth = img_pil_for_auth.convert("RGB")

    auth_result = is_authentic(img_pil_for_auth, predicted_idx, confidence)

    return {
        'classification': {
            'motif': CLASS_NAMES[predicted_idx],
            'confidence': f"{confidence:.2f}"
        },
        'authentication': auth_result
    }

# ----------------------------------
# 6. API ENDPOINTS
# ----------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model belum dimuat'}), 500
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    temp_path = None
    try:
        filename = secure_filename(file.filename)
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        result = process_and_predict_from_path(temp_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'class_names': CLASS_NAMES
    })

# ----------------------------------
# 7. JALANKAN SERVER
# ----------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('SERVER_PORT', 5000))
    app.run(host='0.0.0.0', port=port)
