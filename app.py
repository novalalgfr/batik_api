import os
import io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
# Tambahan untuk mengelola file sementara
from werkzeug.utils import secure_filename

# ----------------------------------
# 1. KONFIGURASI (Tetap sama)
# ----------------------------------
app = Flask(__name__)
CORS(app)

IMG_SIZE = (224, 224)
CLASS_NAMES = ['batik_ceplok', 'batik_kawung', 'batik_nitik', 'batik_parang', 'batik_sidoluhur', 'batik_truntum']
MODEL_PATH = 'batik_model_jogja_final.h5'

# Threshold autentikasi (Tetap sama)
TEXTURE_VAR_THRESH = 12000
LINE_LEN_THRESH = 250
COLOR_GRAD_THRESH = 2200
COLOR_DIV_THRESH = 25
MOTIF_CONF_THRESH = 70
WAX_DOT_THRESH = 15

# ----------------------------------
# 2. LOAD MODEL (Tetap sama)
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
# 3. AUTENTIKASI ANALISIS (Tetap sama)
# ----------------------------------
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

def analyze_wax_dots(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(opening, connectivity=8)
    dot_count = 0
    for i in range(1, num_labels):
        if 2 <= stats[i, cv2.CC_STAT_AREA] <= 50:
            dot_count += 1
    return dot_count

def is_authentic(img_pil, predicted_class_idx, confidence):
    edge_var, avg_line_len = analyze_texture(img_pil)
    color_grad, color_div = analyze_color(img_pil)
    wax_dots_count = analyze_wax_dots(img_pil)

    is_handmade_texture = (edge_var > TEXTURE_VAR_THRESH) and (avg_line_len < LINE_LEN_THRESH)
    has_natural_color = (color_grad < COLOR_GRAD_THRESH) and (color_div >= COLOR_DIV_THRESH)
    is_traditional_motif = (confidence >= MOTIF_CONF_THRESH) and (CLASS_NAMES[predicted_class_idx] in CLASS_NAMES)
    has_wax_dots = wax_dots_count >= WAX_DOT_THRESH

    authentic_status = is_handmade_texture and has_natural_color and is_traditional_motif and has_wax_dots

    reasons = []
    if not authentic_status:
        if not is_handmade_texture:
            reasons.append("Tekstur terlalu rapi.")
        if not has_natural_color:
            reasons.append("Warna tidak alami.")
        if not is_traditional_motif:
            reasons.append("Motif tidak tradisional atau confidence rendah.")
        if not has_wax_dots:
            reasons.append("Tidak cukup titik lilin.")
    else:
        reasons.append("Memenuhi semua kriteria batik tulis tradisional.")

    return {
        'is_authentic': bool(authentic_status),
        'details': {
            'Tekstur (Handmade)': bool(is_handmade_texture),
            'Pewarnaan (Alami)': bool(has_natural_color),
            'Motif (Tradisional)': bool(is_traditional_motif),
            'Artefak (Titik Lilin)': bool(has_wax_dots)
        },
        'debug_values': {
            'texture_variance': f"{edge_var:.2f} (>{TEXTURE_VAR_THRESH})",
            'avg_line_length': f"{avg_line_len:.2f} (<{LINE_LEN_THRESH})",
            'color_gradient': f"{color_grad:.2f} (<{COLOR_GRAD_THRESH})",
            'color_diversity': f"{color_div} (≥{COLOR_DIV_THRESH})",
            'motif_confidence': f"{confidence:.2f}% (≥{MOTIF_CONF_THRESH}%)",
            'wax_dots_count': f"{wax_dots_count} (≥{WAX_DOT_THRESH})"
        },
        'reasons': reasons
    }

# ----------------------------------------------------
# 4. UTAMA: Proses Prediksi (Dimodifikasi)
# ----------------------------------------------------
def process_and_predict_from_path(img_path):
    # --- LANGKAH 1: PREDIKSI MOTIF (Sama seperti di Notebook) ---
    # Memuat dan me-resize gambar menggunakan Keras, bukan Pillow
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array_exp = tf.expand_dims(img_array, 0)
    img_preprocessed = preprocess_input(img_array_exp)

    preds = model.predict(img_preprocessed)
    predicted_idx = np.argmax(preds[0])
    confidence = float(preds[0][predicted_idx]) * 100

    # --- LANGKAH 2: AUTENTIKASI (Menggunakan gambar yang sama) ---
    # Untuk analisis autentikasi, kita tetap butuh objek gambar PIL.
    # Kita muat lagi dengan Pillow untuk memastikan ukurannya sesuai untuk analisis.
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
# 5. API ENDPOINTS (Dimodifikasi)
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
        # --- PERUBAHAN DI SINI ---
        # Simpan file yang di-upload ke direktori sementara
        filename = secure_filename(file.filename)
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True) # Buat direktori jika belum ada
        temp_path = os.path.join(temp_dir, filename)
        file.save(temp_path)

        # Panggil fungsi yang menggunakan file path, bukan bytes
        result = process_and_predict_from_path(temp_path)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Hapus file sementara setelah selesai
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
# 6. JALANKAN SERVER (Tetap sama)
# ----------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('SERVER_PORT', 5000))
    app.run(host='0.0.0.0', port=port)