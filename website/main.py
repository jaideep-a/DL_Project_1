import os
import uuid
from flask import Flask, render_template, request, jsonify
from .config import Config
from .api_client import get_prediction, check_model_health, ModelAPIError
from .utils import allowed_file, validate_file_size

app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')
app.secret_key = Config.SECRET_KEY

def get_severity_from_confidence(conf):
    if conf > 0.9: return 'severe'
    if conf > 0.7: return 'moderate'
    return 'mild'

def generate_analysis_text(pred, conf, severity):
    if pred == 'NORMAL':
        return 'No signs of pneumonia detected. Lungs appear clear.'
    return f'Signs consistent with pneumonia detected ({severity} confidence: {conf:.1%}). Please consult a healthcare professional for clinical diagnosis.'

def generate_fallback_result(filename):
    return {
        'filename': filename,
        'status': 'error',
        'error': 'Model API unavailable. Please try again later.'
    }

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_api': 'connected' if check_model_health() else 'disconnected'})

@app.route('/api/v1/model-status')
def api_model_status():
    is_healthy = check_model_health()
    return jsonify({
        'status': 'online' if is_healthy else 'offline',
        'message': 'AI model is ready' if is_healthy else 'AI model is warming up'
    })

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'images' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400

        files = request.files.getlist('images')
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400

        upload_id = str(uuid.uuid4())
        results = []
        max_mb = Config.MAX_FILE_SIZE_MB
        
        for file in files:
            if not file or not file.filename:
                continue
            if not allowed_file(file.filename):
                results.append({'filename': file.filename, 'status': 'error', 'error': 'Use JPG or PNG.'})
                continue
            if not validate_file_size(file):
                results.append({'filename': file.filename, 'status': 'error', 'error': f'File > {max_mb}MB.'})
                continue

            try:
                prediction = get_prediction(file)
                pred_result = prediction.get('prediction', 'UNCERTAIN')
                conf_result = prediction.get('confidence', 0)
                severity = get_severity_from_confidence(conf_result)

                results.append({
                    'filename': file.filename,
                    'status': 'success',
                    'prediction': pred_result,
                    'confidence': conf_result,
                    'severity': severity,
                    'processing_time_ms': prediction.get('processing_time_ms', 0),
                    'model_version': prediction.get('model_version', 'v1.0'),
                    'heatmap': prediction.get('heatmap'),
                    'probabilities': prediction.get('probabilities', {}),
                    'source': 'api',
                    'analysis': generate_analysis_text(pred_result, conf_result, severity)
                })
            except Exception as e:
                results.append(generate_fallback_result(file.filename))

        return jsonify({'upload_id': upload_id, 'results': results})

    return render_template('index.html')
