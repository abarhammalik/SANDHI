"""
ARTHROSCAN-NER Multimodal Prediction Web Server.
Pure Python standard library HTTP Server with zero external web framework dependencies.
Serves:
  - Rich interactive HTML5/CSS3/JavaScript clinical diagnostic interface
  - REST API for live inference, risk assessment, and real sample loading
"""
import os
import sys
import json
import time
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import joblib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATIC_DIR = os.path.join(PROJECT_ROOT, 'interface', 'static')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'early_fusion_best.joblib')
DATASET_PATH = os.path.join(PROJECT_ROOT, 'data', 'features', 'multimodal_fusion_dataset.parquet')

# Exact feature list in training order
IMU_FEATURE_COLS = [
    'imu_stride_time', 'imu_cadence', 'imu_gait_speed', 'imu_stance_time', 'imu_swing_time',
    'imu_stride_time_variability', 'imu_stance_phase_asymmetry', 'imu_stride_asymmetry',
    'imu_acceleration_mean', 'imu_acceleration_std', 'imu_acceleration_rms',
    'imu_angular_velocity_mean', 'imu_angular_velocity_std', 'imu_angular_velocity_rms',
    'imu_peak_acceleration', 'imu_peak_angular_velocity', 'imu_foot_angular_velocity_asymmetry'
]

SEMG_FEATURE_COLS = [
    'semg_vm_rms', 'semg_vl_rms', 'semg_rf_rms', 'semg_bf_rms',
    'semg_vm_median_frequency', 'semg_vl_median_frequency', 'semg_rf_median_frequency', 'semg_bf_median_frequency',
    'semg_vm_vl_rms_ratio', 'semg_vm_vl_activation_asymmetry', 'semg_hq_ratio',
    'semg_vm_peak_rms', 'semg_vl_peak_rms', 'semg_rf_peak_rms', 'semg_bf_peak_rms',
    'semg_vm_rms_area', 'semg_vl_rms_area', 'semg_rf_rms_area', 'semg_bf_rms_area',
    'semg_vm_activation_duration', 'semg_vl_activation_duration', 'semg_rf_activation_duration', 'semg_bf_activation_duration'
]

DEMO_COLS = ['age', 'gender', 'bmi']
META_FLAG_COLS = ['imu_mask', 'semg_mask', 'imu_quality', 'semg_quality']
ALL_FEATURE_COLS = IMU_FEATURE_COLS + SEMG_FEATURE_COLS + DEMO_COLS + META_FLAG_COLS

print(f"Loading multimodal model from: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("Multimodal model loaded successfully.")

print(f"Preloading dataset from: {DATASET_PATH}")
df_full = pd.read_parquet(DATASET_PATH)
test_real = df_full[(df_full['split'] == 'TEST') & (df_full['source_type'] == 'REAL') & (df_full['class_label'].isin([0, 1]))]
synth_paired = df_full[(df_full['source_type'] == 'SYNTHETIC') & (df_full['has_imu'] == 1) & (df_full['has_semg'] == 1)]
print(f"Dataset ready. Real test patients: {len(test_real)}, Paired synthetic: {len(synth_paired)}")

class ArthroScanHandler(BaseHTTPRequestHandler):
    def end_headers_with_cors(self, content_type='application/json'):
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers_with_cors()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ['/', '/index.html']:
            file_path = os.path.join(STATIC_DIR, 'index.html')
            self.serve_file(file_path, 'text/html; charset=utf-8')
        elif path == '/style.css':
            file_path = os.path.join(STATIC_DIR, 'style.css')
            self.serve_file(file_path, 'text/css; charset=utf-8')
        elif path == '/app.js':
            file_path = os.path.join(STATIC_DIR, 'app.js')
            self.serve_file(file_path, 'application/javascript; charset=utf-8')
        elif path == '/api/sample':
            params = urllib.parse.parse_qs(parsed.query)
            sample_type = params.get('type', ['healthy'])[0]
            sample_data = self.get_sample(sample_type)
            self.send_response(200)
            self.end_headers_with_cors()
            self.wfile.write(json.dumps(sample_data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers_with_cors('text/plain')
            self.wfile.write(b"404 Not Found")

    def serve_file(self, path, content_type):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.end_headers_with_cors(content_type)
            self.wfile.write(content)
        else:
            self.send_response(404)
            self.end_headers_with_cors('text/plain')
            self.wfile.write(b"File not found")

    def get_sample(self, sample_type):
        if sample_type == 'healthy_real':
            subset = test_real[test_real['class_label'] == 0]
        elif sample_type == 'koa_real':
            subset = test_real[test_real['class_label'] == 1]
        elif sample_type == 'imu_only':
            subset = test_real[test_real['has_imu'] == 1]
        elif sample_type == 'semg_only':
            subset = test_real[test_real['has_semg'] == 1]
        elif sample_type == 'paired_multimodal':
            subset = synth_paired
        else:
            subset = test_real

        row = subset.sample(1, random_state=int(time.time() * 1000) % 100000).iloc[0]
        out = {
            'patient_id': str(row['patient_id']),
            'true_label': int(row['class_label']),
            'source_type': str(row['source_type']),
            'has_imu': int(row['has_imu']),
            'has_semg': int(row['has_semg']),
            'imu_quality': float(row['imu_quality']) if pd.notna(row['imu_quality']) else 1.0,
            'semg_quality': float(row['semg_quality']) if pd.notna(row['semg_quality']) else 1.0,
            'age': float(row['age']) if pd.notna(row['age']) else 60.0,
            'gender': float(row['gender']) if pd.notna(row['gender']) else 1.0,
            'bmi': float(row['bmi']) if pd.notna(row['bmi']) else 25.0
        }
        for col in IMU_FEATURE_COLS:
            val = row.get(col)
            out[col] = float(val) if pd.notna(val) else None
        for col in SEMG_FEATURE_COLS:
            val = row.get(col)
            out[col] = float(val) if pd.notna(val) else None
        return out

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            post_body = self.rfile.read(content_length)
            try:
                data = json.loads(post_body.decode('utf-8'))
                t0 = time.time()
                
                # Build feature vector
                row_dict = {}
                has_imu = 1.0 if data.get('has_imu', True) else 0.0
                has_semg = 1.0 if data.get('has_semg', True) else 0.0

                row_dict['imu_mask'] = has_imu
                row_dict['semg_mask'] = has_semg
                row_dict['imu_quality'] = float(data.get('imu_quality', 1.0)) if has_imu else 0.0
                row_dict['semg_quality'] = float(data.get('semg_quality', 1.0)) if has_semg else 0.0

                row_dict['age'] = float(data.get('age', 60.0))
                row_dict['gender'] = float(data.get('gender', 1.0))
                row_dict['bmi'] = float(data.get('bmi', 25.0))

                for col in IMU_FEATURE_COLS:
                    val = data.get(col)
                    row_dict[col] = float(val) if (has_imu and val is not None and str(val).strip() != '') else np.nan

                for col in SEMG_FEATURE_COLS:
                    val = data.get(col)
                    row_dict[col] = float(val) if (has_semg and val is not None and str(val).strip() != '') else np.nan

                # Construct 1-row DataFrame in exact order
                df_input = pd.DataFrame([row_dict], columns=ALL_FEATURE_COLS)

                # Inference
                proba = float(model.predict_proba(df_input)[0, 1])
                pred = int(model.predict(df_input)[0])
                latency_ms = round((time.time() - t0) * 1000, 2)

                # Clinical risk classification
                if proba < 0.35:
                    risk_level = "Low Risk (Healthy Gait Pattern)"
                    risk_class = "healthy"
                elif proba < 0.65:
                    risk_level = "Moderate Watchlist / Borderline"
                    risk_class = "moderate"
                else:
                    risk_level = "High Probability Knee Osteoarthritis (KOA)"
                    risk_class = "koa"

                # Explainability Biomarker Indicators
                biomarkers = []
                # Check Cadence
                cadence = row_dict.get('imu_cadence')
                if pd.notna(cadence):
                    if cadence < 95.0:
                        biomarkers.append({'name': 'Reduced Cadence', 'val': f"{cadence:.1f} spm", 'impact': 'high_risk', 'desc': 'Below normal walking frequency (<95 spm)'})
                    else:
                        biomarkers.append({'name': 'Normal Cadence', 'val': f"{cadence:.1f} spm", 'impact': 'protective', 'desc': 'Normal physiological gait tempo'})

                # Check Asymmetry
                asym = row_dict.get('imu_stride_asymmetry')
                if pd.notna(asym):
                    if asym > 0.08:
                        biomarkers.append({'name': 'Gait Asymmetry', 'val': f"{asym*100:.1f}%", 'impact': 'high_risk', 'desc': 'Antalgic gait compensation detected'})
                    else:
                        biomarkers.append({'name': 'Symmetric Gait', 'val': f"{asym*100:.1f}%", 'impact': 'protective', 'desc': 'Balanced bilateral step kinematics'})

                # Check VM/VL Balance
                vm_vl = row_dict.get('semg_vm_vl_rms_ratio')
                if pd.notna(vm_vl):
                    if vm_vl < 0.85:
                        biomarkers.append({'name': 'Vastus Medialis Inhibition', 'val': f"Ratio {vm_vl:.2f}", 'impact': 'high_risk', 'desc': 'VM weakness relative to VL (<0.85)'})
                    else:
                        biomarkers.append({'name': 'Quadriceps Balance', 'val': f"Ratio {vm_vl:.2f}", 'impact': 'protective', 'desc': 'Sufficient VM co-activation'})

                # Check BMI
                bmi = row_dict.get('bmi')
                if bmi and bmi > 28.0:
                    biomarkers.append({'name': 'Elevated BMI', 'val': f"{bmi:.1f} kg/m²", 'impact': 'high_risk', 'desc': 'Increased joint mechanical loading'})

                response_obj = {
                    'status': 'success',
                    'prediction': pred,
                    'probability': proba,
                    'risk_percentage': round(proba * 100, 1),
                    'risk_level': risk_level,
                    'risk_class': risk_class,
                    'latency_ms': latency_ms,
                    'modalities_used': {
                        'has_imu': bool(has_imu),
                        'has_semg': bool(has_semg),
                        'imu_quality': row_dict['imu_quality'],
                        'semg_quality': row_dict['semg_quality']
                    },
                    'biomarkers': biomarkers
                }

                self.send_response(200)
                self.end_headers_with_cors('application/json')
                self.wfile.write(json.dumps(response_obj).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.end_headers_with_cors('application/json')
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers_with_cors('text/plain')
            self.wfile.write(b"404 Not Found")

def run_server(port=8050):
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, ArthroScanHandler)
    print(f"\n==================================================================")
    print(f"ARTHROSCAN WEB DIAGNOSTIC INTERFACE READY!")
    print(f"Server running at: http://127.0.0.1:{port}/")
    print(f"==================================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()

if __name__ == '__main__':
    port = 8050
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
