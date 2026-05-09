from flask import Flask, render_template, request, jsonify
from ai_model import CancerAIModel
import json

app = Flask(__name__)
model = CancerAIModel()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate input
        if not data or 'biomarkers' not in data or 'clinical_data' not in data:
            return jsonify({'error': 'Invalid input format'}), 400
        
        # Convert string inputs to appropriate types
        biomarkers = {
            'ctDNA_positive': int(data['biomarkers'].get('ctDNA_positive', 0)),
            'EGFR_mutation': int(data['biomarkers'].get('EGFR_mutation', 0)),
            'KRAS_mutation': int(data['biomarkers'].get('KRAS_mutation', 0)),
            'APC_mutation': int(data['biomarkers'].get('APC_mutation', 0)),
            'p53_mutation': int(data['biomarkers'].get('p53_mutation', 0)),
            'CEA_level': float(data['biomarkers'].get('CEA_level', 0)),
            'CYFRA_level': float(data['biomarkers'].get('CYFRA_level', 0))
        }
        
        clinical_data = {
            'age': int(data['clinical_data'].get('age', 50)),
            'smoking_status': int(data['clinical_data'].get('smoking_status', 0)),
            'family_history': int(data['clinical_data'].get('family_history', 0)),
            'respiratory_symptoms': int(data['clinical_data'].get('respiratory_symptoms', 0)),
            'GI_symptoms': int(data['clinical_data'].get('GI_symptoms', 0))
        }
        
        patient_data = {
            'biomarkers': biomarkers,
            'clinical_data': clinical_data
        }
        
        result = model.predict(patient_data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def info():
    return jsonify({
        'system': 'Cancer AI Detection System',
        'version': '1.0',
        'cancers_detected': ['Lung Cancer', 'Colorectal Cancer'],
        'biomarkers': [
            'ctDNA (Circulating Tumor DNA)',
            'EGFR mutation',
            'KRAS mutation',
            'APC mutation',
            'p53 mutation',
            'CEA protein level',
            'CYFRA 21-1 protein level'
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
