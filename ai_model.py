import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import json

class CancerAIModel:
    """
    AI Model for Early Cancer Detection using Biomarkers
    Detects: Lung Cancer & Colorectal Cancer
    """
    
    def __init__(self):
        self.model_lung = None
        self.model_colon = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'ctDNA_positive',
            'EGFR_mutation',
            'KRAS_mutation',
            'APC_mutation',
            'p53_mutation',
            'CEA_level',
            'CYFRA_level',
            'age',
            'smoking_status',
            'family_history',
            'respiratory_symptoms',
            'GI_symptoms'
        ]
        self.train_model()
    
    def generate_training_data(self, n_samples=1000):
        """
        Generate synthetic training data for demonstration
        In production: use real patient data from hospitals
        """
        np.random.seed(42)
        
        X = np.random.rand(n_samples, len(self.feature_names))
        
        # Lung cancer logic
        y_lung = (
            (X[:, 0] * 0.4) +  # ctDNA
            (X[:, 1] * 0.3) +  # EGFR
            (X[:, 2] * 0.2) +  # KRAS
            (X[:, 6] * 0.1) +  # CYFRA
            (X[:, 8] * 0.15)   # smoking
        ) > 0.5
        
        # Colorectal cancer logic
        y_colon = (
            (X[:, 0] * 0.4) +  # ctDNA
            (X[:, 3] * 0.35) + # APC
            (X[:, 2] * 0.2) +  # KRAS
            (X[:, 5] * 0.1) +  # CEA
            (X[:, 10] * 0.1)   # GI symptoms
        ) > 0.5
        
        return X, y_lung.astype(int), y_colon.astype(int)
    
    def train_model(self):
        """
        Train AI models for lung and colorectal cancer detection
        """
        print("[TRAINING] Generating synthetic data...")
        X, y_lung, y_colon = self.generate_training_data(n_samples=1000)
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        print("[TRAINING] Training Lung Cancer Model...")
        self.model_lung = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model_lung.fit(X_scaled, y_lung)
        
        print("[TRAINING] Training Colorectal Cancer Model...")
        self.model_colon = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model_colon.fit(X_scaled, y_colon)
        
        print("[TRAINING] Models trained successfully!")
    
    def apply_decision_rules(self, biomarkers, clinical_data):
        """
        Apply 6 Decision Rules from the clinical protocol
        """
        ctDNA = biomarkers['ctDNA_positive']
        EGFR = biomarkers['EGFR_mutation']
        KRAS = biomarkers['KRAS_mutation']
        APC = biomarkers['APC_mutation']
        p53 = biomarkers['p53_mutation']
        CEA = biomarkers['CEA_level']
        CYFRA = biomarkers['CYFRA_level']
        
        # RULE 1: HIGH RISK
        if ctDNA and (EGFR or KRAS or APC or p53):
            return "HIGH RISK", 0.85
        
        # RULE 2: MODERATE/UNCERTAIN RISK
        if ctDNA and not (EGFR or KRAS or APC or p53) and (CEA > 3 or CYFRA > 3):
            return "MODERATE RISK", 0.6
        
        # RULE 3: POSSIBLE EARLY SIGNAL
        if (EGFR or KRAS or APC) and not ctDNA:
            return "MODERATE RISK", 0.55
        
        # RULE 4: CLINICAL HIGH RISK
        if not (EGFR or KRAS or APC or p53 or ctDNA):
            if clinical_data.get('smoking_status', 0) > 0.7 or \
               clinical_data.get('family_history', 0) > 0.7:
                return "UNCERTAIN", 0.5
        
        # RULE 5: LOW RISK
        if not ctDNA and not (EGFR or KRAS or APC or p53) and \
           CEA < 2 and CYFRA < 2:
            return "LOW RISK", 0.15
        
        return "MODERATE RISK", 0.5
    
    def predict(self, patient_data):
        """
        Make prediction for a patient
        
        INPUT FORMAT:
        {
            'biomarkers': {
                'ctDNA_positive': 1 (yes=1, no=0),
                'EGFR_mutation': 1,
                'KRAS_mutation': 0,
                'APC_mutation': 1,
                'p53_mutation': 0,
                'CEA_level': 5.2,  (normal <3)
                'CYFRA_level': 4.1  (normal <3)
            },
            'clinical_data': {
                'age': 65,
                'smoking_status': 1,  (yes=1, no=0)
                'family_history': 0,
                'respiratory_symptoms': 1,
                'GI_symptoms': 0
            }
        }
        """
        biomarkers = patient_data['biomarkers']
        clinical = patient_data['clinical_data']
        
        # Prepare feature vector
        features = np.array([
            biomarkers['ctDNA_positive'],
            biomarkers['EGFR_mutation'],
            biomarkers['KRAS_mutation'],
            biomarkers['APC_mutation'],
            biomarkers['p53_mutation'],
            biomarkers['CEA_level'],
            biomarkers['CYFRA_level'],
            clinical['age'] / 100,  # normalize
            clinical['smoking_status'],
            clinical['family_history'],
            clinical['respiratory_symptoms'],
            clinical['GI_symptoms']
        ]).reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get probabilities
        lung_prob = self.model_lung.predict_proba(features_scaled)[0][1]
        colon_prob = self.model_colon.predict_proba(features_scaled)[0][1]
        
        # Apply decision rules
        risk_category, base_confidence = self.apply_decision_rules(biomarkers, clinical)
        
        # Calculate organ scores based on biomarkers
        lung_score = self.calculate_lung_score(biomarkers, clinical)
        colon_score = self.calculate_colon_score(biomarkers, clinical)
        
        # Determine primary organ
        if lung_score > colon_score:
            primary_organ = "LUNG"
            primary_prob = min(lung_prob * 100, 99)
            secondary_prob = min(colon_prob * 100, 99)
        else:
            primary_organ = "COLORECTAL"
            primary_prob = min(colon_prob * 100, 99)
            secondary_prob = min(lung_prob * 100, 99)
        
        # Generate recommendation
        recommendation = self.get_recommendation(risk_category, primary_organ, primary_prob)
        
        return {
            'risk_category': risk_category,
            'cancer_activity_high': lung_prob > 0.5 or colon_prob > 0.5,
            'primary_organ': primary_organ,
            'organ_probabilities': {
                'lung': round(min(lung_prob * 100, 99), 1),
                'colorectal': round(min(colon_prob * 100, 99), 1),
                'other': round(100 - min(lung_prob * 100, 99) - min(colon_prob * 100, 99), 1)
            },
            'recommendation': recommendation,
            'follow_up': self.get_followup(risk_category),
            'confidence': round(base_confidence * 100, 1)
        }
    
    def calculate_lung_score(self, biomarkers, clinical):
        """Calculate lung cancer score"""
        score = 0
        if biomarkers['EGFR_mutation']:
            score += 0.3
        if biomarkers['KRAS_mutation']:
            score += 0.2
        if biomarkers['CYFRA_level'] > 3:
            score += 0.2
        if clinical['smoking_status']:
            score += 0.15
        if clinical['respiratory_symptoms']:
            score += 0.15
        return min(score, 1.0)
    
    def calculate_colon_score(self, biomarkers, clinical):
        """Calculate colorectal cancer score"""
        score = 0
        if biomarkers['APC_mutation']:
            score += 0.35
        if biomarkers['KRAS_mutation']:
            score += 0.2
        if biomarkers['CEA_level'] > 3:
            score += 0.2
        if clinical['family_history']:
            score += 0.15
        if clinical['GI_symptoms']:
            score += 0.1
        return min(score, 1.0)
    
    def get_recommendation(self, risk_category, primary_organ, probability):
        """Get clinical recommendation based on risk"""
        if risk_category == "HIGH RISK":
            if primary_organ == "LUNG":
                return "IMMEDIATE Low-Dose CT Scan (LDCT) recommended"
            else:
                return "IMMEDIATE Colonoscopy recommended"
        elif risk_category == "MODERATE RISK":
            return f"Repeat biomarker test in 3-6 months + Clinical correlation required"
        elif risk_category == "UNCERTAIN":
            if primary_organ == "LUNG":
                return "Imaging guided by symptoms (CT if respiratory symptoms present)"
            else:
                return "Imaging guided by symptoms (Colonoscopy if GI symptoms present)"
        else:  # LOW RISK
            return "Routine follow-up screening (annual check)"
    
    def get_followup(self, risk_category):
        """Get follow-up timeline"""
        followup_map = {
            "HIGH RISK": "Immediate (within 1 week)",
            "MODERATE RISK": "3-6 months",
            "UNCERTAIN": "1-3 months with symptoms monitoring",
            "LOW RISK": "Annual screening"
        }
        return followup_map.get(risk_category, "3-6 months")


if __name__ == "__main__":
    # Example usage
    model = CancerAIModel()
    
    # Test patient example
    test_patient = {
        'biomarkers': {
            'ctDNA_positive': 1,
            'EGFR_mutation': 1,
            'KRAS_mutation': 0,
            'APC_mutation': 0,
            'p53_mutation': 0,
            'CEA_level': 2.5,
            'CYFRA_level': 4.5
        },
        'clinical_data': {
            'age': 62,
            'smoking_status': 1,
            'family_history': 0,
            'respiratory_symptoms': 1,
            'GI_symptoms': 0
        }
    }
    
    result = model.predict(test_patient)
    print("\n" + "="*60)
    print("CANCER AI DETECTION RESULT")
    print("="*60)
    print(json.dumps(result, indent=2))
