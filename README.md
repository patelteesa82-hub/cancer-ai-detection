# Cancer AI Detection System 🔬

**Blood-based Biomarker Analysis + AI Risk Prediction for Early Cancer Detection**

---

## 📋 Overview

This system combines:
- **Biomarker Detection**: ctDNA, EGFR, KRAS, APC, CEA, CYFRA 21-1
- **Clinical Data**: Age, smoking status, family history, symptoms
- **AI Risk Engine**: Predicts cancer risk and recommends follow-up

**Focuses on:**
- 🫁 Lung Cancer
- 🔴 Colorectal Cancer

---

## 🎯 Features

✅ **AI Model Training** - Gradient Boosting on biomarker patterns
✅ **6 Decision Rules** - Clinical decision support logic
✅ **Organ-Specific Scoring** - Lung vs Colorectal probability
✅ **Web Interface** - Easy-to-use browser app
✅ **Command Line** - Python script for automation
✅ **Risk Stratification** - LOW / MODERATE / HIGH risk categories
✅ **Clinical Recommendations** - Imaging and follow-up suggestions

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

```bash
# Clone repository
git clone https://github.com/patelteesa82-hub/cancer-ai-detection.git
cd cancer-ai-detection

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1: Web Application (EASIEST)

```bash
python web_app.py
```

Then open: **http://localhost:5000**

Fill in the form and get instant predictions! 🎯

### Option 2: Python Script (Command Line)

```bash
python ai_model.py
```

Example output:
```json
{
  "risk_category": "HIGH RISK",
  "cancer_activity_high": true,
  "primary_organ": "LUNG",
  "organ_probabilities": {
    "lung": 78.2,
    "colorectal": 12.1,
    "other": 9.7
  },
  "recommendation": "IMMEDIATE Low-Dose CT Scan (LDCT) recommended",
  "follow_up": "Immediate (within 1 week)",
  "confidence": 85.0
}
```

---

## 📊 Input Format

### Biomarkers (Blood Test Results)

```python
biomarkers = {
    'ctDNA_positive': 1,        # 0=negative, 1=positive
    'EGFR_mutation': 1,         # 0=none, 1=mutation detected
    'KRAS_mutation': 0,         # 0=none, 1=mutation detected
    'APC_mutation': 1,          # 0=none, 1=mutation detected
    'p53_mutation': 0,          # 0=none, 1=mutation detected
    'CEA_level': 5.2,           # ng/mL (normal <3)
    'CYFRA_level': 4.1          # ng/mL (normal <3)
}
```

### Clinical Data

```python
clinical_data = {
    'age': 65,                  # Age in years
    'smoking_status': 1,        # 0=never, 1=former/current
    'family_history': 0,        # 0=no, 1=yes
    'respiratory_symptoms': 1,  # 0=no, 1=yes (cough, shortness of breath)
    'GI_symptoms': 0            # 0=no, 1=yes (bleeding, pain)
}
```

---

## 🧠 AI Decision Rules

### Rule 1: HIGH RISK ⚠️
```
IF ctDNA positive AND (EGFR OR KRAS OR APC OR p53 mutation)
→ HIGH RISK
→ Action: Organ imaging recommended
```

### Rule 2: MODERATE RISK 🟨
```
IF ctDNA positive BUT no driver mutation AND protein markers borderline
→ MODERATE RISK  
→ Action: Repeat test 3-6 months
```

### Rule 3: EARLY SIGNAL 🟡
```
IF organ-related mutation BUT ctDNA negative
→ MODERATE RISK
→ Action: Repeat blood test, monitor symptoms
```

### Rule 4: CLINICAL HIGH RISK 🟠
```
IF molecular markers negative BUT high clinical risk (heavy smoker, strong family history)
→ UNCERTAIN
→ Action: Imaging or specialist evaluation
```

### Rule 5: LOW RISK ✅
```
IF ctDNA negative AND no mutations AND normal proteins AND no clinical risk factors
→ LOW RISK
→ Action: Routine follow-up screening (annual)
```

### Rule 6: SYMPTOM OVERRIDE 🚨
```
IF severe/progressive symptoms appear (persistent cough, bleeding, weight loss)
→ IMAGING OVERRIDES BLOOD RESULT
→ Action: Immediate diagnostic imaging
```

---

## 📈 Risk Categories & Recommendations

| Risk Level | Action | Timeline |
|-----------|--------|----------|
| **HIGH RISK** | Organ imaging (CT/Colonoscopy) | Immediate (1 week) |
| **MODERATE RISK** | Repeat biomarker test | 3-6 months |
| **UNCERTAIN** | Imaging guided by symptoms | 1-3 months |
| **LOW RISK** | Routine screening | Annual |

---

## 🔬 Biomarker Rationale

### General Cancer Signal
- **ctDNA**: Circulating tumor DNA (broad cancer indicator)
- **TP53**: Mutation in p53 gene (common in many cancers)

### Lung Cancer Specific
- **EGFR**: Epidermal Growth Factor Receptor mutation (lung cancer driver)
- **CYFRA 21-1**: Cytokeratin 19 fragment (lung cancer protein marker)

### Colorectal Cancer Specific
- **APC**: Adenomatous Polyposis Coli (colorectal cancer driver)
- **CEA**: Carcinoembryonic Antigen (colorectal cancer protein marker)

### Both Cancers
- **KRAS**: RAS proto-oncogene (common in both lung and colorectal)

---

## 🛠️ Architecture

```
Input Data (Biomarkers + Clinical)
        ↓
  Feature Engineering
        ↓
  AI Model (Gradient Boosting)
        ↓
  Organ-Specific Scoring
        ↓
  Decision Rule Engine
        ↓
  Risk Classification
        ↓
Clinical Recommendation Output
```

---

## 📊 Model Performance

- **Training Data**: 1000 synthetic patient scenarios
- **Algorithm**: Gradient Boosting Classifier
- **Validation**: Cross-validation on biomarker patterns
- **Target Accuracy**: 70-75% (Phase 1), 85%+ (with real clinical data)

**Note**: Current model uses synthetic data for demonstration. Real accuracy requires validation on actual patient datasets.

---

## ⚠️ Limitations

1. **Synthetic Data**: Demo model uses generated data, not real patients
2. **Clinical Validation Needed**: Must validate on 500+ real patient samples
3. **Not Diagnostic**: AI is decision support, not final diagnosis
4. **Biomarker Variability**: Accuracy improves with larger datasets
5. **Hardware Integration**: Actual biomarker detection requires cartridge testing

---

## 🚀 Next Steps (Phase 2)

1. **Hospital Partnerships** - Collect real patient blood samples
2. **Clinical Validation** - Validate on 500-1000 patients
3. **Model Retraining** - Improve accuracy with real data
4. **CE Mark Preparation** - Regulatory compliance documentation
5. **Hardware Integration** - Microfluidic cartridge development

---

## 📝 How to Use (Step-by-Step)

### Using Web App:

1. **Start server**: `python web_app.py`
2. **Open browser**: http://localhost:5000
3. **Fill biomarker values**:
   - Select YES/NO for mutations
   - Enter protein levels (CEA, CYFRA)
4. **Fill clinical data**:
   - Age, smoking status, symptoms
5. **Click PREDICT**
6. **Get results**: Risk level + recommendations

### Using Python Script:

```python
from ai_model import CancerAIModel

model = CancerAIModel()

patient = {
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

result = model.predict(patient)
print(result)
```

---

## 📞 Important Notes

⚠️ **DISCLAIMER**: This is a research and development tool. Not for clinical diagnosis.

✅ **USE CASES**:
- Research and development
- Clinical decision support (with doctors)
- Algorithm testing and validation
- Hospital partnership pilots

❌ **NOT FOR**:
- Self-diagnosis
- Medical advice
- Definitive diagnosis
- Standalone clinical use

---

## 📜 Project Status

- ✅ Phase 1: AI Model + Web App (COMPLETE)
- ⏳ Phase 2: Hospital Validation (NEXT)
- ⏳ Phase 3: CE Mark (PLANNED)
- ⏳ Phase 4: Market Launch (FUTURE)

---

## 🤝 Contributing

This is an open-source project. Contributions welcome!

- Report bugs
- Suggest improvements  
- Hospital partnerships
- Clinical data collaboration

---

## 📧 Contact

- GitHub: @patelteesa82-hub
- Project: Cancer AI Detection System

---

## 📄 License

MIT License - See LICENSE file for details

---

**Built with ❤️ for early cancer detection**
