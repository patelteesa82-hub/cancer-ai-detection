# Sample test cases for Cancer AI Detection

TEST_PATIENTS = [
    {
        'name': 'Patient 1 - High Risk Lung Cancer',
        'data': {
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
        },
        'expected_output': 'HIGH RISK - Lung cancer probable'
    },
    {
        'name': 'Patient 2 - High Risk Colorectal Cancer',
        'data': {
            'biomarkers': {
                'ctDNA_positive': 1,
                'EGFR_mutation': 0,
                'KRAS_mutation': 1,
                'APC_mutation': 1,
                'p53_mutation': 0,
                'CEA_level': 5.8,
                'CYFRA_level': 2.1
            },
            'clinical_data': {
                'age': 58,
                'smoking_status': 0,
                'family_history': 1,
                'respiratory_symptoms': 0,
                'GI_symptoms': 1
            }
        },
        'expected_output': 'HIGH RISK - Colorectal cancer probable'
    },
    {
        'name': 'Patient 3 - Moderate Risk',
        'data': {
            'biomarkers': {
                'ctDNA_positive': 1,
                'EGFR_mutation': 0,
                'KRAS_mutation': 0,
                'APC_mutation': 0,
                'p53_mutation': 0,
                'CEA_level': 3.2,
                'CYFRA_level': 3.5
            },
            'clinical_data': {
                'age': 55,
                'smoking_status': 0,
                'family_history': 0,
                'respiratory_symptoms': 0,
                'GI_symptoms': 0
            }
        },
        'expected_output': 'MODERATE RISK - Repeat test needed'
    },
    {
        'name': 'Patient 4 - Low Risk',
        'data': {
            'biomarkers': {
                'ctDNA_positive': 0,
                'EGFR_mutation': 0,
                'KRAS_mutation': 0,
                'APC_mutation': 0,
                'p53_mutation': 0,
                'CEA_level': 1.5,
                'CYFRA_level': 1.8
            },
            'clinical_data': {
                'age': 45,
                'smoking_status': 0,
                'family_history': 0,
                'respiratory_symptoms': 0,
                'GI_symptoms': 0
            }
        },
        'expected_output': 'LOW RISK - Routine screening'
    },
    {
        'name': 'Patient 5 - Early Signal (Lung)',
        'data': {
            'biomarkers': {
                'ctDNA_positive': 0,
                'EGFR_mutation': 1,
                'KRAS_mutation': 0,
                'APC_mutation': 0,
                'p53_mutation': 0,
                'CEA_level': 2.0,
                'CYFRA_level': 2.9
            },
            'clinical_data': {
                'age': 60,
                'smoking_status': 1,
                'family_history': 0,
                'respiratory_symptoms': 0,
                'GI_symptoms': 0
            }
        },
        'expected_output': 'MODERATE RISK - Monitor symptoms'
    }
]

if __name__ == '__main__':
    from ai_model import CancerAIModel
    import json
    
    print("\n" + "="*80)
    print("CANCER AI DETECTION - TEST SUITE")
    print("="*80)
    
    model = CancerAIModel()
    
    for i, patient in enumerate(TEST_PATIENTS, 1):
        print(f"\n{'='*80}")
        print(f"Test {i}: {patient['name']}")
        print(f"{'='*80}")
        
        result = model.predict(patient['data'])
        
        print("\nInput - Biomarkers:")
        for key, value in patient['data']['biomarkers'].items():
            print(f"  {key}: {value}")
        
        print("\nInput - Clinical Data:")
        for key, value in patient['data']['clinical_data'].items():
            print(f"  {key}: {value}")
        
        print("\nAI Output:")
        print(json.dumps(result, indent=2))
        
        print(f"\nExpected: {patient['expected_output']}")
        print(f"Actual: {result['risk_category']} - {result['primary_organ']} ({result['organ_probabilities'][result['primary_organ'].lower()]}%)")
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
