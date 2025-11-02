# import pandas as pd
# import numpy as np
# import joblib
# import warnings
# from datetime import datetime
# from typing import Dict, List, Tuple, Union
# import os

# warnings.filterwarnings('ignore')

# class CHDRiskPredictor:
#     """
#     Coronary Heart Disease Risk Prediction Pipeline
    
#     This class provides a complete pipeline for predicting CHD risk with:
#     - Risk categorization (Low, Moderate, High)
#     - Confidence scoring
#     - Input validation and preprocessing
#     - Detailed prediction reports
#     """
    
#     def __init__(self, model_path: str = 'trained_models/best_model_logisticregression.pkl'):
#         """
#         Initialize the CHD Risk Predictor
        
#         Args:
#             model_path (str): Path to the trained model pickle file
#         """
#         self.model_path = model_path
#         self.model_package = None
#         self.risk_thresholds = {
#             'low': (0.0, 0.3),
#             'moderate': (0.3, 0.7),
#             'high': (0.7, 1.0)
#         }
#         self.load_model()
    
#     def load_model(self):
#         """Load the trained model and preprocessing components"""
#         try:
#             self.model_package = joblib.load(self.model_path)
#             print(f"✓ Model loaded successfully: {self.model_package['model_name']}")
#             print(f"✓ Training date: {self.model_package['training_date']}")
#             print(f"✓ Model F1-Score: {self.model_package['performance_metrics']['f1']:.4f}")
#             print(f"✓ Required features: {len(self.model_package['selected_features'])}")
#         except FileNotFoundError:
#             raise FileNotFoundError(f"Model file not found: {self.model_path}")
#         except Exception as e:
#             raise Exception(f"Error loading model: {str(e)}")
    
#     def validate_input(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
#         """
#         Validate and preprocess input data
        
#         Args:
#             data: Input data as DataFrame or dictionary
            
#         Returns:
#             pd.DataFrame: Validated and preprocessed data
#         """
#         # Convert to DataFrame if dictionary
#         if isinstance(data, dict):
#             data = pd.DataFrame([data])
#         elif isinstance(data, pd.Series):
#             data = pd.DataFrame([data])
        
#         # Check required features
#         required_features = self.model_package['selected_features']
#         missing_features = set(required_features) - set(data.columns)
        
#         if missing_features:
#             raise ValueError(f"Missing required features: {missing_features}")
        
#         # Select only required features
#         data = data[required_features].copy()
        
#         # Handle missing values (fill with median - same as training)
#         for col in data.columns:
#             if data[col].isnull().any():
#                 median_val = data[col].median()
#                 if pd.isna(median_val):  # If all values are NaN, use 0
#                     median_val = 0
#                 data[col].fillna(median_val, inplace=True)
#                 print(f" Filled missing values in {col} with {median_val}")
        
#         return data
    
#     def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
#         """
#         Apply preprocessing steps (scaling) to input data
        
#         Args:
#             data: Validated input DataFrame
            
#         Returns:
#             np.ndarray: Preprocessed data ready for prediction
#         """
#         # Apply scaling if the model requires it (e.g., Logistic Regression)
#         model_name = self.model_package['model_name']
        
#         if model_name == 'LogisticRegression':
#             # Scale the data using the saved scaler
#             scaled_data = self.model_package['scaler'].transform(data)
#             return scaled_data
#         else:
#             # For tree-based models, no scaling needed
#             return data.values
    
#     def categorize_risk(self, probability: float) -> Tuple[str, str, str]:
#         """
#         Categorize risk based on probability thresholds
        
#         Args:
#             probability: Predicted probability (0-1)
            
#         Returns:
#             Tuple: (risk_category, risk_description, recommendation)
#         """
#         if self.risk_thresholds['low'][0] <= probability < self.risk_thresholds['low'][1]:
#             return (
#                 'Low Risk',
#                 'Low probability of developing CHD in the next 10 years',
#                 'Continue healthy lifestyle habits and regular check-ups'
#             )
#         elif self.risk_thresholds['moderate'][0] <= probability < self.risk_thresholds['moderate'][1]:
#             return (
#                 'Moderate Risk',
#                 'Moderate probability of developing CHD in the next 10 years',
#                 'Consider lifestyle modifications and consult healthcare provider'
#             )
#         else:
#             return (
#                 'High Risk',
#                 'High probability of developing CHD in the next 10 years',
#                 'Immediate medical consultation and lifestyle changes recommended'
#             )
    
#     def calculate_confidence(self, probabilities: np.ndarray) -> Tuple[float, str]:
#         """
#         Calculate model confidence based on probability distribution
        
#         Args:
#             probabilities: Array of class probabilities [prob_class_0, prob_class_1]
            
#         Returns:
#             Tuple: (confidence_score, confidence_level)
#         """
#         # Confidence is based on how far the probability is from 0.5 (maximum uncertainty)
#         max_prob = np.max(probabilities)
#         confidence_score = abs(max_prob - 0.5) * 2  # Scale to 0-1
        
#         # Categorize confidence level
#         if confidence_score >= 0.8:
#             confidence_level = 'Very High'
#         elif confidence_score >= 0.6:
#             confidence_level = 'High'
#         elif confidence_score >= 0.4:
#             confidence_level = 'Moderate'
#         elif confidence_score >= 0.2:
#             confidence_level = 'Low'
#         else:
#             confidence_level = 'Very Low'
        
#         return confidence_score, confidence_level
    
#     def get_feature_contributions(self, data: pd.DataFrame, probability: float) -> pd.DataFrame:
#         """
#         Get feature contributions to the prediction (for tree-based models)
        
#         Args:
#             data: Input data DataFrame
#             probability: Predicted probability
            
#         Returns:
#             pd.DataFrame: Feature contributions sorted by importance
#         """
#         model = self.model_package['model']
        
#         # For tree-based models, use feature importances
#         if hasattr(model, 'feature_importances_'):
#             feature_importance = pd.DataFrame({
#                 'Feature': self.model_package['selected_features'],
#                 'Importance': model.feature_importances_,
#                 'Value': data.iloc[0].values
#             }).sort_values('Importance', ascending=False)
            
#             return feature_importance
#         else:
#             # For other models, return basic feature values
#             return pd.DataFrame({
#                 'Feature': self.model_package['selected_features'],
#                 'Value': data.iloc[0].values,
#                 'Importance': [0.1] * len(self.model_package['selected_features'])
#             })
    
#     def predict_single(self, data: Union[pd.DataFrame, Dict]) -> Dict:
#         """
#         Make prediction for a single instance
        
#         Args:
#             data: Input data as DataFrame or dictionary
            
#         Returns:
#             Dict: Comprehensive prediction results
#         """
#         # Validate and preprocess input
#         validated_data = self.validate_input(data)
#         processed_data = self.preprocess_data(validated_data)
        
#         # Make prediction
#         model = self.model_package['model']
#         prediction = model.predict(processed_data)[0]
#         probabilities = model.predict_proba(processed_data)[0]
#         chd_probability = probabilities[1]  # Probability of CHD (class 1)
        
#         # Risk categorization
#         risk_category, risk_description, recommendation = self.categorize_risk(chd_probability)
        
#         # Confidence calculation
#         confidence_score, confidence_level = self.calculate_confidence(probabilities)
        
#         # Feature contributions
#         feature_contributions = self.get_feature_contributions(validated_data, chd_probability)
        
#         # Compile results
#         results = {
#             'prediction': {
#                 'chd_risk_probability': round(chd_probability, 4),
#                 'predicted_class': int(prediction),
#                 'risk_category': risk_category,
#                 'risk_description': risk_description,
#                 'recommendation': recommendation
#             },
#             'confidence': {
#                 'confidence_score': round(confidence_score, 4),
#                 'confidence_level': confidence_level,
#                 'class_probabilities': {
#                     'no_chd': round(probabilities[0], 4),
#                     'chd': round(probabilities[1], 4)
#                 }
#             },
#             'feature_analysis': {
#                 'top_contributing_features': feature_contributions.head(5).to_dict('records'),
#                 'input_values': validated_data.iloc[0].to_dict()
#             },
#             'model_info': {
#                 'model_name': self.model_package['model_name'],
#                 'model_performance': self.model_package['performance_metrics'],
#                 'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             }
#         }
        
#         return results
    
#     def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
#         """
#         Make predictions for multiple instances
        
#         Args:
#             data: Input DataFrame with multiple rows
            
#         Returns:
#             pd.DataFrame: Results for all instances
#         """
#         results = []
        
#         for idx, row in data.iterrows():
#             try:
#                 result = self.predict_single(row.to_dict())
                
#                 # Flatten result for DataFrame
#                 flat_result = {
#                     'id': idx,
#                     'chd_probability': result['prediction']['chd_risk_probability'],
#                     'predicted_class': result['prediction']['predicted_class'],
#                     'risk_category': result['prediction']['risk_category'],
#                     'confidence_score': result['confidence']['confidence_score'],
#                     'confidence_level': result['confidence']['confidence_level'],
#                     'no_chd_prob': result['confidence']['class_probabilities']['no_chd'],
#                     'chd_prob': result['confidence']['class_probabilities']['chd']
#                 }
                
#                 results.append(flat_result)
                
#             except Exception as e:
#                 print(f"Error processing row {idx}: {str(e)}")
#                 continue
        
#         return pd.DataFrame(results)
    
#     def print_prediction_report(self, results: Dict):
#         """
#         Print a formatted prediction report
        
#         Args:
#             results: Prediction results dictionary
#         """
#         print("=" * 60)
#         print("         CHD RISK PREDICTION REPORT")
#         print("=" * 60)
        
#         # Prediction Summary
#         pred = results['prediction']
#         conf = results['confidence']
        
#         print(f"🎯 PREDICTION SUMMARY")
#         print(f"   CHD Risk Probability: {pred['chd_risk_probability']:.1%}")
#         print(f"   Risk Category: {pred['risk_category']}")
#         print(f"   Description: {pred['risk_description']}")
        
#         print(f"\n🔍 MODEL CONFIDENCE")
#         print(f"   Confidence Level: {conf['confidence_level']}")
#         print(f"   Confidence Score: {conf['confidence_score']:.1%}")
#         print(f"   No CHD Probability: {conf['class_probabilities']['no_chd']:.1%}")
#         print(f"   CHD Probability: {conf['class_probabilities']['chd']:.1%}")
        
#         print(f"\n💡 RECOMMENDATION")
#         print(f"   {pred['recommendation']}")
        
#         print(f"\n📊 TOP CONTRIBUTING FACTORS")
#         for i, feature in enumerate(results['feature_analysis']['top_contributing_features'][:3], 1):
#             print(f"   {i}. {feature['Feature']}: {feature['Value']} (Importance: {feature['Importance']:.3f})")
        
#         print(f"\n🔧 MODEL INFORMATION")
#         model_info = results['model_info']
#         print(f"   Model: {model_info['model_name']}")
#         print(f"   F1-Score: {model_info['model_performance']['f1']:.3f}")
#         print(f"   Prediction Date: {model_info['prediction_date']}")
        
#         print("=" * 60)


# def create_sample_data():
#     """Create sample data for testing"""
#     sample_data = {
#         # Low risk profile
#         'low_risk_patient': {
#             'male': 0, 'age': 35, 'education': 3, 'cigsPerDay': 0,
#             'totChol': 180, 'sysBP': 110, 'diaBP': 70, 'BMI': 22,
#             'heartRate': 70, 'glucose': 85
#         },
#         # Moderate risk profile
#         'moderate_risk_patient': {
#             'male': 1, 'age': 50, 'education': 2, 'cigsPerDay': 10,
#             'totChol': 220, 'sysBP': 140, 'diaBP': 90, 'BMI': 28,
#             'heartRate': 80, 'glucose': 100
#         },
#         # High risk profile
#         'high_risk_patient': {
#             'male': 1, 'age': 65, 'education': 1, 'cigsPerDay': 30,
#             'totChol': 280, 'sysBP': 180, 'diaBP': 110, 'BMI': 35,
#             'heartRate': 95, 'glucose': 130
#         }
#     }
#     return sample_data


# # Example usage and testing
# if __name__ == "__main__":
#     # Initialize the predictor
#     try:
#         predictor = CHDRiskPredictor()
        
#         # Create sample data
#         sample_patients = create_sample_data()
        
#         print("\n🧪 TESTING PREDICTION PIPELINE")
#         print("=" * 60)
        
#         # Test single predictions
#         for patient_name, patient_data in sample_patients.items():
#             print(f"\n📋 Analyzing {patient_name.replace('_', ' ').title()}")
            
#             # Make prediction
#             results = predictor.predict_single(patient_data)
            
#             # Print formatted report
#             predictor.print_prediction_report(results)
            
#             input("\nPress Enter to continue to next patient...")
        
#         # Test batch prediction
#         print("\n📊 BATCH PREDICTION EXAMPLE")
#         batch_data = pd.DataFrame([
#             sample_patients['low_risk_patient'],
#             sample_patients['moderate_risk_patient'],
#             sample_patients['high_risk_patient']
#         ])
        
#         batch_results = predictor.predict_batch(batch_data)
#         print("\nBatch Prediction Results:")
#         print(batch_results[['chd_probability', 'risk_category', 'confidence_level']])
        
#         # Save batch results
#         batch_results.to_csv('prediction_results.csv', index=False)
#         print("\n✓ Batch results saved to 'prediction_results.csv'")
        
#     except FileNotFoundError:
#         print("❌ Error: Model file not found!")
#         print("Please ensure you have run the training script first.")
#         print("Expected file: 'trained_models/best_model_randomforest.pkl'")
#     except Exception as e:
#         print(f"❌ Error: {str(e)}")


# # Additional utility functions for easy integration
# def quick_predict(patient_data: Dict, model_path: str = 'trained_models/best_model_randomforest.pkl'):
#     """
#     Quick prediction function for single patients
    
#     Args:
#         patient_data: Dictionary with patient information
#         model_path: Path to model file
        
#     Returns:
#         Dict: Prediction results
#     """
#     predictor = CHDRiskPredictor(model_path)
#     return predictor.predict_single(patient_data)


# def predict_from_csv(csv_path: str, model_path: str = 'trained_models/best_model_randomforest.pkl'):
#     """
#     Predict from CSV file
    
#     Args:
#         csv_path: Path to CSV file with patient data
#         model_path: Path to model file
        
#     Returns:
#         pd.DataFrame: Prediction results
#     """
#     predictor = CHDRiskPredictor(model_path)
#     data = pd.read_csv(csv_path)
#     return predictor.predict_batch(data)














import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from typing import Dict, Tuple, Union

warnings.filterwarnings('ignore')

class CHDRiskPredictor:
    """
    Simplified Coronary Heart Disease Risk Prediction Pipeline for Backend Integration
    
    This class provides a streamlined pipeline for predicting CHD risk with:
    - Risk categorization (Low, Moderate, High)
    - Input validation and preprocessing
    - Simplified prediction reports
    """
    
    def __init__(self, model_path: str = 'trained_models/best_model_logisticregression.pkl'):
        """
        Initialize the CHD Risk Predictor
        
        Args:
            model_path (str): Path to the trained model pickle file
        """
        self.model_path = model_path
        self.model_package = None
        self.risk_thresholds = {
            'low': (0.0, 0.3),
            'moderate': (0.3, 0.7),
            'high': (0.7, 1.0)
        }
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing components"""
        try:
            self.model_package = joblib.load(self.model_path)
            print(f"✓ Model loaded successfully: {self.model_package['model_name']}")
            print(f"✓ Training date: {self.model_package['training_date']}")
            print(f"✓ Model F1-Score: {self.model_package['performance_metrics']['f1']:.4f}")
            print(f"✓ Required features: {len(self.model_package['selected_features'])}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    def validate_input(self, data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
        """
        Validate and preprocess input data
        
        Args:
            data: Input data as DataFrame or dictionary
            
        Returns:
            pd.DataFrame: Validated and preprocessed data
        """
        # Convert to DataFrame if dictionary
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            data = pd.DataFrame([data])
        
        # Check required features
        required_features = self.model_package['selected_features']
        missing_features = set(required_features) - set(data.columns)
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only required features
        data = data[required_features].copy()
        
        # Handle missing values (fill with median - same as training)
        for col in data.columns:
            if data[col].isnull().any():
                median_val = data[col].median()
                if pd.isna(median_val):  # If all values are NaN, use 0
                    median_val = 0
                data[col].fillna(median_val, inplace=True)
                print(f"⚠ Filled missing values in {col} with {median_val}")
        
        return data
    
    def preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """
        Apply preprocessing steps (scaling) to input data
        
        Args:
            data: Validated input DataFrame
            
        Returns:
            np.ndarray: Preprocessed data ready for prediction
        """
        # Apply scaling if the model requires it (e.g., Logistic Regression)
        model_name = self.model_package['model_name']
        
        if model_name == 'LogisticRegression':
            # Scale the data using the saved scaler
            scaled_data = self.model_package['scaler'].transform(data)
            return scaled_data
        else:
            # For tree-based models, no scaling needed
            return data.values
    
    def categorize_risk(self, probability: float) -> Tuple[str, str, str]:
        """
        Categorize risk based on probability thresholds
        
        Args:
            probability: Predicted probability (0-1)
            
        Returns:
            Tuple: (risk_category, risk_description, recommendation)
        """
        if self.risk_thresholds['low'][0] <= probability < self.risk_thresholds['low'][1]:
            return (
                'Low Risk',
                'Low probability of developing CHD in the next 10 years',
                'Continue healthy lifestyle habits and regular check-ups'
            )
        elif self.risk_thresholds['moderate'][0] <= probability < self.risk_thresholds['moderate'][1]:
            return (
                'Moderate Risk',
                'Moderate probability of developing CHD in the next 10 years',
                'Consider lifestyle modifications and consult healthcare provider'
            )
        else:
            return (
                'High Risk',
                'High probability of developing CHD in the next 10 years',
                'Immediate medical consultation and lifestyle changes recommended'
            )
    
    def predict_single(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Make prediction for a single instance
        
        Args:
            data: Input data as DataFrame or dictionary
            
        Returns:
            Dict: Simplified prediction results
        """
        # Validate and preprocess input
        validated_data = self.validate_input(data)
        processed_data = self.preprocess_data(validated_data)
        
        # Make prediction
        model = self.model_package['model']
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        chd_probability = probabilities[1]  # Probability of CHD (class 1)
        
        # Risk categorization
        risk_category, risk_description, recommendation = self.categorize_risk(chd_probability)
        
        # Compile simplified results
        results = {
            'prediction': {
                'chd_risk_probability': round(chd_probability, 4),
                'predicted_class': int(prediction),
                'risk_category': risk_category,
                'risk_description': risk_description,
                'recommendation': recommendation,
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return results
    
    def print_prediction_report(self, results: Dict):
        """
        Print a simplified formatted prediction report
        
        Args:
            results: Prediction results dictionary
        """
        print("=" * 60)
        print("         CHD RISK PREDICTION REPORT")
        print("=" * 60)
        
        # Prediction Summary
        pred = results['prediction']
        
        print(f"🎯 PREDICTION SUMMARY")
        print(f"   CHD Risk Probability: {pred['chd_risk_probability']:.1%}")
        print(f"   Risk Category: {pred['risk_category']}")
        print(f"   Description: {pred['risk_description']}")
        print(f"   Prediction Date: {pred['prediction_date']}")
        
        print(f"\n\n💡 RECOMMENDATION")
        print(f"   {pred['recommendation']}")
        
        print("\n")


def create_sample_data():
    """Create sample data for testing different risk levels"""
    sample_data = {
        # Low risk profile - young, healthy female
        'low_risk_patient': {
            'male': 0,           # Female
            'age': 35,           # Young age
            'education': 3,      # Higher education
            'cigsPerDay': 0,     # Non-smoker
            'totChol': 180,      # Normal cholesterol
            'sysBP': 110,        # Normal systolic BP
            'diaBP': 70,         # Normal diastolic BP
            'BMI': 22,           # Normal BMI
            'heartRate': 70,     # Normal heart rate
            'glucose': 85        # Normal glucose
        },
        
        # Moderate risk profile - middle-aged male with some risk factors
        'moderate_risk_patient': {
            'male': 1,           # Male
            'age': 50,           # Middle age
            'education': 2,      # Moderate education
            'cigsPerDay': 10,    # Light smoker
            'totChol': 220,      # Slightly elevated cholesterol
            'sysBP': 140,        # Stage 1 hypertension
            'diaBP': 90,         # Stage 1 hypertension
            'BMI': 28,           # Overweight
            'heartRate': 80,     # Slightly elevated heart rate
            'glucose': 100       # Upper normal glucose
        },
        
        # High risk profile - older male with multiple risk factors
        'high_risk_patient': {
            'male': 1,           # Male
            'age': 65,           # Older age
            'education': 1,      # Lower education
            'cigsPerDay': 30,    # Heavy smoker
            'totChol': 280,      # High cholesterol
            'sysBP': 180,        # Stage 2 hypertension
            'diaBP': 110,        # Stage 2 hypertension
            'BMI': 35,           # Obese
            'heartRate': 95,     # Elevated heart rate
            'glucose': 130       # Pre-diabetic glucose level
        }
    }
    return sample_data


# Quick prediction function for easy backend integration
def quick_predict(patient_data: Dict, model_path: str = 'trained_models/best_model_logisticregression.pkl') -> Dict:
    """
    Quick prediction function for single patients - Perfect for backend integration
    
    Args:
        patient_data: Dictionary with patient information
        model_path: Path to model file
        
    Returns:
        Dict: Prediction results
    """
    predictor = CHDRiskPredictor(model_path)
    return predictor.predict_single(patient_data)


# Example usage and testing
if __name__ == "__main__":
    # Initialize the predictor
    try:
        predictor = CHDRiskPredictor()
        
        # Create sample data
        sample_patients = create_sample_data()
        
        print("\n🧪 TESTING SIMPLIFIED PREDICTION PIPELINE")
        print("=" * 60)
        
        # Test single predictions for each risk level
        for patient_name, patient_data in sample_patients.items():
            print(f"\n📋 Analyzing {patient_name.replace('_', ' ').title()}")
            print(f"Input data: {patient_data}")
            
            # Make prediction
            results = predictor.predict_single(patient_data)
            
            # Print formatted report
            predictor.print_prediction_report(results)
            
            # Also show the raw results for backend integration reference
            print("Raw Results (for backend use):")
            print(f"Risk Probability: {results['prediction']['chd_risk_probability']}")
            print(f"Risk Category: {results['prediction']['risk_category']}")
            print(f"Recommendation: {results['prediction']['recommendation']}")
            
            print("-" * 60)
        
        print("\n✅ TESTING COMPLETED!")
        print("Use the 'quick_predict()' function for easy backend integration")
        
        # Example of quick_predict usage
        print("\n🔧 QUICK PREDICT EXAMPLE:")
        quick_result = quick_predict(sample_patients['low_risk_patient'])
        print(f"Quick predict result: {quick_result['prediction']['risk_category']}")
        
    except FileNotFoundError:
        print("❌ Error: Model file not found!")
        print("Please ensure you have run the training script first.")
        print("Expected file: 'trained_models/best_model_logisticregression.pkl'")
    except Exception as e:
        print(f"❌ Error: {str(e)}")