# import cv2
# import pytesseract
# import re
# import os
# from PIL import Image

# # Tesseract installation path (update if needed)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# class MedicalOCRPipeline:
#     def __init__(self, lang="eng"):
#         self.lang = lang

#         # Updated patterns for CHD prediction features
#         self.patterns = {
#             "Age": r"(?:Age|age)\s*[:\-]?\s*(\d+)",
#             "Sex": r"(?:Sex|Gender|sex|gender)\s*[:\-]?\s*(Male|Female|M|F|male|female|m|f)",
#             "Education": r"(?:Education|education)\s*[:\-]?\s*(\d+)",
#             "Smoking": r"(?:Cigarettes|Smoking|cigarettes|smoking|cigs)\s*[:\-]?\s*(\d+)",
#             "Blood Pressure": r"(?:Blood\s*Pressure|BP|blood\s*pressure|bp)\s*[:\-]?\s*(\d+[/\-]\d+)",
#             "Total Cholesterol": r"(?:Total\s*Cholesterol|Cholesterol|cholesterol|total\s*cholesterol)\s*[:\-]?\s*(\d+)",
#             "BMI": r"(?:BMI|bmi|Body\s*Mass\s*Index)\s*[:\-]?\s*([\d.]+)",
#             "Heart Rate": r"(?:Heart\s*Rate|Pulse|heart\s*rate|pulse|HR|hr)\s*[:\-]?\s*(\d+)",
#             "Glucose": r"(?:Glucose|Blood\s*Sugar|glucose|blood\s*sugar|sugar)\s*[:\-]?\s*(\d+)",
#             "Hypertension": r"(?:Hypertension|hypertension|High\s*BP)\s*[:\-]?\s*(Yes|No|Present|Absent|yes|no|present|absent)"
#         }

#     def preprocess_image(self, image_path):
#         """Load and preprocess the image for OCR."""
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"Could not load image from {image_path}")

#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Apply thresholding for noise removal
#         _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         return thresh

#     def extract_text(self, image_path):
#         """Perform OCR on the image and return extracted text."""
#         processed_img = self.preprocess_image(image_path)
#         text = pytesseract.image_to_string(processed_img, lang=self.lang, config="--psm 6")
#         return text

#     def parse_medical_values(self, text):
#         """Parse structured medical values from OCR text."""
#         results = {}
#         for field, pattern in self.patterns.items():
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 results[field] = match.group(1)
#         return results

#     def map_to_chd_format(self, ocr_data):
#         """Map OCR extracted data to CHD prediction format"""
#         mapping = {}
        
#         # Map age
#         if 'Age' in ocr_data:
#             try:
#                 mapping['age'] = int(ocr_data['Age'])
#             except ValueError:
#                 pass
        
#         # Map sex (male=1, female=0) 
#         if 'Sex' in ocr_data:
#             sex_value = ocr_data['Sex'].lower()
#             # Keep original text version
#             mapping['sex'] = ocr_data['Sex'].capitalize()  
            
#             # Encode for model input
#             if sex_value in ['male', 'm']:
#                 mapping['male'] = 1
#             elif sex_value in ['female', 'f']:
#                 mapping['male'] = 0

        
#         # Map education (default to 2 if not found)
#         if 'Education' in ocr_data:
#             try:
#                 mapping['education'] = int(ocr_data['Education'])
#             except ValueError:
#                 mapping['education'] = 2  # Default value
        
#         # Map smoking
#         if 'Smoking' in ocr_data:
#             try:
#                 mapping['cigsPerDay'] = int(ocr_data['Smoking'])
#             except ValueError:
#                 pass
        
#         # Map blood pressure
#         if 'Blood Pressure' in ocr_data:
#             bp = ocr_data['Blood Pressure']
#             if '/' in bp or '-' in bp:
#                 separator = '/' if '/' in bp else '-'
#                 bp_parts = bp.split(separator)
#                 if len(bp_parts) == 2:
#                     try:
#                         mapping['sysBP'] = int(bp_parts[0].strip())
#                         mapping['diaBP'] = int(bp_parts[1].strip())
#                     except ValueError:
#                         pass
        
#         # Map cholesterol
#         if 'Total Cholesterol' in ocr_data:
#             try:
#                 mapping['totChol'] = int(ocr_data['Total Cholesterol'])
#             except ValueError:
#                 pass
        
#         # Map BMI
#         if 'BMI' in ocr_data:
#             try:
#                 mapping['BMI'] = float(ocr_data['BMI'])
#             except ValueError:
#                 pass
        
#         # Map heart rate
#         if 'Heart Rate' in ocr_data:
#             try:
#                 mapping['heartRate'] = int(ocr_data['Heart Rate'])
#             except ValueError:
#                 pass
        
#         # Map glucose
#         if 'Glucose' in ocr_data:
#             try:
#                 mapping['glucose'] = int(ocr_data['Glucose'])
#             except ValueError:
#                 pass
        
#         # Map hypertension (prevalentHyp: 1 if yes, 0 if no)
#         if 'Hypertension' in ocr_data:
#             hyp_value = ocr_data['Hypertension'].lower()
#             if hyp_value in ['yes', 'present']:
#                 mapping['prevalentHyp'] = 1
#             elif hyp_value in ['no', 'absent']:
#                 mapping['prevalentHyp'] = 0
        
#         return mapping

#     def run_pipeline(self, image_path):
#         """Run the full OCR pipeline: preprocess → OCR → parse → map to CHD format."""
#         try:
#             text = self.extract_text(image_path)
#             parsed_data = self.parse_medical_values(text)
#             mapped_data = self.map_to_chd_format(parsed_data)
            
#             return {
#                 "success": True,
#                 "raw_text": text,
#                 "parsed_data": parsed_data,
#                 "chd_format_data": mapped_data,
#                 "fields_extracted": len(mapped_data)
#             }
#         except Exception as e:
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "raw_text": "",
#                 "parsed_data": {},
#                 "chd_format_data": {},
#                 "fields_extracted": 0
#             }

#     def process_uploaded_file(self, file_path):
#         """Process an uploaded file and return CHD-ready data"""
#         if not os.path.exists(file_path):
#             return {"success": False, "error": "File not found"}
        
#         result = self.run_pipeline(file_path)
        
#         # Clean up the uploaded file after processing
#         try:
#             os.remove(file_path)
#         except:
#             pass  # Don't fail if cleanup fails
        
#         return result


# def process_medical_document(file_path):
#     """
#     Simple function to process a medical document and return CHD prediction data
    
#     Args:
#         file_path (str): Path to the uploaded image file
        
#     Returns:
#         dict: Processed data ready for CHD prediction
#     """
#     ocr_pipeline = MedicalOCRPipeline()
#     return ocr_pipeline.process_uploaded_file(file_path)


# # Test function for development
# if __name__ == "__main__":
#     # Test the OCR pipeline
#     test_image = "Medical_Image.png"  # Replace with actual test image
#     if os.path.exists(test_image):
#         result = process_medical_document(test_image)
#         print("OCR Result:", result)
#     else:
#         print("No test image found. Place a test image and update the path.")








# import cv2
# import pytesseract
# import re
# import os
# from PIL import Image

# # Path to Tesseract OCR executable (update if installed elsewhere)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# class MedicalOCRPipeline:
#     def __init__(self, lang="eng"):
#         self.lang = lang

#         # Regex patterns for medical fields (expanded with OCR error tolerance)
#         self.patterns = {
#             "Age": r"(?:Age)\s*[:\-]?\s*(\d+)",
#             "Sex": r"(?:Sex|Gender)\s*[:\-]?\s*(Male|Female|M|F|male|female|m|f)",
#             "Education": r"(?:Education)\s*[:\-]?\s*(\d+)",
#             "Smoking": r"(?:Cigarettes|Smoking|Cigs)\s*[:\-]?\s*(\d+)",
#             "Blood Pressure": r"(?:Blood\s*Pressure|BP)\s*[:\-]?\s*(\d+[/\-]\d+)",
#             "Total Cholesterol": r"(?:Total\s*Cholesterol|Cholesterol|Sholesterol)\s*[:\-]?\s*(\d+)",
#             "BMI": r"(?:BMI|Body\s*Mass\s*Index)\s*[:\-]?\s*([\d.]+)",
#             "Heart Rate": r"(?:Heart\s*Rate|Pulse|HR)\s*[:\-]?\s*(\d+)",
#             "Glucose": r"(?:Glucose|Blood\s*Sugar|Sugar)\s*[:\-]?\s*(\d+)",
#             "Hypertension": r"(?:Hypertension|High\s*BP)\s*[:\-]?\s*(Yes|No|Present|Absent|yes|no|present|absent)"
#         }

#     def preprocess_image(self, image_path):
#         """Load and preprocess the image for OCR."""
#         img = cv2.imread(image_path)
#         if img is None:
#             raise ValueError(f"Could not load image from {image_path}")

#         # Convert to grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         # Apply Gaussian blur to reduce noise
#         blur = cv2.GaussianBlur(gray, (5, 5), 0)

#         # Adaptive thresholding (works better with uneven lighting)
#         thresh = cv2.adaptiveThreshold(
#             blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY, 31, 2
#         )

#         return thresh

#     def extract_text(self, image_path):
#         """Perform OCR on the image and return extracted text."""
#         processed_img = self.preprocess_image(image_path)
#         text = pytesseract.image_to_string(processed_img, lang=self.lang, config="--psm 6")
#         return text

#     def parse_medical_values(self, text):
#         """Parse structured medical values from OCR text."""
#         results = {}
#         for field, pattern in self.patterns.items():
#             match = re.search(pattern, text, re.IGNORECASE)
#             if match:
#                 results[field] = match.group(1).strip()
#         return results

#     def map_to_chd_format(self, ocr_data):
#         """Map OCR extracted data to CHD prediction format."""
#         mapping = {}

#         # Age
#         if "Age" in ocr_data:
#             try:
#                 mapping["age"] = int(ocr_data["Age"])
#             except ValueError:
#                 mapping["age"] = None

#         # Sex
#         if "Sex" in ocr_data:
#             sex_value = ocr_data["Sex"].lower()
#             mapping["sex"] = ocr_data["Sex"].capitalize()
#             mapping["male"] = 1 if sex_value in ["male", "m"] else 0

#         # Education
#         mapping["education"] = int(ocr_data.get("Education", 2))  # Default = 2

#         # Smoking
#         if "Smoking" in ocr_data:
#             try:
#                 mapping["cigsPerDay"] = int(ocr_data["Smoking"])
#             except ValueError:
#                 mapping["cigsPerDay"] = None

#         # Blood Pressure
#         if "Blood Pressure" in ocr_data:
#             bp = ocr_data["Blood Pressure"]
#             sep = "/" if "/" in bp else "-"
#             parts = bp.split(sep)
#             if len(parts) == 2:
#                 try:
#                     mapping["sysBP"] = int(parts[0].strip())
#                     mapping["diaBP"] = int(parts[1].strip())
#                 except ValueError:
#                     mapping["sysBP"], mapping["diaBP"] = None, None

#         # Cholesterol
#         if "Total Cholesterol" in ocr_data:
#             try:
#                 mapping["totChol"] = int(ocr_data["Total Cholesterol"])
#             except ValueError:
#                 mapping["totChol"] = None

#         # BMI
#         if "BMI" in ocr_data:
#             try:
#                 mapping["BMI"] = float(ocr_data["BMI"])
#             except ValueError:
#                 mapping["BMI"] = None

#         # Heart Rate
#         if "Heart Rate" in ocr_data:
#             try:
#                 mapping["heartRate"] = int(ocr_data["Heart Rate"])
#             except ValueError:
#                 mapping["heartRate"] = None

#         # Glucose
#         if "Glucose" in ocr_data:
#             try:
#                 mapping["glucose"] = int(ocr_data["Glucose"])
#             except ValueError:
#                 mapping["glucose"] = None

#         # Hypertension
#         if "Hypertension" in ocr_data:
#             hyp_value = ocr_data["Hypertension"].lower()
#             mapping["prevalentHyp"] = 1 if hyp_value in ["yes", "present"] else 0

#         return mapping

#     def run_pipeline(self, image_path):
#         """Run full OCR pipeline."""
#         try:
#             text = self.extract_text(image_path)
#             parsed_data = self.parse_medical_values(text)
#             mapped_data = self.map_to_chd_format(parsed_data)

#             return {
#                 "success": True,
#                 "raw_text": text,
#                 "parsed_data": parsed_data,
#                 "chd_format_data": mapped_data,
#                 "fields_extracted": len(mapped_data)
#             }
#         except Exception as e:
#             return {
#                 "success": False,
#                 "error": str(e),
#                 "raw_text": "",
#                 "parsed_data": {},
#                 "chd_format_data": {},
#                 "fields_extracted": 0
#             }

#     def process_uploaded_file(self, file_path):
#         """Process an uploaded file and return CHD-ready data."""
#         if not os.path.exists(file_path):
#             return {"success": False, "error": "File not found"}

#         result = self.run_pipeline(file_path)

#         # Cleanup
#         try:
#             os.remove(file_path)
#         except Exception:
#             pass

#         return result


# def process_medical_document(file_path):
#     """Simple wrapper for pipeline."""
#     ocr_pipeline = MedicalOCRPipeline()
#     return ocr_pipeline.process_uploaded_file(file_path)


# # Test mode
# if __name__ == "__main__":
#     test_image = "Medical_Image_new.png"
#     if os.path.exists(test_image):
#         result = process_medical_document(test_image)
#         print("OCR Result:", result)
#     else:
#         print("⚠️ No test image found. Place a test image and update the path.")














import cv2
import pytesseract
import re
import os
from PIL import Image

# Tesseract installation path (update if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class MedicalOCRPipeline:
    def __init__(self, lang="eng"):
        self.lang = lang

        # Exact patterns for CHD prediction features based on your document format
        self.patterns = {
        "Age": r"Age\s*[:\-]?\s*(\d+)",
        "Sex": r"(?:Sex|Gender)\s*[:\-]?\s*(Male|Female|M|F)",
        "Education": r"Education\s*[:\-]?\s*(\d+)",
        "Cigarettes": r"(?:Cigarettes\s*per\s*Day|Smoking)\s*[:\-]?\s*(\d+)",
        "Blood_Pressure": r"(?:Blood\s*Pressure|BP)\s*[:\-]?\s*(\d{2,3}/\d{2,3})",
        "Total_Cholesterol": r"(?:Total\s*Cholesterol|Cholesterol)\s*[:\-]?\s*(\d+)",
        "BMI": r"BMI\s*[:\-]?\s*([\d.]+)",
        "Heart_Rate": r"(?:Heart\s*Rate|Pulse)\s*[:\-]?\s*(\d+)",
        "Glucose": r"Glucose\s*[:\-]?\s*(\d+)"
        }

    def preprocess_image(self, image_path):
        """Load and preprocess the image for OCR."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding for noise removal
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh

    def extract_text(self, image_path):
        """Perform OCR on the image and return extracted text."""
        processed_img = self.preprocess_image(image_path)
        text = pytesseract.image_to_string(processed_img, lang=self.lang, config="--psm 6")
        return text

    def parse_medical_values(self, text):
        """Parse structured medical values from OCR text."""
        results = {}
        for field, pattern in self.patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results[field] = match.group(1)
        return results

    def map_to_chd_format(self, ocr_data):
        """Map OCR extracted data to CHD prediction format"""
        mapping = {}
        
        # Map age
        if 'Age' in ocr_data:
            try:
                mapping['age'] = int(ocr_data['Age'])
            except ValueError:
                pass
        
        # Map sex (male=1, female=0)
        if 'Sex' in ocr_data:
            sex_value = ocr_data['Sex'].lower()
            if sex_value in ['male', 'm']:
                mapping['male'] = 1
            elif sex_value in ['female', 'f']:
                mapping['male'] = 0
        
        # Map education
        if 'Education' in ocr_data:
            try:
                mapping['education'] = int(ocr_data['Education'])
            except ValueError:
                pass
        
        # Map cigarettes per day
        if 'Cigarettes' in ocr_data:
            try:
                mapping['cigsPerDay'] = int(ocr_data['Cigarettes'])
            except ValueError:
                pass
        
        # Map blood pressure
        if 'Blood_Pressure' in ocr_data:
            bp = ocr_data['Blood_Pressure']
            if '/' in bp:
                bp_parts = bp.split('/')
                if len(bp_parts) == 2:
                    try:
                        mapping['sysBP'] = int(bp_parts[0].strip())
                        mapping['diaBP'] = int(bp_parts[1].strip())
                    except ValueError:
                        pass
        
        # Map total cholesterol
        if 'Total_Cholesterol' in ocr_data:
            try:
                mapping['totChol'] = int(ocr_data['Total_Cholesterol'])
            except ValueError:
                pass
        
        # Map BMI
        if 'BMI' in ocr_data:
            try:
                mapping['BMI'] = float(ocr_data['BMI'])
            except ValueError:
                pass
        
        # Map heart rate
        if 'Heart_Rate' in ocr_data:
            try:
                mapping['heartRate'] = int(ocr_data['Heart_Rate'])
            except ValueError:
                pass
        
        # Map glucose
        if 'Glucose' in ocr_data:
            try:
                mapping['glucose'] = int(ocr_data['Glucose'])
            except ValueError:
                pass
        
        return mapping

    def run_pipeline(self, image_path):
        """Run the full OCR pipeline: preprocess → OCR → parse → map to CHD format."""
        try:
            text = self.extract_text(image_path)
            parsed_data = self.parse_medical_values(text)
            mapped_data = self.map_to_chd_format(parsed_data)
            
            return {
                "success": True,
                "raw_text": text,
                "parsed_data": parsed_data,
                "chd_format_data": mapped_data,
                "fields_extracted": len(mapped_data)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_text": "",
                "parsed_data": {},
                "chd_format_data": {},
                "fields_extracted": 0
            }

    def process_uploaded_file(self, file_path):
        """Process an uploaded file and return CHD-ready data"""
        if not os.path.exists(file_path):
            return {"success": False, "error": "File not found"}
        
        result = self.run_pipeline(file_path)
        
        # Clean up the uploaded file after processing
        try:
            os.remove(file_path)
        except:
            pass  # Don't fail if cleanup fails
        
        return result


def process_medical_document(file_path):
    """
    Simple function to process a medical document and return CHD prediction data
    
    Args:
        file_path (str): Path to the uploaded image file
        
    Returns:
        dict: Processed data ready for CHD prediction
    """
    ocr_pipeline = MedicalOCRPipeline()
    return ocr_pipeline.process_uploaded_file(file_path)


# Test function for development
if __name__ == "__main__":
    # Test the OCR pipeline
    test_image = "Medical_Image_new.png"  # Replace with actual test image
    if os.path.exists(test_image):
        result = process_medical_document(test_image)
        print("OCR Result:", result)
    else:
        print("No test image found. Place a test image and update the path.")