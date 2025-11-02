// Simple OCR functionality
document.addEventListener('DOMContentLoaded', function() {
    const runOcrBtn = document.getElementById('run-ocr-btn');
    const ocrPredictBtn = document.getElementById('ocr-predict-btn');
    
    if (runOcrBtn) {
        runOcrBtn.addEventListener('click', handleOCR);
    }
    
    if (ocrPredictBtn) {
        ocrPredictBtn.addEventListener('click', handleOCR);
    }
});

async function handleOCR() {
    const fileInput = document.getElementById('ocr-image');
    const statusDiv = document.getElementById('ocr-status');
    
    if (!fileInput.files || !fileInput.files.length) {
        statusDiv.textContent = 'Please select an image file first.';
        return;
    }
    
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    
    statusDiv.textContent = 'Processing image...';
    
    try {
        const response = await fetch('/ocr_predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success && data.ocr && data.ocr.chd_format_data) {
            // Fill form fields
            fillFormFields(data.ocr.chd_format_data);
            statusDiv.textContent = `OCR completed! Extracted ${data.ocr.fields_extracted} fields. Please review and fill any missing fields.`;
            statusDiv.style.color = 'green';
        } else {
            statusDiv.textContent = 'OCR failed: ' + (data.error || 'Unknown error');
            statusDiv.style.color = 'red';
        }
    } catch (error) {
        statusDiv.textContent = 'Error: ' + error.message;
        statusDiv.style.color = 'red';
    }
}

function fillFormFields(data) {
    Object.keys(data).forEach(key => {
        const input = document.getElementById(key);
        if (input && data[key] !== null && data[key] !== undefined) {
            input.value = data[key];
        }
    });
}