document.addEventListener('DOMContentLoaded', () => {
    // API endpoint base path
    const API_BASE = window.location.origin;

    // API Status & System State Elements
    const statusIndicator = document.getElementById('status-indicator');
    const statusText = document.getElementById('status-text');

    // Workspace Form and Upload Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const dropPrompt = document.getElementById('drop-prompt');
    const fileNameText = document.getElementById('file-name-text');
    const fileSizeText = document.getElementById('file-size-text');

    // Diagnostic Workspace Output Containers
    const resultsPlaceholder = document.getElementById('results-placeholder');
    const loadingState = document.getElementById('loading-state');
    const resultsDisplay = document.getElementById('results-display');
    const diagnosticCard = document.getElementById('diagnostic-card');
    const outcomeMainIcon = document.getElementById('outcome-main-icon');
    const resultBadge = document.getElementById('result-badge');
    const resultText = document.getElementById('result-text');
    const diagnosticDesc = document.getElementById('diagnostic-desc');
    const yoloResultContainer = document.getElementById('yolo-result-container');
    const annotatedImagePreview = document.getElementById('annotated-image-preview');
    const viewerViewport = document.getElementById('viewer-viewport');

    // Metrics & Performance Summary Elements
    const fractureVal = document.getElementById('fracture-val');
    const fractureBar = document.getElementById('fracture-bar');
    const noFractureVal = document.getElementById('no-fracture-val');
    const noFractureBar = document.getElementById('no-fracture-bar');

    // Interactive YOLO Viewer Control Buttons
    const btnZoomIn = document.getElementById('btn-zoom-in');
    const btnZoomOut = document.getElementById('btn-zoom-out');
    const btnZoomReset = document.getElementById('btn-zoom-reset');
    const btnFullscreen = document.getElementById('btn-fullscreen');
    const btnDownload = document.getElementById('btn-download');

    // Live Date & Time Elements
    const liveDateVal = document.getElementById('live-date-val');
    const liveTimeVal = document.getElementById('live-time-val');

    // State Variables
    let selectedFile = null;
    let zoomScale = 1.0;
    let isFullscreen = false;

    // --- Live Date & Time Updater ---
    if (liveDateVal && liveTimeVal) {
        const updateTime = () => {
            const now = new Date();
            const day = String(now.getDate()).padStart(2, '0');
            const month = String(now.getMonth() + 1).padStart(2, '0');
            const year = now.getFullYear();
            liveDateVal.innerText = `${day}-${month}-${year}`;
            
            const hours = String(now.getHours()).padStart(2, '0');
            const minutes = String(now.getMinutes()).padStart(2, '0');
            liveTimeVal.innerText = `${hours}:${minutes} IST`;
        };
        updateTime();
        setInterval(updateTime, 60000);
    }

    // --- Part 1: Backend Health Verification ---
    async function checkHealth() {
        try {
            const response = await fetch(`${API_BASE}/api/health`);
            if (!response.ok) throw new Error("Server connection error");
            const data = await response.json();
            
            statusIndicator.className = 'status-indicator';
            
            if (data.model_loaded) {
                statusIndicator.classList.add('online');
                statusText.innerText = "System Online";
            } else {
                statusIndicator.classList.add('offline');
                statusText.innerHTML = `Models Offline | <span style="color:#f43f5e;font-weight:600">Train Weights</span>`;
            }
        } catch (error) {
            statusIndicator.className = 'status-indicator offline';
            statusText.innerText = "Connection Failed";
            console.error("Health check error:", error);
        }
    }

    // Run health check immediately and check every 10 seconds
    checkHealth();
    setInterval(checkHealth, 10000);

    // --- Part 2: Drag & Drop Files Upload Area ---
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target) && selectedFile === null) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (fileInput.files.length > 0) {
            handleFile(fileInput.files[0]);
        }
    });

    function formatBytes(bytes, decimals = 0) {
        if (bytes === 0) return '0 KB';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        if (i === 0) return bytes + ' Bytes';
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Invalid file format. Please upload a valid medical image file (PNG, JPG, JPEG, BMP).');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert('File size exceeds the 10MB maximum limit. Please optimize and try again.');
            return;
        }

        selectedFile = file;
        fileNameText.innerText = file.name;
        fileSizeText.innerText = formatBytes(file.size);
        
        // Show Image Preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => {
            imagePreview.src = reader.result;
            dropPrompt.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            analyzeBtn.disabled = false;
        };
    }

    // Remove Image Uploads
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Avoid triggering file chooser dialog
        resetUploadArea();
        resetResultsArea();
    });

    function resetUploadArea() {
        selectedFile = null;
        fileInput.value = '';
        imagePreview.src = '#';
        previewContainer.classList.add('hidden');
        dropPrompt.classList.remove('hidden');
        analyzeBtn.disabled = true;
    }

    function resetResultsArea() {
        resultsPlaceholder.classList.remove('hidden');
        loadingState.classList.add('hidden');
        resultsDisplay.classList.add('hidden');
        yoloResultContainer.classList.add('hidden');
        annotatedImagePreview.src = '#';
        zoomScale = 1.0;
        annotatedImagePreview.style.transform = 'scale(1)';
    }

    // --- Part 3: API Request & Inference Analysis ---
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading state
        resultsPlaceholder.classList.add('hidden');
        resultsDisplay.classList.add('hidden');
        loadingState.classList.remove('hidden');
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const response = await fetch(`${API_BASE}/api/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || "Prediction request failed");
            }

            const result = await response.json();
            displayResults(result);

        } catch (error) {
            console.error("Inference analysis failed:", error);
            alert(`Analysis failed: ${error.message}. Please verify the file and try again.`);
            resetResultsArea();
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    // --- Part 4: Diagnostics Render Engine ---
    function displayResults(data) {
        // Hide loading
        loadingState.classList.add('hidden');
        resultsDisplay.classList.remove('hidden');

        const prediction = data.prediction;
        const confidence = data.confidence * 100;
        const probs = data.probabilities;

        // Clear card styles
        diagnosticCard.className = 'outcome-alert-card';
        
        if (prediction === 'fracture') {
            diagnosticCard.classList.add('fracture-detected');
            resultBadge.innerText = "FRACTURE DETECTED";
            resultText.innerText = "Fracture Detected";
            diagnosticDesc.innerText = `The neural network detected structural patterns matching a cervical spine fracture with ${confidence.toFixed(1)}% confidence. Please prioritize for clinical radiology review.`;
            if (outcomeMainIcon) {
                outcomeMainIcon.className = 'fa-solid fa-triangle-exclamation';
            }
        } else {
            diagnosticCard.classList.add('no-fracture');
            resultBadge.innerText = "NO FRACTURE";
            resultText.innerText = "No Fracture Detected";
            diagnosticDesc.innerText = `No fracture pattern was detected by the neural network (Confidence: ${confidence.toFixed(1)}%). The cervical vertebrae appear structurally intact based on the scan features.`;
            if (outcomeMainIcon) {
                outcomeMainIcon.className = 'fa-solid fa-shield-halved';
            }
        }

        // Update probability progress bars
        const fractureProb = ((probs['fracture'] || 0) * 100).toFixed(1);
        const healthyKey = Object.keys(probs).find(key => key !== 'fracture') || 'no_fracture';
        const healthyProb = ((probs[healthyKey] || 0) * 100).toFixed(1);

        fractureVal.innerText = `${fractureProb}%`;
        fractureBar.style.width = `${fractureProb}%`;

        noFractureVal.innerText = `${healthyProb}%`;
        noFractureBar.style.width = `${healthyProb}%`;

        // Load YOLO Image
        if (prediction === 'fracture' && data.annotated_image) {
            annotatedImagePreview.src = `data:image/png;base64,${data.annotated_image}`;
            yoloResultContainer.classList.remove('hidden');
        } else {
            yoloResultContainer.classList.add('hidden');
            annotatedImagePreview.src = '#';
        }
        
        // Reset interactive zoom parameters
        zoomScale = 1.0;
        annotatedImagePreview.style.transform = 'scale(1)';
    }

    // --- Part 5: Interactive YOLO Picture Controls ---
    btnZoomIn.addEventListener('click', () => {
        if (zoomScale < 3.0) {
            zoomScale += 0.2;
            annotatedImagePreview.style.transform = `scale(${zoomScale})`;
        }
    });

    btnZoomOut.addEventListener('click', () => {
        if (zoomScale > 0.6) {
            zoomScale -= 0.2;
            annotatedImagePreview.style.transform = `scale(${zoomScale})`;
        }
    });

    btnZoomReset.addEventListener('click', () => {
        zoomScale = 1.0;
        annotatedImagePreview.style.transform = 'scale(1)';
    });

    btnFullscreen.addEventListener('click', () => {
        if (!isFullscreen) {
            viewerViewport.classList.add('fullscreen');
            btnFullscreen.innerHTML = '<i class="fa-solid fa-compress"></i>';
            isFullscreen = true;
        } else {
            viewerViewport.classList.remove('fullscreen');
            btnFullscreen.innerHTML = '<i class="fa-solid fa-expand"></i>';
            isFullscreen = false;
        }
    });

    // Escape key exits fullscreen
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && isFullscreen) {
            viewerViewport.classList.remove('fullscreen');
            btnFullscreen.innerHTML = '<i class="fa-solid fa-expand"></i>';
            isFullscreen = false;
        }
    });

    btnDownload.addEventListener('click', () => {
        if (!annotatedImagePreview.src || annotatedImagePreview.src.includes('#')) return;
        
        const link = document.createElement('a');
        link.download = `Fracture_Localization_Detection.png`;
        link.href = annotatedImagePreview.src;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    });
});
