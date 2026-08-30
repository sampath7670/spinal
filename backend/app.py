import os
import sys
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Append the project root to sys.path to allow importing from 'src'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

try:
    from src.predict import FracturePredictor
except ImportError:
    # Handle if running from parent dir directly
    sys.path.append(os.path.join(project_root, 'src'))
    from predict import FracturePredictor

app = FastAPI(
    title="Cervical Fracture Detection API",
    description="Backend API to classify cervical spine images as showing a fracture or no fracture.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
model_path = os.path.join(project_root, 'models', 'best_model.pth')
frontend_path = os.path.join(project_root, 'frontend')

# Global predictor instance
predictor = None
model_mtime = None

@app.on_event("startup")
def load_model():
    global predictor, model_mtime
    print(f"Server starting up. Looking for model checkpoint at: {model_path}")
    if os.path.exists(model_path):
        try:
            predictor = FracturePredictor(model_path)
            model_mtime = os.path.getmtime(model_path)
            print(f"Model loaded successfully on startup! (mtime: {model_mtime})")
        except Exception as e:
            print(f"Error loading model on startup: {e}")
    else:
        print("WARNING: 'models/best_model.pth' not found. Prediction endpoint will be unavailable until a model is trained.")

@app.get("/api/health")
def health_check():
    """
    Returns the server health and model load status.
    """
    global predictor
    status = "healthy"
    model_loaded = predictor is not None
    return {
        "status": status,
        "model_loaded": model_loaded,
        "message": "API is online." if model_loaded else "API is online but model weights are missing. Please run model training."
    }

@app.post("/api/predict")
async def predict_fracture(file: UploadFile = File(...)):
    """
    Upload an image of a cervical spine (X-ray/CT slice) to check for fractures.
    """
    global predictor, model_mtime
    
    # Reload model if weights file changed (e.g. newly trained model)
    if os.path.exists(model_path):
        current_mtime = os.path.getmtime(model_path)
        if predictor is None or model_mtime is None or current_mtime > model_mtime:
            print("New or updated model checkpoint detected. Reloading weights...")
            try:
                predictor = FracturePredictor(model_path)
                model_mtime = current_mtime
                print("Model weights reloaded successfully!")
            except Exception as e:
                print(f"Failed to reload updated model: {e}")
                
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model weights not found or failed to load. Please train the model using 'python src/train.py' before making predictions."
        )
            
    # Validate file extension
    allowed_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed types: {', '.join(allowed_extensions)}"
        )

    # Save uploaded file to a temp file for predictor
    try:
        suffix = ext
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
        # Perform prediction
        pred_class, probs, annotated_base64 = predictor.predict(temp_file_path, original_filename=file.filename)
        
        # Clean up temp file
        os.remove(temp_file_path)
        
        return {
            "filename": file.filename,
            "prediction": pred_class,
            "probabilities": probs,
            "confidence": probs[pred_class],
            "annotated_image": annotated_base64
        }
    except Exception as e:
        # Clean up in case of error
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Serve frontend static files
# Make sure frontend directory exists
if os.path.exists(frontend_path):
    # Mount frontend files at root URL
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
else:
    print(f"WARNING: Frontend path '{frontend_path}' does not exist. UI static files will not be served.")
