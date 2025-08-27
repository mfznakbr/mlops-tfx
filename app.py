from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import os
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()

# Instrumentator untuk Prometheus
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Load model dengan tf.saved_model.load (sudah terbukti work)
model = tf.saved_model.load('model')
print("✅ Model loaded successfully with tf.saved_model.load!")
print("Available signatures:", list(model.signatures.keys()))

class PredictionRequest(BaseModel):
    features: list[float]

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": True,
        "signatures": list(model.signatures.keys())
    }

@app.get("/")
async def root():
    return {"message": "Customer Churn Prediction API - Model Loaded Successfully!"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Gunakan serving_default signature
        infer = model.signatures["serving_default"]
        
        # Convert features to tensor (sesuai input model lo)
        # Ganti 'input_1' dengan nama input layer yang bener
        input_tensor = tf.constant([request.features], dtype=tf.float32)
        
        # Prediction
        prediction = infer(input_tensor)
        
        # Ambil output (biasanya 'output_0' atau 'dense_3')
        output_key = list(prediction.keys())[0]
        prediction_value = prediction[output_key].numpy()
        
        return {
            "prediction": prediction_value.tolist(),
            "churn_probability": float(prediction_value[0][0]),
            "output_key": output_key
        }
    except Exception as e:
        return {"error": str(e), "input_features": request.features}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # fallback ke 8000 kalau lokal
    uvicorn.run(app, host="0.0.0.0", port=port)
