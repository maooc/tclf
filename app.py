from typing import List, Dict, Union, Optional
import io
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from tclf.classical_classifier import ClassicalClassifier, ALLOWED_FUNC_LITERALS

app = FastAPI(
    title="tclf Trade Classification API",
    description="A REST API for trade classification using tclf library",
    version="1.0.0"
)

class ClassificationRequest(BaseModel):
    data: List[List[Union[float, None]]]
    features: Optional[List[str]] = None
    layers: List[List[Union[str, str]]] = [["quote", "ex"]]
    strategy: str = "random"
    random_state: Optional[int] = 42

class ClassificationResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]
    message: str = "Classification completed successfully"

@app.get("/")
async def root():
    return {
        "message": "Welcome to tclf Trade Classification API",
        "endpoints": {
            "/predict/json": "POST - Classify trades from JSON data",
            "/predict/csv": "POST - Classify trades from CSV file",
            "/docs": "Swagger UI documentation",
            "/openapi.json": "OpenAPI specification"
        },
        "supported_algorithms": list(ALLOWED_FUNC_LITERALS.__args__)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/predict/json", response_model=ClassificationResponse)
async def predict_json(request: ClassificationRequest):
    """
    Classify trades from JSON data.
    
    The request should include:
    - data: 2D array of trade data
    - features: Optional list of feature names (required if using numpy array)
    - layers: List of classification layers (algorithm and subset)
    - strategy: Fallback strategy for unclassified trades ("random" or "const")
    - random_state: Random seed for reproducibility
    
    Example features format: ["trade_price", "bid_ex", "ask_ex"]
    Example layers format: [["quote", "ex"], ["tick", "ex"]]
    """
    try:
        # Convert layers to tuple format
        layers = [(layer[0], layer[1]) for layer in request.layers]
        
        # Create and fit classifier
        clf = ClassicalClassifier(
            layers=layers,
            strategy=request.strategy,
            random_state=request.random_state,
            features=request.features
        )
        
        # Convert data to numpy array
        X = np.array(request.data, dtype=np.float64)
        
        # Fit and predict
        clf.fit(X)
        predictions = clf.predict(X).tolist()
        probabilities = clf.predict_proba(X).tolist()
        
        return ClassificationResponse(
            predictions=predictions,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/csv", response_model=ClassificationResponse)
async def predict_csv(
    file: UploadFile = File(...),
    layers: str = '[["quote", "ex"]]',
    strategy: str = "random",
    random_state: int = 42
):
    """
    Classify trades from a CSV file.
    
    The CSV file should have headers matching the naming convention:
    - trade_price: Trade price
    - bid_ex: Bid price at exchange level
    - ask_ex: Ask price at exchange level
    - bid_best: Best bid price (NBBO)
    - ask_best: Best ask price (NBBO)
    - etc.
    
    Example layers: '[["quote", "ex"], ["tick", "ex"]]'
    """
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Parse layers
        import json
        layers_list = json.loads(layers)
        layers_tuple = [(layer[0], layer[1]) for layer in layers_list]
        
        # Create and fit classifier
        clf = ClassicalClassifier(
            layers=layers_tuple,
            strategy=strategy,
            random_state=random_state
        )
        
        # Fit and predict
        clf.fit(df)
        predictions = clf.predict(df).tolist()
        probabilities = clf.predict_proba(df).tolist()
        
        return ClassificationResponse(
            predictions=predictions,
            probabilities=probabilities
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
