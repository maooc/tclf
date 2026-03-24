"""FastAPI REST API for tclf trade classification."""

import io
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from tclf.classical_classifier import ClassicalClassifier

app = FastAPI(
    title="tclf API",
    description="Trade Classification API - Classify financial market transactions into buyer- and seller-initiated trades",
    version="0.3.0",
)

clf = ClassicalClassifier(layers=[("quote", "ex"), ("quote", "best")], strategy="random")


class TradeData(BaseModel):
    """Trade data model for JSON input."""

    data: list[list[float | None]] = Field(..., description="2D array of trade data")
    features: list[str] = Field(
        ...,
        description="List of feature names. Must follow naming conventions: trade_price, bid_ex, ask_ex, bid_best, ask_best, etc.",
    )


class PredictRequest(BaseModel):
    """Request model for prediction."""

    trade_data: TradeData
    layers: list[tuple[str, str]] | None = Field(
        default=[("quote", "ex"), ("quote", "best")],
        description="Classification layers, e.g., [('quote', 'ex'), ('quote', 'best')]",
    )
    strategy: Literal["random", "const"] = Field(
        default="random",
        description="Strategy for unclassified trades: 'random' or 'const'",
    )


class PredictResponse(BaseModel):
    """Response model for prediction."""

    predictions: list[int] = Field(..., description="Predicted trade classes: -1 (sell), 1 (buy)")
    probabilities: list[list[float]] = Field(
        ..., description="Class probabilities [prob_sell, prob_buy]"
    )
    n_trades: int = Field(..., description="Number of trades classified")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "tclf API",
        "description": "Trade Classification API",
        "version": "0.3.0",
        "endpoints": {
            "/predict": "POST - Classify trades from JSON data",
            "/predict/csv": "POST - Classify trades from CSV file",
            "/health": "GET - Health check",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Classify trades from JSON data.

    Args:
        request: PredictRequest containing trade data, layers, and strategy.

    Returns:
        PredictResponse with predictions, probabilities, and trade count.
    """
    try:
        X = np.array(request.trade_data.data, dtype=np.float64)

        classifier = ClassicalClassifier(
            layers=request.layers,
            strategy=request.strategy,
            features=request.trade_data.features,
        )
        classifier.fit(X)
        predictions = classifier.predict(X)
        probabilities = classifier.predict_proba(X)

        return PredictResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist(),
            n_trades=len(predictions),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/predict/csv", response_model=PredictResponse)
async def predict_csv(
    file: UploadFile = File(...),
    layers: str = "quote:ex,quote:best",
    strategy: Literal["random", "const"] = "random",
):
    """Classify trades from CSV file.

    The CSV file must have column names following tclf naming conventions:
    - trade_price: Trade price
    - bid_ex, ask_ex: Exchange-level bid/ask prices
    - bid_best, ask_best: NBBO bid/ask prices
    - trade_size: Trade size
    - bid_size_ex, ask_size_ex: Exchange-level bid/ask sizes

    Args:
        file: CSV file with trade data.
        layers: Comma-separated layers in format 'rule:subset', e.g., 'quote:ex,quote:best'.
        strategy: Strategy for unclassified trades.

    Returns:
        PredictResponse with predictions, probabilities, and trade count.
    """
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a CSV file")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        parsed_layers = []
        for layer in layers.split(","):
            rule, subset = layer.strip().split(":")
            parsed_layers.append((rule, subset))

        classifier = ClassicalClassifier(
            layers=parsed_layers if parsed_layers else None,
            strategy=strategy,
        )
        classifier.fit(df)
        predictions = classifier.predict(df)
        probabilities = classifier.predict_proba(df)

        return PredictResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist(),
            n_trades=len(predictions),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/demo")
async def demo():
    """Run a demo classification with sample data."""
    X = pd.DataFrame(
        [
            [1.5, 1, 3, 2, 2.5],
            [2.5, 1, 3, 1, 3],
            [1.5, 3, 1, 1, 3],
            [2.5, 3, 1, 1, 3],
            [1, np.nan, 1, 1, 3],
            [3, np.nan, np.nan, 1, 3],
        ],
        columns=["trade_price", "bid_ex", "ask_ex", "bid_best", "ask_best"],
    )

    clf.fit(X)
    predictions = clf.predict(X)
    probabilities = clf.predict_proba(X)

    return {
        "input_data": X.to_dict(orient="records"),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "legend": {-1: "seller-initiated (sell)", 1: "buyer-initiated (buy)"},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
