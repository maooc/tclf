"""FastAPI REST API for tclf trade classification.

This module provides a REST API to classify financial trades using
the tclf library's ClassicalClassifier.
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from tclf.classical_classifier import ClassicalClassifier

app = FastAPI(
    title="TCLF Trade Classification API",
    description="A REST API for classifying financial trades using classical trade classification rules",
    version="1.0.0",
)


class TradeData(BaseModel):
    """Trade data input model for JSON requests."""

    trade_price: list[float] = Field(..., description="Trade prices")
    bid_ex: list[float | None] = Field(None, description="Bid prices at exchange level")
    ask_ex: list[float | None] = Field(None, description="Ask prices at exchange level")
    bid_best: list[float | None] = Field(None, description="Best bid prices (NBBO)")
    ask_best: list[float | None] = Field(None, description="Best ask prices (NBBO)")
    price_ex_lag: list[float | None] = Field(None, description="Previous trade price at exchange")


class ClassificationRequest(BaseModel):
    """Classification request model."""

    data: TradeData
    layers: list[tuple[str, str]] = Field(
        default=[("quote", "ex")],
        description="Classification layers, e.g., [('quote', 'ex'), ('tick', 'ex')]",
    )
    strategy: str = Field(default="random", description="Fallback strategy: 'random' or 'const'")


class ClassificationResponse(BaseModel):
    """Classification response model."""

    predictions: list[int]
    probabilities: list[list[float]] | None = None
    message: str = "Success"


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    version: str = "1.0.0"


def _convert_to_dataframe(data: TradeData) -> pd.DataFrame:
    """Convert TradeData to pandas DataFrame.

    Args:
        data: TradeData model

    Returns:
        pd.DataFrame with proper column names
    """
    df_dict: dict[str, Any] = {"trade_price": data.trade_price}

    if data.bid_ex is not None:
        df_dict["bid_ex"] = data.bid_ex
    if data.ask_ex is not None:
        df_dict["ask_ex"] = data.ask_ex
    if data.bid_best is not None:
        df_dict["bid_best"] = data.bid_best
    if data.ask_best is not None:
        df_dict["ask_best"] = data.ask_best
    if data.price_ex_lag is not None:
        df_dict["price_ex_lag"] = data.price_ex_lag

    return pd.DataFrame(df_dict)


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    """Root endpoint for health check."""
    return HealthResponse(status="healthy")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="healthy")


@app.post("/predict", response_model=ClassificationResponse)
async def predict(request: ClassificationRequest) -> ClassificationResponse:
    """Classify trades using classical classification rules.

    Args:
        request: ClassificationRequest containing trade data and configuration

    Returns:
        ClassificationResponse with predictions and probabilities
    """
    try:
        # Convert to DataFrame
        X = _convert_to_dataframe(request.data)

        # Validate layers
        valid_layers = []
        for layer in request.layers:
            if len(layer) != 2:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid layer format: {layer}. Expected (rule, subset)",
                )
            valid_layers.append(layer)

        # Create and fit classifier
        clf = ClassicalClassifier(
            layers=valid_layers,
            strategy=request.strategy,
        )
        clf.fit(X)

        # Get predictions
        predictions = clf.predict(X).tolist()

        # Get probabilities if available
        try:
            probabilities = clf.predict_proba(X).tolist()
        except Exception:
            probabilities = None

        return ClassificationResponse(
            predictions=predictions,
            probabilities=probabilities,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/csv", response_model=ClassificationResponse)
async def predict_csv(
    file: UploadFile = File(...),
    layers: str = "quote,ex",
    strategy: str = "random",
) -> ClassificationResponse:
    """Classify trades from CSV file upload.

    Args:
        file: CSV file with trade data
        layers: Comma-separated layers, e.g., "quote,ex|tick,ex"
        strategy: Fallback strategy

    Returns:
        ClassificationResponse with predictions
    """
    try:
        # Read CSV file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Parse layers
        layer_list = []
        for layer_str in layers.split("|"):
            parts = layer_str.split(",")
            if len(parts) == 2:
                layer_list.append((parts[0].strip(), parts[1].strip()))

        if not layer_list:
            layer_list = [("quote", "ex")]

        # Create and fit classifier
        clf = ClassicalClassifier(layers=layer_list, strategy=strategy)
        clf.fit(df)

        # Get predictions
        predictions = clf.predict(df).tolist()

        # Get probabilities if available
        try:
            probabilities = clf.predict_proba(df).tolist()
        except Exception:
            probabilities = None

        return ClassificationResponse(
            predictions=predictions,
            probabilities=probabilities,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/batch")
async def predict_batch(
    data: list[dict[str, Any]],
    layers: list[tuple[str, str]] = [("quote", "ex")],
    strategy: str = "random",
) -> dict[str, Any]:
    """Classify a batch of trades from JSON array.

    Args:
        data: List of trade records
        layers: Classification layers
        strategy: Fallback strategy

    Returns:
        Dictionary with predictions and metadata
    """
    try:
        df = pd.DataFrame(data)

        clf = ClassicalClassifier(layers=layers, strategy=strategy)
        clf.fit(df)

        predictions = clf.predict(df).tolist()

        try:
            probabilities = clf.predict_proba(df).tolist()
        except Exception:
            probabilities = None

        return {
            "count": len(predictions),
            "predictions": predictions,
            "probabilities": probabilities,
            "layers": layers,
            "strategy": strategy,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
