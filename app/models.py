from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """Base model for text-based requests."""
    text: str = Field(..., min_length=1, description="The input text to process")


class SentimentResponse(BaseModel):
    """Response model for sentiment analysis."""
    sentiment: str = Field(..., description="Sentiment classification (positive, negative, neutral)")
    score: float = Field(..., description="Confidence score for the sentiment")
    

class EntityResponse(BaseModel):
    """Model for a single named entity."""
    text: str = Field(..., description="The entity text")
    label: str = Field(..., description="The entity type/label")
    start: int = Field(..., description="Start position in the original text")
    end: int = Field(..., description="End position in the original text")


class NERResponse(BaseModel):
    """Response model for named entity recognition."""
    entities: List[EntityResponse] = Field(..., description="List of recognized entities")


class SummaryResponse(BaseModel):
    """Response model for text summarization."""
    summary: str = Field(..., description="Generated summary of the input text")


class LanguageResponse(BaseModel):
    """Response model for language detection."""
    language: str = Field(..., description="Detected language code")
    confidence: float = Field(..., description="Confidence score for the detection")


class KeywordResponse(BaseModel):
    """Response model for keyword extraction."""
    keywords: List[Dict[str, Any]] = Field(..., description="List of extracted keywords with scores")


class ErrorResponse(BaseModel):
    """Model for error responses."""
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
