from fastapi import APIRouter, HTTPException, Depends
from typing import Any

from app.models import (
    TextRequest, 
    SentimentResponse, 
    NERResponse, 
    SummaryResponse,
    LanguageResponse, 
    KeywordResponse
)
from app.services.nlp_service import NLPService

router = APIRouter(prefix="/api/v1", tags=["NLP"])

# Dependency to get NLP service
def get_nlp_service():
    return NLPService()


@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TextRequest, nlp_service: NLPService = Depends(get_nlp_service)) -> Any:
    """
    Analyze the sentiment of the provided text.
    
    Returns sentiment classification (positive, negative, neutral) and confidence score.
    """
    try:
        result = nlp_service.analyze_sentiment(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")


@router.post("/entities", response_model=NERResponse)
async def extract_entities(request: TextRequest, nlp_service: NLPService = Depends(get_nlp_service)) -> Any:
    """
    Extract named entities from the provided text.
    
    Returns a list of entities with their types, positions, and text.
    """
    try:
        entities = nlp_service.extract_entities(request.text)
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting entities: {str(e)}")


@router.post("/summarize", response_model=SummaryResponse)
async def summarize_text(
    request: TextRequest, 
    max_length: int = 150,
    nlp_service: NLPService = Depends(get_nlp_service)
) -> Any:
    """
    Generate a summary of the provided text.
    
    Returns a concise summary of the input text.
    """
    try:
        summary = nlp_service.summarize_text(request.text, max_length=max_length)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {str(e)}")


@router.post("/language", response_model=LanguageResponse)
async def detect_language(request: TextRequest, nlp_service: NLPService = Depends(get_nlp_service)) -> Any:
    """
    Detect the language of the provided text.
    
    Returns the detected language code and confidence score.
    """
    try:
        result = nlp_service.detect_language(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}")


@router.post("/keywords", response_model=KeywordResponse)
async def extract_keywords(
    request: TextRequest, 
    top_n: int = 10,
    nlp_service: NLPService = Depends(get_nlp_service)
) -> Any:
    """
    Extract keywords from the provided text.
    
    Returns a list of keywords with their importance scores.
    """
    try:
        keywords = nlp_service.extract_keywords(request.text, top_n=top_n)
        return {"keywords": keywords}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting keywords: {str(e)}")
