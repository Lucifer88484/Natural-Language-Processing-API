import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.services.nlp_service import NLPService

client = TestClient(app)


# Mock NLP service for testing
@pytest.fixture
def mock_nlp_service():
    with patch("app.routes.nlp_routes.get_nlp_service") as mock_get_service:
        mock_service = MagicMock(spec=NLPService)
        mock_get_service.return_value = mock_service
        yield mock_service


def test_root_endpoint():
    """Test the root endpoint returns correct information."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "documentation" in data


def test_sentiment_analysis(mock_nlp_service):
    """Test sentiment analysis endpoint."""
    # Mock the service response
    mock_nlp_service.analyze_sentiment.return_value = {
        "sentiment": "positive",
        "score": 0.95
    }
    
    # Test the endpoint
    response = client.post(
        "/api/v1/sentiment",
        json={"text": "I love this product!"}
    )
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "positive"
    assert data["score"] == 0.95
    
    # Verify the service was called correctly
    mock_nlp_service.analyze_sentiment.assert_called_once_with("I love this product!")


def test_named_entity_recognition(mock_nlp_service):
    """Test named entity recognition endpoint."""
    # Mock the service response
    mock_nlp_service.extract_entities.return_value = [
        {"text": "John", "label": "PERSON", "start": 0, "end": 4}
    ]
    
    # Test the endpoint
    response = client.post(
        "/api/v1/entities",
        json={"text": "John went to New York."}
    )
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert len(data["entities"]) == 1
    assert data["entities"][0]["text"] == "John"
    assert data["entities"][0]["label"] == "PERSON"
    
    # Verify the service was called correctly
    mock_nlp_service.extract_entities.assert_called_once_with("John went to New York.")


def test_text_summarization(mock_nlp_service):
    """Test text summarization endpoint."""
    # Mock the service response
    mock_nlp_service.summarize_text.return_value = "This is a summary."
    
    # Test the endpoint
    response = client.post(
        "/api/v1/summarize",
        json={"text": "This is a long text that needs to be summarized."}
    )
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == "This is a summary."
    
    # Verify the service was called correctly
    mock_nlp_service.summarize_text.assert_called_once()


def test_language_detection(mock_nlp_service):
    """Test language detection endpoint."""
    # Mock the service response
    mock_nlp_service.detect_language.return_value = {
        "language": "en",
        "confidence": 0.99
    }
    
    # Test the endpoint
    response = client.post(
        "/api/v1/language",
        json={"text": "This is English text."}
    )
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert data["language"] == "en"
    assert data["confidence"] == 0.99
    
    # Verify the service was called correctly
    mock_nlp_service.detect_language.assert_called_once_with("This is English text.")


def test_keyword_extraction(mock_nlp_service):
    """Test keyword extraction endpoint."""
    # Mock the service response
    mock_nlp_service.extract_keywords.return_value = [
        {"keyword": "important", "score": 0.8},
        {"keyword": "keyword", "score": 0.7}
    ]
    
    # Test the endpoint
    response = client.post(
        "/api/v1/keywords",
        json={"text": "This text contains important keywords."}
    )
    
    # Verify the response
    assert response.status_code == 200
    data = response.json()
    assert len(data["keywords"]) == 2
    assert data["keywords"][0]["keyword"] == "important"
    
    # Verify the service was called correctly
    mock_nlp_service.extract_keywords.assert_called_once()
