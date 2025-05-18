import logging
from typing import List, Dict, Any, Tuple
import spacy
from transformers import pipeline
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

# Initialize logging
logger = logging.getLogger(__name__)

# Set seed for reproducibility
DetectorFactory.seed = 0

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


class NLPService:
    """Service class for NLP operations."""
    
    def __init__(self):
        """Initialize NLP models and pipelines."""
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all required NLP models."""
        logger.info("Initializing NLP models...")
        
        # Check if CUDA is available
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Using device: {'CUDA' if self.device == 0 else 'CPU'}")
        
        # Initialize sentiment analysis model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device
        )
        
        # Initialize summarization model
        self.summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=self.device
        )
        
        # Initialize spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Downloading spaCy model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        logger.info("NLP models initialized successfully")
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment label and score
        """
        try:
            logger.info("Performing sentiment analysis")
            result = self.sentiment_analyzer(text)[0]
            
            # Map the label to positive, negative, or neutral
            label = result['label'].lower()
            if label == 'positive' or label == 'pos':
                sentiment = 'positive'
            elif label == 'negative' or label == 'neg':
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
                
            return {
                "sentiment": sentiment,
                "score": result['score']
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of dictionaries containing entity information
        """
        try:
            logger.info("Performing named entity recognition")
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            return entities
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            raise
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """
        Generate a summary of the given text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of the summary
            
        Returns:
            Summarized text
        """
        try:
            logger.info("Performing text summarization")
            # Ensure text is long enough to summarize
            if len(text.split()) < 30:
                return text
            
            # Calculate appropriate min and max length
            min_length = min(30, max(10, len(text.split()) // 4))
            max_length = min(max_length, max(min_length + 5, len(text.split()) // 2))
            
            summary = self.summarizer(
                text, 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=False
            )[0]['summary_text']
            
            return summary
        except Exception as e:
            logger.error(f"Error in text summarization: {str(e)}")
            raise
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language of the given text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with language code and confidence
        """
        try:
            logger.info("Performing language detection")
            lang_code = detect(text)
            
            # Since langdetect doesn't provide confidence scores directly,
            # we'll return a placeholder confidence of 1.0
            return {
                "language": lang_code,
                "confidence": 1.0
            }
        except LangDetectException as e:
            logger.error(f"Error in language detection: {str(e)}")
            if "No features in text" in str(e):
                return {"language": "unknown", "confidence": 0.0}
            raise
        except Exception as e:
            logger.error(f"Error in language detection: {str(e)}")
            raise
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Extract keywords from the given text using TF-IDF.
        
        Args:
            text: Input text to analyze
            top_n: Number of top keywords to return
            
        Returns:
            List of dictionaries containing keywords and their scores
        """
        try:
            logger.info("Performing keyword extraction")
            # Tokenize and remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(text.lower())
            filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
            
            # If text is too short, return tokens as keywords
            if len(filtered_tokens) < 5:
                return [{"keyword": token, "score": 1.0} for token in filtered_tokens[:top_n]]
            
            # Use TF-IDF for keyword extraction
            vectorizer = TfidfVectorizer(max_features=2*top_n)
            tfidf_matrix = vectorizer.fit_transform([' '.join(filtered_tokens)])
            
            # Get feature names and TF-IDF scores
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Create keyword-score pairs and sort by score
            keyword_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top N keywords
            return [{"keyword": kw, "score": float(score)} for kw, score in keyword_scores[:top_n]]
        except Exception as e:
            logger.error(f"Error in keyword extraction: {str(e)}")
            raise
