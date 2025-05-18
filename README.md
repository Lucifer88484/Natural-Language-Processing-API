# 🧠 Natural Language Processing API

A RESTful API for performing core NLP tasks such as sentiment analysis, named entity recognition, text summarization, keyword extraction, and language detection. Built using **FastAPI** and **state-of-the-art NLP models** from Hugging Face and spaCy.

---

## 🚀 Features

- ✅ Sentiment Analysis  
- ✅ Named Entity Recognition (NER)  
- ✅ Text Summarization  
- ✅ Keyword Extraction  
- ✅ Language Detection  
- 🔒 CORS & error handling  
- 📝 Auto-generated API docs with Swagger/OpenAPI

---

## 🛠️ Tech Stack

- Python 3.10+
- FastAPI
- spaCy / Hugging Face Transformers
- Pydantic
- Uvicorn (ASGI Server)
- Docker (optional)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/nlp-api.git
cd nlp-api
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download required models

```bash
python -m spacy download en_core_web_sm
```

---

## 🚀 Running the API

### Development server

```bash
uvicorn app.main:app --reload
```

The API will be available at <http://localhost:8000>

### Production server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 📚 API Documentation

Once the server is running, you can access:

- Interactive API documentation: <http://localhost:8000/docs>
- OpenAPI specification: <http://localhost:8000/openapi.json>

---

## 🧪 Running Tests

```bash
pytest
```

---

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/sentiment` | POST | Analyze sentiment of text |
| `/api/v1/entities` | POST | Extract named entities |
| `/api/v1/summarize` | POST | Generate text summary |
| `/api/v1/language` | POST | Detect language of text |
| `/api/v1/keywords` | POST | Extract keywords from text |

---

## 📄 License

Hrishikesh D. Mohite © 2025

---

## 🤝 Contributing

Mail me at: <hrishikeshmohite001@gmail.com>
