# EN-VI Scientific Document Translation System

A comprehensive translation system for English-Vietnamese scientific documents using MarianMT, RAG, and Knowledge Graph technologies.

## 🌟 Features

### Core Translation Engine
- **MarianMT Model**: Open-source machine translation model running locally (Edge AI)
- **Batch Translation**: Efficient processing of multiple texts
- **Caching**: Smart caching for improved performance

### RAG-based Translation Memory
- **Smart Memory**: Retrieval-Augmented Generation for consistent translations
- **Vector Database**: ChromaDB for similarity search
- **Context-Aware**: Retrieves similar past translations for consistency

### Knowledge Graph for Terminology
- **Standardization**: Ensures consistent scientific terminology
- **Domain-Specific**: Supports multiple scientific domains
- **Synonym Management**: Handles term variations

### Management Platform
- **Drupal Integration**: Content management with translation capabilities
- **REST API**: FastAPI backend with comprehensive endpoints
- **User-Friendly**: Intuitive interface for translation management

### Analytics
- **Apache Superset**: Real-time translation performance analytics
- **Metrics Tracking**: Usage statistics, quality metrics, and trends
- **Dashboards**: Customizable visualization of translation data

### Mobile Application
- **Cross-Platform**: React Native for iOS and Android
- **Offline Support**: Local translation history
- **Real-Time**: Instant translation with API integration

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Applications                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Web UI     │  │  Drupal CMS  │  │  Mobile App  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────┬───────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   FastAPI Backend  │
                    │   (REST API)       │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌─────────▼────────┐  ┌────────▼────────┐
│  MarianMT      │  │  Translation     │  │  Knowledge      │
│  Translation   │  │  Memory (RAG)    │  │  Graph          │
│  Engine        │  │  + ChromaDB      │  │  (Terminology)  │
└────────────────┘  └──────────────────┘  └─────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Apache Superset  │
                    │   (Analytics)      │
                    └────────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+ (for mobile app)
- Docker & Docker Compose (recommended)
- 4GB+ RAM (for ML models)
- PostgreSQL or MySQL (for analytics)

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd translation_system

# Start all services
docker-compose up -d

# The API will be available at http://localhost:8000
# Superset at http://localhost:8088
```

### Manual Installation

1. **Install Python dependencies**:
```bash
cd translation_system
pip install -r requirements.txt
```

2. **Start the API server**:
```bash
cd api
python main.py
```

3. **Access the API**:
- Open http://localhost:8000/docs for API documentation
- Use the interactive Swagger UI to test endpoints

## 📖 Documentation

### Core Components

#### 1. Translation Engine (`core/translator.py`)
- MarianMT-based translator
- Supports single and batch translation
- Edge AI deployment

```python
from core.translator import TranslationEngine

engine = TranslationEngine()
result = engine.translate("Hello world")
print(result['target'])  # Vietnamese translation
```

#### 2. RAG Translation Memory (`rag/translation_memory.py`)
- Vector-based similarity search
- Smart caching and retrieval
- Context-aware translation

```python
from rag.translation_memory import TranslationMemory, RAGTranslationEngine

memory = TranslationMemory()
rag_engine = RAGTranslationEngine(base_translator, memory)
result = rag_engine.translate_with_memory("Scientific term")
```

#### 3. Knowledge Graph (`knowledge_graph/terminology.py`)
- Terminology standardization
- Domain-specific vocabulary
- Relationship management

```python
from knowledge_graph.terminology import TerminologyKnowledgeGraph

kg = TerminologyKnowledgeGraph()
kg.add_term("algorithm", "thuật toán", "computer_science")
translation = kg.get_translation("algorithm")
```

### API Endpoints

#### Translation
- `POST /translate` - Translate text
- `POST /translate/document` - Translate document (multiple sentences)

#### Memory
- `POST /memory/add` - Add translation to memory
- `GET /memory/stats` - Get memory statistics

#### Terminology
- `POST /terminology/add` - Add new term
- `GET /terminology/{term}` - Get term information
- `GET /terminology/category/{category}` - Get terms by category
- `GET /terminology/stats` - Get knowledge graph statistics

#### System
- `GET /health` - Health check
- `GET /model/info` - Get model information

## 📱 Mobile App

See [mobile/README.md](mobile/README.md) for detailed mobile app documentation.

Quick start:
```bash
cd mobile
npm install
npm run android  # or npm run ios
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Translation Memory
CHROMA_PERSIST_DIR=./data/translation_memory
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# MarianMT
MARIAN_MODEL=Helsinki-NLP/opus-mt-en-vi
DEVICE=cpu  # or cuda

# Database (for analytics)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=translation_db
DB_USER=postgres
DB_PASSWORD=password

# Superset
SUPERSET_HOST=localhost
SUPERSET_PORT=8088
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=translation_system tests/
```

## 📊 Performance

### Benchmarks

- **Single Translation**: ~100-200ms
- **Batch Translation** (10 texts): ~500-800ms
- **Memory Lookup**: ~50-100ms
- **Terminology Standardization**: ~10-20ms

### Optimization Tips

1. Use batch translation for multiple texts
2. Enable translation memory for consistent content
3. Use GPU for faster inference (if available)
4. Cache frequently used translations

## 🔐 Security

- API authentication (configure in production)
- Input validation and sanitization
- Rate limiting
- CORS configuration
- Secure storage of sensitive data

## 🚢 Deployment

### Production Checklist

- [ ] Configure environment variables
- [ ] Set up database backup
- [ ] Enable HTTPS
- [ ] Configure authentication
- [ ] Set up monitoring and logging
- [ ] Configure rate limiting
- [ ] Review security settings

### Docker Deployment

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes Deployment

See [deployment/kubernetes/](deployment/kubernetes/) for Kubernetes manifests.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 👥 Team

- Development Team
- Contact: [team@example.com](mailto:team@example.com)

## 🙏 Acknowledgments

- MarianMT for the translation model
- Helsinki-NLP for pre-trained models
- LangChain and ChromaDB for RAG implementation
- FastAPI for the backend framework
- React Native for mobile development

## 📚 References

1. [MarianMT Documentation](https://huggingface.co/docs/transformers/model_doc/marian)
2. [ChromaDB Documentation](https://docs.trychroma.com/)
3. [FastAPI Documentation](https://fastapi.tiangolo.com/)
4. [Apache Superset Documentation](https://superset.apache.org/)

## 🆘 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Contact the development team
- Check the documentation

---

**Version**: 1.0.0
**Last Updated**: 2024
