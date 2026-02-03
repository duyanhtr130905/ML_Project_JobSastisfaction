# Installation and Setup Guide

## Quick Installation

### Option 1: Docker (Recommended)

The easiest way to get started:

```bash
cd translation_system
docker-compose up -d
```

This will start:
- Translation API on port 8000
- PostgreSQL database on port 5432
- Apache Superset on port 8088
- Redis cache on port 6379

Access the API documentation at: http://localhost:8000/docs

### Option 2: Manual Installation

#### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (for ML models)
- Git

#### Step 1: Clone and Navigate

```bash
git clone <repository-url>
cd translation_system
```

#### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch and Transformers (for MarianMT)
- LangChain and ChromaDB (for RAG)
- NetworkX and RDFLib (for Knowledge Graph)
- FastAPI and Uvicorn (for API server)
- And other dependencies

**Note**: This may take 10-15 minutes depending on your internet connection.

#### Step 4: Download Models (Optional)

The models will download automatically on first use, but you can pre-download them:

```python
from transformers import MarianMTModel, MarianTokenizer

# Download MarianMT model
MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-vi')
MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-vi')

# Download sentence transformer
from sentence_transformers import SentenceTransformer
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
```

#### Step 5: Start the API Server

```bash
cd api
python main.py
```

Or using uvicorn directly:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at: http://localhost:8000

## Verification

### Check System Structure

```bash
python tests/verify_structure.py
```

This will verify that all files are in place and have valid syntax.

### Test Basic Functionality

After installing dependencies:

```bash
python tests/test_basic.py
```

### Run Examples

```bash
python examples/usage_examples.py
```

This will demonstrate all major features of the translation system.

## Configuration

### Environment Variables

Create a `.env` file in the `translation_system` directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Translation Memory
CHROMA_PERSIST_DIR=./data/translation_memory
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# MarianMT
MARIAN_MODEL=Helsinki-NLP/opus-mt-en-vi
DEVICE=cpu  # Change to "cuda" if you have GPU

# Database (for analytics)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=translation_db
DB_USER=postgres
DB_PASSWORD=postgres123

# Superset
SUPERSET_HOST=localhost
SUPERSET_PORT=8088
```

### GPU Support (Optional)

If you have a CUDA-capable GPU:

1. Install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

2. Set `DEVICE=cuda` in your `.env` file

This can significantly speed up translation (2-10x faster).

## Troubleshooting

### Issue: "No module named 'torch'"

**Solution**: Install PyTorch:
```bash
pip install torch
```

### Issue: "No module named 'transformers'"

**Solution**: Install transformers:
```bash
pip install transformers sentencepiece
```

### Issue: "Model download fails"

**Solution**: Check your internet connection and try again. Models are large (200-500MB each).

### Issue: "Out of memory"

**Solution**: 
- Reduce batch size in settings
- Use CPU instead of GPU
- Close other applications
- Upgrade RAM if possible

### Issue: "Port 8000 already in use"

**Solution**: Change the port:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### Issue: "ChromaDB errors"

**Solution**: Clear the database:
```bash
rm -rf data/translation_memory/
```

## Testing the API

Once the API is running, test it:

### Using curl

```bash
# Health check
curl http://localhost:8000/health

# Translate text
curl -X POST "http://localhost:8000/translate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "use_memory": true, "use_terminology": true}'

# Get model info
curl http://localhost:8000/model/info
```

### Using Python

```python
import requests

# Translate
response = requests.post(
    "http://localhost:8000/translate",
    json={
        "text": "Machine learning is powerful",
        "use_memory": True,
        "use_terminology": True
    }
)
print(response.json())
```

### Using Swagger UI

Open http://localhost:8000/docs in your browser for interactive API documentation.

## Mobile App Setup

See [mobile/README.md](mobile/README.md) for mobile app installation.

Quick start:

```bash
cd mobile
npm install
npm run android  # or npm run ios
```

## Drupal Integration

See [drupal/README.md](drupal/README.md) for Drupal module installation.

Quick start:

```bash
cd drupal/translation_module
# Copy to your Drupal modules/custom/ directory
drush en translation_module -y
```

## Analytics Setup

See [analytics/README.md](analytics/README.md) for Apache Superset setup.

With Docker Compose, Superset is automatically configured.
Access it at: http://localhost:8088 (username: admin, password: admin)

## Production Deployment

For production deployment:

1. Use a proper database (PostgreSQL)
2. Set up proper authentication
3. Use HTTPS
4. Configure CORS properly
5. Set up monitoring and logging
6. Use environment variables for secrets
7. Scale with multiple workers

Example production command:

```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

## Support

For issues or questions:
- Check the documentation in each module's directory
- Run the verification script: `python tests/verify_structure.py`
- Review the examples: `python examples/usage_examples.py`
- Check the API documentation: http://localhost:8000/docs

## Next Steps

After successful installation:

1. Review the [README.md](README.md) for system overview
2. Run the examples: `python examples/usage_examples.py`
3. Test the API endpoints using Swagger UI
4. Explore the mobile app
5. Set up Drupal integration if needed
6. Configure analytics dashboards

Happy translating! 🌐
