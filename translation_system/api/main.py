"""
FastAPI Backend for Translation System
Provides REST API endpoints for all translation services
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
from datetime import datetime

# Import translation components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="EN-VI Translation System API",
    description="Scientific Document Translation with MarianMT, RAG, and Knowledge Graph",
    version="1.0.0"
)

# CORS middleware for web/mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TranslationRequest(BaseModel):
    text: str = Field(..., description="English text to translate")
    use_memory: bool = Field(True, description="Use translation memory")
    use_terminology: bool = Field(True, description="Use knowledge graph for terminology")


class TranslationResponse(BaseModel):
    source: str
    target: str
    method: str
    similar_translations: Optional[List[Dict]] = []
    standardized_terms: Optional[List[Dict]] = []
    timestamp: str


class DocumentTranslationRequest(BaseModel):
    sentences: List[str] = Field(..., description="List of sentences to translate")
    use_memory: bool = Field(True, description="Use translation memory")
    use_terminology: bool = Field(True, description="Use knowledge graph for terminology")


class TerminologyRequest(BaseModel):
    term_en: str
    term_vi: str
    category: str = "general"
    synonyms_en: Optional[List[str]] = []
    synonyms_vi: Optional[List[str]] = []
    definition: Optional[str] = None


class TerminologyResponse(BaseModel):
    term_id: str
    term_en: str
    term_vi: str
    category: str


class MemoryAddRequest(BaseModel):
    source: str
    target: str
    metadata: Optional[Dict] = {}


# Global instances (will be initialized on startup)
translation_engine = None
rag_engine = None
knowledge_graph = None


@app.on_event("startup")
async def startup_event():
    """Initialize translation components on startup"""
    global translation_engine, rag_engine, knowledge_graph
    
    logger.info("Initializing translation system...")
    
    try:
        # Import and initialize components
        from core.translator import TranslationEngine
        from rag.translation_memory import TranslationMemory, RAGTranslationEngine
        from knowledge_graph.terminology import TerminologyKnowledgeGraph, create_default_terminology
        
        # Initialize translation engine
        translation_engine = TranslationEngine(use_cache=True)
        
        # Initialize translation memory
        translation_memory = TranslationMemory(
            collection_name="en_vi_translations",
            persist_directory="./data/translation_memory"
        )
        
        # Initialize RAG engine
        rag_engine = RAGTranslationEngine(
            base_translator=translation_engine.translator,
            translation_memory=translation_memory
        )
        
        # Initialize knowledge graph
        knowledge_graph = create_default_terminology()
        
        logger.info("Translation system initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing translation system: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "EN-VI Translation System",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "translation_engine": translation_engine is not None,
            "rag_engine": rag_engine is not None,
            "knowledge_graph": knowledge_graph is not None
        }
    }


@app.post("/translate", response_model=TranslationResponse)
async def translate_text(request: TranslationRequest):
    """
    Translate English text to Vietnamese
    
    Uses RAG-based translation memory and knowledge graph for terminology
    """
    try:
        # Translate using RAG engine
        result = rag_engine.translate_with_memory(
            text=request.text,
            use_memory=request.use_memory
        )
        
        # Apply terminology standardization if requested
        standardized_terms = []
        if request.use_terminology:
            found_terms = knowledge_graph.find_terms_in_text(request.text)
            if found_terms:
                result["target"] = knowledge_graph.replace_terms_in_translation(
                    request.text, 
                    result["target"]
                )
                standardized_terms = [
                    {"term_en": t[0], "term_vi": t[1]} for t in found_terms
                ]
        
        return TranslationResponse(
            source=result["source"],
            target=result["target"],
            method=result["method"],
            similar_translations=result.get("similar_translations", []),
            standardized_terms=standardized_terms,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/translate/document")
async def translate_document(request: DocumentTranslationRequest):
    """
    Translate an entire document (list of sentences)
    """
    try:
        results = []
        
        for sentence in request.sentences:
            # Translate each sentence
            result = rag_engine.translate_with_memory(
                text=sentence,
                use_memory=request.use_memory
            )
            
            # Apply terminology standardization
            standardized_terms = []
            if request.use_terminology:
                found_terms = knowledge_graph.find_terms_in_text(sentence)
                if found_terms:
                    result["target"] = knowledge_graph.replace_terms_in_translation(
                        sentence,
                        result["target"]
                    )
                    standardized_terms = [
                        {"term_en": t[0], "term_vi": t[1]} for t in found_terms
                    ]
            
            results.append({
                "source": result["source"],
                "target": result["target"],
                "method": result["method"],
                "standardized_terms": standardized_terms
            })
        
        return {
            "translations": results,
            "total_sentences": len(request.sentences),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Document translation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/add")
async def add_to_memory(request: MemoryAddRequest):
    """Add a translation pair to memory"""
    try:
        translation_id = rag_engine.memory.add_translation(
            source=request.source,
            target=request.target,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "translation_id": translation_id,
            "message": "Translation added to memory"
        }
        
    except Exception as e:
        logger.error(f"Error adding to memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/memory/stats")
async def get_memory_stats():
    """Get translation memory statistics"""
    try:
        stats = rag_engine.memory.get_collection_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting memory stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/terminology/add", response_model=TerminologyResponse)
async def add_terminology(request: TerminologyRequest):
    """Add a new terminology to knowledge graph"""
    try:
        term_id = knowledge_graph.add_term(
            term_en=request.term_en,
            term_vi=request.term_vi,
            category=request.category,
            synonyms_en=request.synonyms_en,
            synonyms_vi=request.synonyms_vi,
            definition=request.definition
        )
        
        return TerminologyResponse(
            term_id=term_id,
            term_en=request.term_en,
            term_vi=request.term_vi,
            category=request.category
        )
        
    except Exception as e:
        logger.error(f"Error adding terminology: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/terminology/{term_en}")
async def get_terminology(term_en: str):
    """Get terminology information"""
    try:
        term_info = knowledge_graph.get_term_info(term_en)
        
        if term_info:
            return term_info
        else:
            raise HTTPException(status_code=404, detail="Term not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting terminology: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/terminology/category/{category}")
async def get_category_terms(category: str):
    """Get all terms in a category"""
    try:
        terms = knowledge_graph.get_category_terms(category)
        return {
            "category": category,
            "terms": terms,
            "total": len(terms)
        }
        
    except Exception as e:
        logger.error(f"Error getting category terms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/terminology/stats")
async def get_terminology_stats():
    """Get knowledge graph statistics"""
    try:
        stats = knowledge_graph.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting terminology stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def get_model_info():
    """Get information about the translation model"""
    try:
        info = translation_engine.translator.get_model_info()
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
