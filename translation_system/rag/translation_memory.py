"""
RAG-based Translation Memory System
Implements smart translation memory using retrieval-augmented generation
"""

from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranslationMemory:
    """
    Smart Translation Memory using RAG (Retrieval-Augmented Generation)
    Stores and retrieves similar translations to improve consistency
    """
    
    def __init__(self, collection_name: str = "translations", 
                 persist_directory: str = "./translation_memory_db",
                 embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Initialize Translation Memory with vector database
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "Translation memory for EN-VI scientific documents"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def add_translation(self, source: str, target: str, 
                       metadata: Optional[Dict] = None) -> str:
        """
        Add a translation pair to memory
        
        Args:
            source: English source text
            target: Vietnamese target text
            metadata: Optional metadata (domain, date, quality score, etc.)
            
        Returns:
            ID of the stored translation
        """
        # Generate embedding for source text
        embedding = self.embedder.encode(source).tolist()
        
        # Create unique ID
        translation_id = f"trans_{hash(source + target)}_{datetime.now().timestamp()}"
        
        # Prepare metadata
        meta = metadata or {}
        meta.update({
            "source": source,
            "target": target,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add to collection
        self.collection.add(
            embeddings=[embedding],
            documents=[source],
            metadatas=[meta],
            ids=[translation_id]
        )
        
        logger.info(f"Added translation to memory: {translation_id}")
        return translation_id
    
    def add_translations_batch(self, translation_pairs: List[Tuple[str, str]], 
                              metadatas: Optional[List[Dict]] = None) -> List[str]:
        """
        Add multiple translation pairs to memory in batch
        
        Args:
            translation_pairs: List of (source, target) tuples
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of translation IDs
        """
        sources = [pair[0] for pair in translation_pairs]
        targets = [pair[1] for pair in translation_pairs]
        
        # Generate embeddings for all sources
        embeddings = self.embedder.encode(sources).tolist()
        
        # Create IDs
        translation_ids = [
            f"trans_{hash(s + t)}_{datetime.now().timestamp()}_{i}"
            for i, (s, t) in enumerate(zip(sources, targets))
        ]
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{}] * len(translation_pairs)
        
        for i, (source, target, meta) in enumerate(zip(sources, targets, metadatas)):
            meta.update({
                "source": source,
                "target": target,
                "timestamp": datetime.now().isoformat()
            })
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=sources,
            metadatas=metadatas,
            ids=translation_ids
        )
        
        logger.info(f"Added {len(translation_pairs)} translations to memory")
        return translation_ids
    
    def retrieve_similar(self, query: str, n_results: int = 5, 
                        min_similarity: float = 0.7) -> List[Dict]:
        """
        Retrieve similar translations from memory
        
        Args:
            query: English text to find similar translations for
            n_results: Maximum number of results to return
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of similar translation dictionaries
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Process results
        similar_translations = []
        
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            for i, metadata in enumerate(results['metadatas'][0]):
                # Calculate similarity score (1 - distance)
                distance = results['distances'][0][i] if 'distances' in results else 0
                similarity = 1 - distance
                
                if similarity >= min_similarity:
                    similar_translations.append({
                        "source": metadata.get("source", ""),
                        "target": metadata.get("target", ""),
                        "similarity": similarity,
                        "metadata": metadata
                    })
        
        logger.info(f"Retrieved {len(similar_translations)} similar translations")
        return similar_translations
    
    def get_best_match(self, query: str, threshold: float = 0.9) -> Optional[Dict]:
        """
        Get the best matching translation if similarity is above threshold
        
        Args:
            query: English text to find match for
            threshold: Similarity threshold for exact match
            
        Returns:
            Best matching translation or None
        """
        similar = self.retrieve_similar(query, n_results=1, min_similarity=threshold)
        
        if similar and len(similar) > 0:
            return similar[0]
        
        return None
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the translation memory
        
        Returns:
            Dictionary with collection statistics
        """
        count = self.collection.count()
        
        return {
            "collection_name": self.collection_name,
            "total_translations": count,
            "persist_directory": self.persist_directory
        }
    
    def clear_memory(self):
        """Clear all translations from memory"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Translation memory for EN-VI scientific documents"}
        )
        logger.info("Translation memory cleared")


class RAGTranslationEngine:
    """
    RAG-enhanced translation engine that uses translation memory
    """
    
    def __init__(self, base_translator, translation_memory: TranslationMemory):
        """
        Initialize RAG translation engine
        
        Args:
            base_translator: Base translation model (e.g., MarianTranslator)
            translation_memory: TranslationMemory instance
        """
        self.base_translator = base_translator
        self.memory = translation_memory
        logger.info("RAG Translation Engine initialized")
    
    def translate_with_memory(self, text: str, 
                             use_memory: bool = True,
                             similarity_threshold: float = 0.9,
                             store_result: bool = True) -> Dict:
        """
        Translate using RAG approach with translation memory
        
        Args:
            text: English text to translate
            use_memory: Whether to use translation memory
            similarity_threshold: Threshold for using cached translation
            store_result: Whether to store the new translation in memory
            
        Returns:
            Translation result with metadata
        """
        result = {
            "source": text,
            "target": "",
            "method": "base",
            "similar_translations": []
        }
        
        # Check memory for similar translations
        if use_memory:
            best_match = self.memory.get_best_match(text, threshold=similarity_threshold)
            
            if best_match:
                # Use cached translation
                result["target"] = best_match["target"]
                result["method"] = "memory_exact"
                result["similarity"] = best_match["similarity"]
                logger.info(f"Using exact match from memory (similarity: {best_match['similarity']:.3f})")
                return result
            
            # Get similar translations for context
            similar = self.memory.retrieve_similar(text, n_results=3, min_similarity=0.7)
            result["similar_translations"] = similar
        
        # Perform new translation
        translated = self.base_translator.translate(text)
        result["target"] = translated
        result["method"] = "base_translation"
        
        # Store in memory if requested
        if store_result and use_memory:
            self.memory.add_translation(text, translated)
            logger.info("New translation stored in memory")
        
        return result
    
    def translate_document_with_memory(self, sentences: List[str]) -> List[Dict]:
        """
        Translate entire document using RAG approach
        
        Args:
            sentences: List of English sentences
            
        Returns:
            List of translation results
        """
        results = []
        
        for sentence in sentences:
            result = self.translate_with_memory(sentence)
            results.append(result)
        
        return results
