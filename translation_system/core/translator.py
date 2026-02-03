"""
Core Translation Engine using MarianMT
Implements Edge AI for local English-Vietnamese translation
"""

import torch
from transformers import MarianMTModel, MarianTokenizer
from typing import List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarianTranslator:
    """
    MarianMT-based translator for English to Vietnamese
    Runs locally for Edge AI deployment
    """
    
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-vi", device: str = "cpu"):
        """
        Initialize the MarianMT translator
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cpu' or 'cuda')
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        logger.info(f"Initializing MarianMT translator on {self.device}")
        
        try:
            self.tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.model = MarianMTModel.from_pretrained(model_name).to(self.device)
            self.model_name = model_name
            logger.info(f"Model {model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def translate(self, text: str, max_length: int = 512) -> str:
        """
        Translate a single text from English to Vietnamese
        
        Args:
            text: English text to translate
            max_length: Maximum length of generated translation
            
        Returns:
            Translated Vietnamese text
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                translated = self.model.generate(**inputs, max_length=max_length)
            
            # Decode output
            translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
            
            return translated_text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return ""
    
    def translate_batch(self, texts: List[str], max_length: int = 512, batch_size: int = 8) -> List[str]:
        """
        Translate multiple texts in batches
        
        Args:
            texts: List of English texts to translate
            max_length: Maximum length of generated translations
            batch_size: Number of texts to process at once
            
        Returns:
            List of translated Vietnamese texts
        """
        translations = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                # Tokenize batch
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate translations
                with torch.no_grad():
                    translated = self.model.generate(**inputs, max_length=max_length)
                
                # Decode outputs
                batch_translations = [
                    self.tokenizer.decode(t, skip_special_tokens=True) 
                    for t in translated
                ]
                translations.extend(batch_translations)
                
            except Exception as e:
                logger.error(f"Batch translation error: {e}")
                translations.extend([""] * len(batch))
        
        return translations
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "vocab_size": self.tokenizer.vocab_size,
            "model_type": "MarianMT"
        }


class TranslationEngine:
    """
    Main translation engine that coordinates different translation strategies
    """
    
    def __init__(self, use_cache: bool = True):
        """
        Initialize translation engine
        
        Args:
            use_cache: Whether to use translation cache
        """
        self.translator = MarianTranslator()
        self.cache = {} if use_cache else None
        logger.info("Translation engine initialized")
    
    def translate(self, text: str, use_cache: bool = True) -> Dict[str, str]:
        """
        Translate text with optional caching
        
        Args:
            text: English text to translate
            use_cache: Whether to check/use cache
            
        Returns:
            Dictionary with source and target texts
        """
        # Check cache
        if use_cache and self.cache is not None and text in self.cache:
            logger.info("Using cached translation")
            return self.cache[text]
        
        # Translate
        translated = self.translator.translate(text)
        
        result = {
            "source": text,
            "target": translated,
            "model": self.translator.model_name
        }
        
        # Store in cache
        if use_cache and self.cache is not None:
            self.cache[text] = result
        
        return result
    
    def translate_document(self, sentences: List[str]) -> List[Dict[str, str]]:
        """
        Translate an entire document (list of sentences)
        
        Args:
            sentences: List of English sentences
            
        Returns:
            List of translation results
        """
        results = []
        
        # Translate in batches
        translations = self.translator.translate_batch(sentences)
        
        for source, target in zip(sentences, translations):
            results.append({
                "source": source,
                "target": target,
                "model": self.translator.model_name
            })
        
        return results
    
    def clear_cache(self):
        """Clear the translation cache"""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")
