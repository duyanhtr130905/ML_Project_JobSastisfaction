"""
Example usage of the EN-VI Translation System
Demonstrates all major features
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.translator import TranslationEngine, MarianTranslator
from rag.translation_memory import TranslationMemory, RAGTranslationEngine
from knowledge_graph.terminology import TerminologyKnowledgeGraph, create_default_terminology


def example_basic_translation():
    """Example 1: Basic translation using MarianMT"""
    print("=" * 60)
    print("Example 1: Basic Translation")
    print("=" * 60)
    
    # Initialize translator
    translator = MarianTranslator()
    
    # Translate a single text
    text = "Machine learning is a subset of artificial intelligence."
    result = translator.translate(text)
    
    print(f"English: {text}")
    print(f"Vietnamese: {result}")
    print()


def example_batch_translation():
    """Example 2: Batch translation"""
    print("=" * 60)
    print("Example 2: Batch Translation")
    print("=" * 60)
    
    translator = MarianTranslator()
    
    texts = [
        "Hello world",
        "Machine learning is powerful",
        "Data science is important",
        "Neural networks are complex"
    ]
    
    results = translator.translate_batch(texts)
    
    for en, vi in zip(texts, results):
        print(f"EN: {en}")
        print(f"VI: {vi}")
        print()


def example_translation_memory():
    """Example 3: Translation memory with RAG"""
    print("=" * 60)
    print("Example 3: Translation Memory (RAG)")
    print("=" * 60)
    
    # Initialize components
    translator = MarianTranslator()
    memory = TranslationMemory(
        collection_name="example_translations",
        persist_directory="./data/example_memory"
    )
    
    # Add some translations to memory
    print("Adding translations to memory...")
    memory.add_translation(
        "machine learning",
        "học máy",
        metadata={"category": "ai", "quality": "high"}
    )
    memory.add_translation(
        "deep learning",
        "học sâu",
        metadata={"category": "ai", "quality": "high"}
    )
    
    # Create RAG engine
    rag_engine = RAGTranslationEngine(translator, memory)
    
    # Translate with memory
    print("\nTranslating with memory...")
    result = rag_engine.translate_with_memory(
        "machine learning",
        use_memory=True,
        similarity_threshold=0.9
    )
    
    print(f"Source: {result['source']}")
    print(f"Target: {result['target']}")
    print(f"Method: {result['method']}")
    
    if result.get('similar_translations'):
        print("\nSimilar translations found:")
        for sim in result['similar_translations']:
            print(f"  - {sim['source']} -> {sim['target']} (similarity: {sim['similarity']:.3f})")
    print()


def example_knowledge_graph():
    """Example 4: Knowledge graph for terminology"""
    print("=" * 60)
    print("Example 4: Knowledge Graph Terminology")
    print("=" * 60)
    
    # Create knowledge graph with default terms
    kg = create_default_terminology()
    
    # Add custom terms
    print("Adding custom terms...")
    kg.add_term(
        "convolutional neural network",
        "mạng nơ-ron tích chập",
        category="deep_learning",
        synonyms_en=["CNN"],
        synonyms_vi=["CNN", "mạng CNN"]
    )
    
    # Get translation for a term
    term = "algorithm"
    translation = kg.get_translation(term)
    print(f"\nTerm: {term}")
    print(f"Translation: {translation}")
    
    # Get detailed term info
    info = kg.get_term_info("machine learning")
    if info:
        print(f"\nDetailed info for 'machine learning':")
        print(f"  Vietnamese: {info['term_vi']}")
        print(f"  Category: {info['category']}")
        print(f"  Synonyms (EN): {info['synonyms_en']}")
        print(f"  Synonyms (VI): {info['synonyms_vi']}")
    
    # Find terms in text
    text = "Machine learning and neural networks are important in artificial intelligence."
    found_terms = kg.find_terms_in_text(text)
    print(f"\nTerms found in text:")
    for term_en, term_vi in found_terms:
        print(f"  {term_en} -> {term_vi}")
    
    # Get category terms
    ai_terms = kg.get_category_terms("ai")
    print(f"\nTerms in 'ai' category: {len(ai_terms)}")
    for term in ai_terms[:3]:
        print(f"  {term['term_en']} -> {term['term_vi']}")
    
    print()


def example_integrated_translation():
    """Example 5: Integrated translation with all features"""
    print("=" * 60)
    print("Example 5: Integrated Translation")
    print("=" * 60)
    
    # Initialize all components
    translator = MarianTranslator()
    memory = TranslationMemory(
        collection_name="integrated_example",
        persist_directory="./data/integrated_memory"
    )
    kg = create_default_terminology()
    rag_engine = RAGTranslationEngine(translator, memory)
    
    # Scientific text to translate
    text = "Machine learning algorithms use neural networks to process data and make predictions."
    
    print(f"Original text:\n{text}\n")
    
    # Translate with memory
    result = rag_engine.translate_with_memory(text, use_memory=True)
    translated = result['target']
    
    print(f"Base translation:\n{translated}\n")
    
    # Apply terminology standardization
    found_terms = kg.find_terms_in_text(text)
    if found_terms:
        print("Standardizing terminology...")
        for term_en, term_vi in found_terms:
            print(f"  {term_en} -> {term_vi}")
        
        # Apply terminology to translation
        standardized = kg.replace_terms_in_translation(text, translated)
        print(f"\nStandardized translation:\n{standardized}\n")
    
    # Show statistics
    memory_stats = memory.get_collection_stats()
    kg_stats = kg.get_stats()
    
    print("System Statistics:")
    print(f"  Translation Memory: {memory_stats['total_translations']} translations")
    print(f"  Knowledge Graph: {kg_stats['total_terms']} terms, {kg_stats['total_relationships']} relationships")
    print()


def example_document_translation():
    """Example 6: Document translation"""
    print("=" * 60)
    print("Example 6: Document Translation")
    print("=" * 60)
    
    translator = MarianTranslator()
    
    # Document as list of sentences
    document = [
        "Introduction to Machine Learning",
        "Machine learning is a branch of artificial intelligence.",
        "It focuses on building systems that learn from data.",
        "Neural networks are a key component of deep learning.",
        "These techniques are widely used in various applications."
    ]
    
    print("Translating document...\n")
    
    translations = translator.translate_batch(document)
    
    for i, (en, vi) in enumerate(zip(document, translations), 1):
        print(f"Sentence {i}:")
        print(f"  EN: {en}")
        print(f"  VI: {vi}")
        print()


def main():
    """Run all examples"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "EN-VI Translation System Examples" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        # Run examples
        example_basic_translation()
        example_batch_translation()
        example_translation_memory()
        example_knowledge_graph()
        example_integrated_translation()
        example_document_translation()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
