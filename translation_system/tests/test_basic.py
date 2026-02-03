"""
Basic tests for the translation system
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from core.translator import TranslationEngine, MarianTranslator
        print("✓ core.translator imported successfully")
    except Exception as e:
        print(f"✗ Error importing core.translator: {e}")
        return False
    
    try:
        from rag.translation_memory import TranslationMemory, RAGTranslationEngine
        print("✓ rag.translation_memory imported successfully")
    except Exception as e:
        print(f"✗ Error importing rag.translation_memory: {e}")
        return False
    
    try:
        from knowledge_graph.terminology import TerminologyKnowledgeGraph, create_default_terminology
        print("✓ knowledge_graph.terminology imported successfully")
    except Exception as e:
        print(f"✗ Error importing knowledge_graph.terminology: {e}")
        return False
    
    try:
        from config.settings import API_HOST, API_PORT
        print("✓ config.settings imported successfully")
    except Exception as e:
        print(f"✗ Error importing config.settings: {e}")
        return False
    
    return True


def test_knowledge_graph():
    """Test knowledge graph functionality (no external dependencies)"""
    print("\nTesting Knowledge Graph...")
    
    try:
        from knowledge_graph.terminology import TerminologyKnowledgeGraph
        
        kg = TerminologyKnowledgeGraph(domain="test")
        
        # Add a term
        term_id = kg.add_term("test", "thử nghiệm", "general")
        assert term_id is not None, "Failed to add term"
        print("✓ Added term to knowledge graph")
        
        # Get translation
        translation = kg.get_translation("test")
        assert translation == "thử nghiệm", f"Expected 'thử nghiệm', got '{translation}'"
        print("✓ Retrieved translation from knowledge graph")
        
        # Get stats
        stats = kg.get_stats()
        assert stats['total_terms'] == 1, f"Expected 1 term, got {stats['total_terms']}"
        print("✓ Knowledge graph statistics working")
        
        return True
    except Exception as e:
        print(f"✗ Knowledge graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_default_terminology():
    """Test default terminology creation"""
    print("\nTesting Default Terminology...")
    
    try:
        from knowledge_graph.terminology import create_default_terminology
        
        kg = create_default_terminology()
        stats = kg.get_stats()
        
        assert stats['total_terms'] > 0, "No terms in default knowledge graph"
        print(f"✓ Default terminology created with {stats['total_terms']} terms")
        
        # Test some default terms
        test_terms = ["algorithm", "machine learning", "neural network"]
        for term in test_terms:
            translation = kg.get_translation(term)
            if translation:
                print(f"  {term} -> {translation}")
        
        return True
    except Exception as e:
        print(f"✗ Default terminology test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration settings"""
    print("\nTesting Configuration...")
    
    try:
        from config.settings import (
            API_HOST, API_PORT, MARIAN_MODEL, DEVICE,
            CHROMA_PERSIST_DIR, EMBEDDING_MODEL
        )
        
        print(f"✓ API Host: {API_HOST}")
        print(f"✓ API Port: {API_PORT}")
        print(f"✓ MarianMT Model: {MARIAN_MODEL}")
        print(f"✓ Device: {DEVICE}")
        print(f"✓ ChromaDB Directory: {CHROMA_PERSIST_DIR}")
        print(f"✓ Embedding Model: {EMBEDDING_MODEL}")
        
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Translation System - Basic Tests")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests that don't require heavy dependencies
    results.append(("Import Test", test_imports()))
    results.append(("Configuration Test", test_configuration()))
    results.append(("Knowledge Graph Test", test_knowledge_graph()))
    results.append(("Default Terminology Test", test_default_terminology()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
