"""
Knowledge Graph for Scientific Terminology Standardization
Ensures semantic consistency across documents
"""

import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerminologyKnowledgeGraph:
    """
    Knowledge Graph for managing scientific terminology
    Ensures consistent translation of domain-specific terms
    """
    
    def __init__(self, domain: str = "scientific"):
        """
        Initialize the knowledge graph
        
        Args:
            domain: Scientific domain (e.g., 'medical', 'computer_science', 'physics')
        """
        self.domain = domain
        self.graph = nx.DiGraph()
        self.term_index = {}  # Fast lookup for terms
        logger.info(f"Initialized Knowledge Graph for domain: {domain}")
    
    def add_term(self, term_en: str, term_vi: str, 
                 category: str = "general",
                 synonyms_en: Optional[List[str]] = None,
                 synonyms_vi: Optional[List[str]] = None,
                 definition: Optional[str] = None) -> str:
        """
        Add a terminology pair to the knowledge graph
        
        Args:
            term_en: English term
            term_vi: Vietnamese translation
            category: Category/field of the term
            synonyms_en: List of English synonyms
            synonyms_vi: List of Vietnamese synonyms
            definition: Definition of the term
            
        Returns:
            Term ID
        """
        term_id = f"term_{hash(term_en)}_{category}"
        
        # Add node to graph
        self.graph.add_node(term_id, 
                           term_en=term_en.lower(),
                           term_vi=term_vi,
                           category=category,
                           synonyms_en=synonyms_en or [],
                           synonyms_vi=synonyms_vi or [],
                           definition=definition,
                           timestamp=datetime.now().isoformat())
        
        # Index for fast lookup
        self.term_index[term_en.lower()] = term_id
        
        # Add synonym edges
        if synonyms_en:
            for syn in synonyms_en:
                syn_lower = syn.lower()
                if syn_lower not in self.term_index:
                    self.term_index[syn_lower] = term_id
        
        logger.info(f"Added term: {term_en} -> {term_vi} (category: {category})")
        return term_id
    
    def add_relationship(self, term_id1: str, term_id2: str, 
                        relationship: str = "related"):
        """
        Add a relationship between two terms
        
        Args:
            term_id1: First term ID
            term_id2: Second term ID
            relationship: Type of relationship (e.g., 'related', 'parent', 'synonym')
        """
        self.graph.add_edge(term_id1, term_id2, relationship=relationship)
        logger.info(f"Added relationship: {term_id1} --[{relationship}]--> {term_id2}")
    
    def get_translation(self, term_en: str) -> Optional[str]:
        """
        Get standardized Vietnamese translation for an English term
        
        Args:
            term_en: English term to translate
            
        Returns:
            Vietnamese translation or None if not found
        """
        term_lower = term_en.lower()
        
        if term_lower in self.term_index:
            term_id = self.term_index[term_lower]
            term_data = self.graph.nodes[term_id]
            return term_data.get('term_vi')
        
        return None
    
    def get_term_info(self, term_en: str) -> Optional[Dict]:
        """
        Get complete information about a term
        
        Args:
            term_en: English term
            
        Returns:
            Dictionary with term information
        """
        term_lower = term_en.lower()
        
        if term_lower in self.term_index:
            term_id = self.term_index[term_lower]
            term_data = self.graph.nodes[term_id]
            
            # Get related terms
            related = []
            for neighbor in self.graph.neighbors(term_id):
                edge_data = self.graph.edges[term_id, neighbor]
                neighbor_data = self.graph.nodes[neighbor]
                related.append({
                    "term_en": neighbor_data.get('term_en'),
                    "term_vi": neighbor_data.get('term_vi'),
                    "relationship": edge_data.get('relationship')
                })
            
            return {
                "term_id": term_id,
                "term_en": term_data.get('term_en'),
                "term_vi": term_data.get('term_vi'),
                "category": term_data.get('category'),
                "synonyms_en": term_data.get('synonyms_en', []),
                "synonyms_vi": term_data.get('synonyms_vi', []),
                "definition": term_data.get('definition'),
                "related_terms": related
            }
        
        return None
    
    def find_terms_in_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Find all known terms in a text
        
        Args:
            text: English text to analyze
            
        Returns:
            List of (term_en, term_vi) tuples found in text
        """
        text_lower = text.lower()
        found_terms = []
        
        # Check each term in index
        for term_en, term_id in self.term_index.items():
            if term_en in text_lower:
                term_data = self.graph.nodes[term_id]
                found_terms.append((term_en, term_data.get('term_vi')))
        
        # Remove duplicates and sort by length (longer terms first)
        found_terms = list(set(found_terms))
        found_terms.sort(key=lambda x: len(x[0]), reverse=True)
        
        return found_terms
    
    def replace_terms_in_translation(self, source_text: str, translated_text: str) -> str:
        """
        Replace terms in translated text with standardized terminology
        
        Args:
            source_text: Original English text
            translated_text: Translated Vietnamese text
            
        Returns:
            Vietnamese text with standardized terminology
        """
        found_terms = self.find_terms_in_text(source_text)
        
        result = translated_text
        for term_en, term_vi in found_terms:
            # This is a simple replacement - could be improved with NLP
            result = result.replace(term_en, term_vi)
        
        return result
    
    def get_category_terms(self, category: str) -> List[Dict]:
        """
        Get all terms in a specific category
        
        Args:
            category: Category name
            
        Returns:
            List of term dictionaries
        """
        terms = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get('category') == category:
                terms.append({
                    "term_id": node_id,
                    "term_en": node_data.get('term_en'),
                    "term_vi": node_data.get('term_vi'),
                    "category": category
                })
        
        return terms
    
    def save_to_file(self, filepath: str):
        """
        Save knowledge graph to JSON file
        
        Args:
            filepath: Path to save file
        """
        data = {
            "domain": self.domain,
            "nodes": [],
            "edges": []
        }
        
        # Save nodes
        for node_id, node_data in self.graph.nodes(data=True):
            node_dict = {"id": node_id}
            node_dict.update(node_data)
            data["nodes"].append(node_dict)
        
        # Save edges
        for source, target, edge_data in self.graph.edges(data=True):
            edge_dict = {
                "source": source,
                "target": target
            }
            edge_dict.update(edge_data)
            data["edges"].append(edge_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Knowledge graph saved to {filepath}")
    
    def load_from_file(self, filepath: str):
        """
        Load knowledge graph from JSON file
        
        Args:
            filepath: Path to load file from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.domain = data.get("domain", "scientific")
        self.graph.clear()
        self.term_index.clear()
        
        # Load nodes
        for node_data in data.get("nodes", []):
            node_id = node_data.pop("id")
            self.graph.add_node(node_id, **node_data)
            
            # Rebuild index
            term_en = node_data.get("term_en", "").lower()
            if term_en:
                self.term_index[term_en] = node_id
            
            # Index synonyms
            for syn in node_data.get("synonyms_en", []):
                self.term_index[syn.lower()] = node_id
        
        # Load edges
        for edge_data in data.get("edges", []):
            source = edge_data.pop("source")
            target = edge_data.pop("target")
            self.graph.add_edge(source, target, **edge_data)
        
        logger.info(f"Knowledge graph loaded from {filepath}")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the knowledge graph
        
        Returns:
            Dictionary with statistics
        """
        return {
            "domain": self.domain,
            "total_terms": self.graph.number_of_nodes(),
            "total_relationships": self.graph.number_of_edges(),
            "indexed_terms": len(self.term_index)
        }


def create_default_terminology():
    """
    Create a default knowledge graph with common scientific terms
    
    Returns:
        TerminologyKnowledgeGraph instance
    """
    kg = TerminologyKnowledgeGraph(domain="general_scientific")
    
    # Add common scientific terms
    terms = [
        ("algorithm", "thuật toán", "computer_science", ["method", "procedure"], ["phương pháp"]),
        ("machine learning", "học máy", "ai", ["ML"], ["máy học"]),
        ("neural network", "mạng nơ-ron", "ai", ["NN"], ["mạng neural"]),
        ("database", "cơ sở dữ liệu", "computer_science", ["DB"], ["CSDL"]),
        ("artificial intelligence", "trí tuệ nhân tạo", "ai", ["AI"], ["AI"]),
        ("deep learning", "học sâu", "ai", ["DL"], []),
        ("data science", "khoa học dữ liệu", "data", ["DS"], []),
        ("hypothesis", "giả thuyết", "research", ["theory"], ["lý thuyết"]),
        ("methodology", "phương pháp luận", "research", ["method"], ["phương pháp"]),
        ("analysis", "phân tích", "general", ["examination"], ["nghiên cứu"]),
    ]
    
    for term_en, term_vi, category, syn_en, syn_vi in terms:
        kg.add_term(term_en, term_vi, category, syn_en, syn_vi)
    
    logger.info("Created default terminology knowledge graph")
    return kg
