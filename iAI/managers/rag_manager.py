"""RAG Manager - Retrieval-Augmented Generation"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re


class SimpleRAGManager:
    """
    Simple RAG without external vector DB.
    Perfect for: understanding + immediate use
    
    Architecture:
    1. Load documents
    2. Create chunks with simple embeddings
    3. Retrieve similar chunks
    4. Generate response with context
    """
    
    def __init__(self, data_dir: str = "data/fine_tune"):
        self.data_dir = Path(data_dir)
        self.chunks = []  # List of document chunks
        self.chunk_metadata = []  # Metadata for each chunk
        self.documents = []  # For compatibility with QueryManager
        
    # ===================== STEP 1: INDEXING =====================
    
    def index_documents(self, qa_pairs: List[Dict]) -> int:
        """
        Index Q&A pairs for retrieval with sentence-level chunking.
        
        Args:
            qa_pairs: List of {"instruction": "...", "output": "..."}
        
        Returns:
            Number of chunks indexed
        """
        
        self.chunks = []
        self.chunk_metadata = []
        self.documents = qa_pairs  # Store for compatibility
        
        for i, item in enumerate(qa_pairs):
            question = item.get('instruction', '').strip()
            answer = item.get('output', '').strip()
            doc_id = item.get('doc_id', 'unknown')  # Track document
            
            if not question or not answer:
                continue
            
            # Split answer into sentences for better matching
            # This way: "Ali is an engineer. He likes coding." becomes 2 chunks
            sentences = re.split(r'[.!?]+', answer)
            
            for sent_idx, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if len(sentence) > 10:  # Only meaningful sentences
                    # Create chunk: question + single sentence
                    chunk_text = f"{question} {sentence}"
                    
                    self.chunks.append(chunk_text)
                    self.chunk_metadata.append({
                        'index': i,
                        'question': question,
                        'answer': sentence,  # Just this sentence, not whole answer
                        'sentence_idx': sent_idx,
                        'doc_id': doc_id  # Store document ID
                    })
        
        return len(self.chunks)
    
    # ===================== STEP 2: SIMPLE EMBEDDINGS =====================
    
    def _simple_embedding(self, text: str) -> List[float]:
        """
        Create simple embedding (TF-IDF like).
        
        Real RAG uses: Sentence-BERT, OpenAI, etc.
        This is simplified for demonstration.
        """
        
        # Clean text
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        
        # Create simple vector (word frequency)
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return as list
        return word_freq
    
    def _similarity_score(self, embedding1: dict, embedding2: dict) -> float:
        """
        Calculate similarity between two embeddings.
        Simple but effective: count how many query keywords appear in document.
        """
        
        if not embedding1 or not embedding2:
            return 0.0
        
        query_words = set(embedding1.keys())
        doc_words = set(embedding2.keys())
        common = query_words & doc_words
        
        if not query_words:
            return 0.0
        
        # Simple scoring: percentage of query words found
        # If query has 5 words and 3 are in document = 0.6 score
        coverage = len(common) / len(query_words)
        
        # Weight by frequency
        if common:
            total_query_freq = sum(embedding1.values())
            total_doc_freq = sum(embedding2.values())
            
            freq_score = 0.0
            for word in common:
                # How much this word matters in the query
                query_importance = embedding1[word] / total_query_freq
                # How much this word appears in the document
                doc_frequency = embedding2[word] / total_doc_freq
                
                freq_score += query_importance * doc_frequency
            
            # Combine coverage and frequency
            final_score = (coverage * 0.6) + (freq_score * 0.4)
        else:
            final_score = 0.0
        
        return min(final_score, 1.0)
    
    # ===================== STEP 3: RETRIEVAL =====================
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve top_k most similar chunks with strict filtering.
        
        Args:
            query: User question
            top_k: Number of results to return
        
        Returns:
            List of highly relevant chunks only
        """
        
        if not self.chunks:
            return []
        
        # Embed query
        query_embedding = self._simple_embedding(query)
        
        # Score all chunks
        scores = []
        for i, chunk in enumerate(self.chunks):
            chunk_embedding = self._simple_embedding(chunk)
            score = self._similarity_score(query_embedding, chunk_embedding)
            scores.append((score, i))
        
        # Filter by relevance: 0.2 minimum (at least 20% match quality)
        min_score = 0.2
        scores = [(s, i) for s, i in scores if s >= min_score]
        
        # Sort by score descending
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Return top_k with highest scores (limit results)
        results = []
        for score, idx in scores[:top_k]:
            results.append({
                'score': score,
                'question': self.chunk_metadata[idx]['question'],
                'answer': self.chunk_metadata[idx]['answer'],
                'full_text': self.chunks[idx],
                'source': f"Item {idx + 1}",
                'doc_id': self.chunk_metadata[idx].get('doc_id', 'unknown')  # Include doc_id
            })
        
        return results
    
    # ===================== STEP 4: GENERATION (with context) =====================
    
    def generate_response(self, query: str, model_response: str = None) -> str:
        """
        Generate response using retrieved context.
        
        Args:
            query: User question
            model_response: Optional model response (if no model, use retrieval)
        
        Returns:
            Enhanced response with context
        """
        
        # Retrieve context
        context_items = self.retrieve(query, top_k=3)
        
        if not context_items:
            return model_response or "No answer found."
        
        # Build context string
        context = "Context retrieved:\n\n"
        for item in context_items:
            if item['score'] > 0.1:  # Only high-confidence results
                context += f"- {item['answer']}\n"
        
        # If we have a model response, enhance it
        if model_response:
            return f"{model_response}\n\nBased on knowledge base:\n{context}"
        
        # Otherwise, use best match
        best = context_items[0]
        return f"Q: {best['question']}\nA: {best['answer']}"
    
    # ===================== UTILITY =====================
    
    def get_stats(self) -> Dict:
        """Get RAG statistics."""
        return {
            'indexed_chunks': len(self.chunks),
            'total_words': sum(len(c.split()) for c in self.chunks),
            'avg_chunk_length': sum(len(c.split()) for c in self.chunks) / len(self.chunks) if self.chunks else 0
        }


# ===================== EXAMPLE: VECTOR-BASED RAG (Advanced) =====================

class AdvancedRAGManager(SimpleRAGManager):
    """
    Advanced RAG using real embeddings.
    Requires: pip install sentence-transformers
    
    Much better than simple embedding!
    """
    
    def __init__(self, data_dir: str = "data/fine_tune"):
        super().__init__(data_dir)
        self.embeddings_model = None
        self._load_embeddings_model()
    
    def _load_embeddings_model(self):
        """Load sentence embedding model (local)."""
        try:
            from sentence_transformers import SentenceTransformer
            # Use small, fast model
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            print("sentence-transformers not installed. Using simple RAG.")
            self.embeddings_model = None
    
    def _simple_embedding(self, text: str):
        """Override with real embeddings."""
        
        if not self.embeddings_model:
            return super()._simple_embedding(text)
        
        # Use real embeddings
        embedding = self.embeddings_model.encode(text)
        return embedding.tolist()  # Convert to list
    
    def _similarity_score(self, embedding1, embedding2) -> float:
        """Calculate cosine similarity for real embeddings."""
        
        if isinstance(embedding1, dict) or isinstance(embedding2, dict):
            return super()._similarity_score(embedding1, embedding2)
        
        # Cosine similarity for vectors
        import numpy as np
        
        if not isinstance(embedding1, list) or not isinstance(embedding2, list):
            return 0.0
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# ===================== TEST EXAMPLE =====================

if __name__ == "__main__":
    
    print("=" * 60)
    print("RAG DEMO - Simple Retrieval-Augmented Generation")
    print("=" * 60)
    
    # Sample Q&A data (from advanced extraction)
    sample_qa = [
        {
            "instruction": "What is Python?",
            "output": "Python is a high-level, interpreted programming language known for its simplicity and readability."
        },
        {
            "instruction": "How do you create a function in Python?",
            "output": "Use the 'def' keyword followed by function name and parameters: def my_function(param1, param2):"
        },
        {
            "instruction": "What are virtual environments?",
            "output": "Virtual environments are isolated Python environments that allow you to manage dependencies per project."
        },
        {
            "instruction": "What is a list in Python?",
            "output": "A list is an ordered, mutable collection that can hold multiple items of different types."
        },
        {
            "instruction": "Explain decorators in Python",
            "output": "Decorators are functions that wrap other functions to modify their behavior without changing the original code."
        },
    ]
    
    # Create RAG manager
    rag = SimpleRAGManager()
    
    # Index documents
    indexed = rag.index_documents(sample_qa)
    print(f"\nIndexed {indexed} chunks\n")
    
    # Example queries
    queries = [
        "What is Python?",
        "How to create functions?",
        "Tell me about environments",
    ]
    
    for query in queries:
        print(f"Query: {query}")
        print("-" * 60)
        
        # Retrieve
        results = rag.retrieve(query, top_k=2)
        
        for i, result in enumerate(results, 1):
            print(f"  [{i}] Score: {result['score']:.2f}")
            print(f"      Q: {result['question']}")
            print(f"      A: {result['answer'][:60]}...\n")
    
    # Stats
    print("\nRAG Stats:")
    stats = rag.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
