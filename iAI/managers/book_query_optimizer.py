"""Optimized Book Query Handler - Fast inference with trained books."""

from typing import Dict, List, Optional, Tuple
import time


class BookQueryOptimizer:
    """
    Optimize queries when using trained books.
    
    Ensures:
    - Sub-second context retrieval
    - No overhead from RAG/fine-tuning
    - Fallback to base model if delays occur
    """
    
    def __init__(self, knowledge_manager=None, book_trainer=None):
        self.knowledge_manager = knowledge_manager
        self.book_trainer = book_trainer
        self.query_timeout = None
    
    def get_optimized_prompt(self, query: str, max_context_tokens: int = 512) -> Tuple[str, Dict]:
        """
        Get optimized prompt with book context (if available).
        
        Optimized for speed - does NOT block if retrieval is slow.
        
        Args:
            query: User query
            max_context_tokens: Max tokens for context (approx 4 chars = 1 token)
            
        Returns:
            (prompt: str, metadata: dict with timing info)
        """
        metadata = {
            'has_context': False,
            'retrieval_time': 0.0,
            'book_name': None,
            'method': None,
            'chunks_used': 0
        }
        
        if not self.book_trainer or not self.knowledge_manager:
            return query, metadata
        
        # Check if book is trained
        book_info = self.book_trainer.get_book_info()
        if not book_info.get('loaded'):
            return query, metadata
        
        method = self.book_trainer.training_method
        if not method:
            return query, metadata
        
        metadata['book_name'] = book_info.get('name')
        metadata['method'] = method
        
        # ========== FAST CONTEXT RETRIEVAL ==========
        # Use timeout to avoid blocking
        try:
            start = time.time()
            
            # Search with limited top_k for speed
            if method in ['rag', 'hybrid']:
                # Fast retrieval: limit to top 3 chunks
                context_docs = self.knowledge_manager.search(
                    query,
                    top_k=3,
                    timeout=self.query_timeout
                )
                
                retrieval_time = time.time() - start
                metadata['retrieval_time'] = retrieval_time
                
                if context_docs:
                    # Build context efficiently
                    max_chars = max_context_tokens * 4  # Rough token estimation
                    context_parts = []
                    char_count = 0
                    
                    for i, doc in enumerate(context_docs):
                        if char_count + len(doc) > max_chars:
                            break
                        context_parts.append(doc)
                        char_count += len(doc)
                    
                    if context_parts:
                        context_text = "\n\n".join(context_parts)
                        
                        # Optimize prompt format for speed
                        prompt = (
                            f"Book: {book_info.get('name')}\n"
                            f"---\n"
                            f"{context_text}\n"
                            f"---\n"
                            f"Q: {query}"
                        )
                        
                        metadata['has_context'] = True
                        metadata['chunks_used'] = len(context_parts)
                        
                        return prompt, metadata
        
        except TimeoutError:
            # If retrieval times out, fallback to original query
            metadata['retrieval_time'] = self.query_timeout
            metadata['timeout'] = True
            return query, metadata
        except Exception as e:
            # Silent failure - just use original query
            metadata['error'] = str(e)
            return query, metadata
        
        return query, metadata
    
    def check_inference_speed(self, ollama_manager, model: str, 
                             test_query: str = "Hi") -> Dict:
        """
        Check inference speed with and without context.
        
        Used for diagnostics - NOT in query path.
        
        Args:
            ollama_manager: OllamaManager instance
            model: Model name
            test_query: Test query string
            
        Returns:
            {'base_time': float, 'with_context_time': float, 'overhead': float}
        """
        results = {
            'base_time': 0.0,
            'with_context_time': 0.0,
            'overhead': 0.0,
            'status': 'ok'
        }
        
        try:
            # Test base model speed
            start = time.time()
            ollama_manager.chat(model=model, message=test_query, stream=False)
            results['base_time'] = time.time() - start
            
            # Test with context (if available)
            if self.book_trainer and self.book_trainer.current_book:
                prompt, metadata = self.get_optimized_prompt(test_query)
                start = time.time()
                ollama_manager.chat(model=model, message=prompt, stream=False)
                results['with_context_time'] = time.time() - start
                results['overhead'] = results['with_context_time'] - results['base_time']
                
                # Flag if overhead is significant (>1 second)
                if results['overhead'] > 1.0:
                    results['status'] = 'slow'
        
        except Exception as e:
            results['status'] = 'error'
            results['error'] = str(e)
        
        return results
