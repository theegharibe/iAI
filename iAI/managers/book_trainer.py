"""Book Trainer - Train models from PDF/books with RAG and Fine-tuning."""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional
import hashlib
import time

try:
    import PyPDF2
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False


class BookTrainer:
    """
    Train models from books using RAG, Fine-tuning, or Hybrid approach.
    
    Integrates with KnowledgeManager (RAG) and LoRAFineTuner (Fine-tuning).
    """
    
    def __init__(self, knowledge_manager=None, lora_trainer=None):
        """
        Initialize BookTrainer.
        
        Args:
            knowledge_manager: KnowledgeManager instance for RAG
            lora_trainer: LoRAFineTuner instance for fine-tuning
        """
        self.knowledge_manager = knowledge_manager
        self.lora_trainer = lora_trainer
        
        self.data_dir = Path("data/fine_tune/books")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Current book state
        self.current_book = None
        self.current_book_name = None
        self.current_book_hash = None
        self.training_method = None
        self.training_progress = 0
        self.training_log = []
        self.training_status = "idle"
        
        # Embedder for RAG (LAZY LOADING)
        self.embedder = None
        self._embedder_loaded = False
    
    def _get_embedder(self):
        """Load embedder on first use (lazy loading)."""
        if not self._embedder_loaded:
            self._embedder_loaded = True
            if HAS_EMBEDDINGS:
                try:
                    print("[BookTrainer] Loading embedder model...")
                    self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e:
                    print(f"[BookTrainer] Error loading embedder: {e}")
        return self.embedder
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file."""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()[:16]
        except:
            return hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    
    def _update_progress(self, step: int, total_steps: int, message: str, 
                        callback: Optional[Callable] = None) -> None:
        """Update progress."""
        self.training_progress = int((step / total_steps) * 100)
        self.training_log.append(message)
        
        if callback:
            callback(self.training_progress, message)
    
    def upload_book(self, file_path: str) -> Dict:
        """
        Upload and verify book file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            {
                'success': bool,
                'message': str,
                'book_name': str,
                'book_hash': str,
                'text_length': int,
                'pages': int
            }
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {'success': False, 'message': 'File not found'}
            
            if file_path.suffix.lower() != '.pdf':
                return {'success': False, 'message': 'Only PDF files supported'}
            
            if not HAS_PYPDF:
                return {'success': False, 'message': 'PyPDF2 not installed'}
            
            # Extract text
            text, num_pages = self._extract_pdf_text(str(file_path))
            
            if not text or len(text) < 100:
                return {'success': False, 'message': 'PDF too short or empty'}
            
            # Store book info
            book_hash = self._get_file_hash(str(file_path))
            book_name = file_path.stem
            
            # Save PDF file
            self._save_uploaded_pdf(str(file_path), book_hash)
            
            self.current_book = text
            self.current_book_name = book_name
            self.current_book_hash = book_hash
            
            return {
                'success': True,
                'message': f'Book loaded: {book_name}',
                'book_name': book_name,
                'book_hash': book_hash,
                'text_length': len(text),
                'pages': num_pages
            }
        
        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)[:100]}'}
    
    def _save_uploaded_pdf(self, file_path: str, book_hash: str) -> bool:
        """
        Save uploaded PDF file.
        
        Args:
            file_path: Path to original file
            book_hash: Unique hash for book
            
        Returns:
            True if successful
        """
        try:
            source = Path(file_path)
            dest_dir = self.data_dir / "uploaded_pdfs"
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Save with hash naming
            dest = dest_dir / f"{book_hash}.pdf"
            
            # Save if not already saved
            if not dest.exists():
                import shutil
                shutil.copy2(str(source), str(dest))
                print(f"[BookTrainer] [OK] PDF saved: {dest}")
            
            return True
        except Exception as e:
            print(f"[BookTrainer] [ERROR] Failed to save PDF: {e}")
            return False
    
    def _extract_pdf_text(self, file_path: str) -> tuple:
        """
        Extract text from PDF.
        
        Returns:
            (text: str, num_pages: int)
        """
        text = []
        
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        
        return '\n'.join(text), num_pages
    
    def train_with_rag(self, method_name: str = "rag", 
                      progress_callback: Optional[Callable] = None) -> Dict:
        """
        Train using RAG (Knowledge Base).
        
        Args:
            method_name: Method identifier
            progress_callback: Callback(progress: int, message: str)
            
        Returns:
            {'success': bool, 'message': str}
        """
        try:
            if not self.current_book:
                return {'success': False, 'message': 'No book loaded'}
            
            if not self.knowledge_manager:
                return {'success': False, 'message': 'Knowledge manager not available'}
            
            self.training_status = "running"
            self.training_progress = 0
            self.training_log = []
            
            # Step 1: Chunk text (10%)
            self._update_progress(1, 5, "Chunking text...", progress_callback)
            chunks = self._chunk_text(self.current_book, chunk_size=512, overlap=50)
            
            if not chunks:
                return {'success': False, 'message': 'Could not chunk text'}
            
            # Step 2: Add to knowledge base (90%)
            self._update_progress(2, 5, f"Adding {len(chunks)} chunks to KB...", 
                                 progress_callback)
            
            for i, chunk in enumerate(chunks):
                try:
                    self.knowledge_manager.add_document(
                        text=chunk,
                        metadata={
                            'source': self.current_book_name,
                            'chunk': i,
                            'book_hash': self.current_book_hash
                        }
                    )
                except:
                    pass
                
                # Update progress
                if (i + 1) % 10 == 0:
                    progress = 20 + (i / len(chunks)) * 80
                    self._update_progress(int(progress/20), 5, 
                                        f"Processed {i+1}/{len(chunks)} chunks", 
                                        progress_callback)
            
            # Step 3: Complete (100%)
            self._update_progress(5, 5, "RAG training complete!", progress_callback)
            
            self.training_status = "completed"
            
            return {
                'success': True,
                'message': f'RAG training complete. Added {len(chunks)} chunks.',
                'method': 'rag',
                'chunks_added': len(chunks)
            }
        
        except Exception as e:
            self.training_status = "error"
            return {'success': False, 'message': f'Error: {str(e)[:100]}'}
    
    def train_with_finetuning(self, progress_callback: Optional[Callable] = None) -> Dict:
        """
        Train using Fine-tuning (LoRA).
        
        Args:
            progress_callback: Callback(progress: int, message: str)
            
        Returns:
            {'success': bool, 'message': str}
        """
        try:
            if not self.current_book:
                return {'success': False, 'message': 'No book loaded'}
            
            if not self.lora_trainer:
                return {'success': False, 'message': 'LoRA trainer not available'}
            
            self.training_status = "running"
            self.training_progress = 0
            self.training_log = []
            
            # Step 1: Generate Q&A pairs (20%)
            self._update_progress(1, 5, "Generating Q&A pairs...", progress_callback)
            qa_pairs = self._generate_qa_pairs(self.current_book)
            
            if len(qa_pairs) < 5:
                return {'success': False, 'message': 'Not enough Q&A pairs generated'}
            
            # Step 2: Save training data (25%)
            self._update_progress(2, 5, "Saving training data...", progress_callback)
            training_file = self.data_dir / f"{self.current_book_hash}_training.json"
            
            with open(training_file, 'w', encoding='utf-8') as f:
                json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
            
            # Step 3: Start training (80%)
            self._update_progress(3, 5, "Starting LoRA training...", progress_callback)
            result = self.lora_trainer.train_from_documents(
                documents=[str(training_file)],
                epochs=3,
                batch_size=4,
                learning_rate=5e-4
            )
            
            if not result.get('success'):
                return result
            
            # Step 4: Merge model (95%)
            self._update_progress(4, 5, "Merging model...", progress_callback)
            merge_result = self.lora_trainer.merge_model()
            
            if not merge_result.get('success'):
                return merge_result
            
            # Step 5: Complete (100%)
            self._update_progress(5, 5, "Fine-tuning complete!", progress_callback)
            
            self.training_status = "completed"
            
            return {
                'success': True,
                'message': f'Fine-tuning complete. Generated {len(qa_pairs)} Q&A pairs.',
                'method': 'finetuning',
                'qa_pairs': len(qa_pairs),
                'output_path': merge_result.get('output_path')
            }
        
        except Exception as e:
            self.training_status = "error"
            return {'success': False, 'message': f'Error: {str(e)[:100]}'}
    
    def train_hybrid(self, progress_callback: Optional[Callable] = None) -> Dict:
        """
        Train using both RAG and Fine-tuning (Hybrid).
        
        Args:
            progress_callback: Callback(progress: int, message: str)
            
        Returns:
            {'success': bool, 'message': str}
        """
        try:
            # First RAG (50% of progress)
            def rag_callback(p, msg):
                if progress_callback:
                    progress_callback(int(p * 0.5), f"RAG: {msg}")
            
            rag_result = self.train_with_rag(progress_callback=rag_callback)
            
            if not rag_result.get('success'):
                return rag_result
            
            # Then Fine-tuning (remaining 50%)
            def ft_callback(p, msg):
                if progress_callback:
                    progress_callback(50 + int(p * 0.5), f"Fine-tuning: {msg}")
            
            ft_result = self.train_with_finetuning(progress_callback=ft_callback)
            
            if not ft_result.get('success'):
                return ft_result
            
            self.training_status = "completed"
            
            return {
                'success': True,
                'message': 'Hybrid training complete.',
                'method': 'hybrid',
                'rag': rag_result,
                'finetuning': ft_result
            }
        
        except Exception as e:
            self.training_status = "error"
            return {'success': False, 'message': f'Error: {str(e)[:100]}'}
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Chunk text into overlapping pieces.
        
        Args:
            text: Full text
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of chunks
        """
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            if len(chunk) > 50:
                chunks.append(chunk)
        
        return chunks
    
    def _generate_qa_pairs(self, text: str, max_pairs: int = 50) -> List[Dict]:
        """
        Generate Q&A pairs from text.
        
        Args:
            text: Source text
            max_pairs: Maximum pairs to generate
            
        Returns:
            List of {instruction, input, output}
        """
        pairs = []
        
        # Split into sentences
        sentences = text.replace('\n', ' ').split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        # Create Q&A from consecutive sentences
        for i in range(0, len(sentences) - 1, 2):
            if len(pairs) >= max_pairs:
                break
            
            question = sentences[i][:100]
            answer = sentences[i+1] if i+1 < len(sentences) else sentences[i]
            
            if len(question) > 10 and len(answer) > 10:
                pairs.append({
                    'instruction': question + '?',
                    'input': '',
                    'output': answer
                })
        
        return pairs
    
    def get_training_status(self) -> Dict:
        """Get current training status."""
        return {
            'status': self.training_status,
            'progress': self.training_progress,
            'log': self.training_log[-5:],
            'book': self.current_book_name,
            'method': self.training_method
        }
    
    def set_training_method(self, method: str) -> None:
        """Set training method (rag/finetuning/hybrid)."""
        if method in ['rag', 'finetuning', 'hybrid']:
            self.training_method = method
    
    def get_book_info(self) -> Dict:
        """Get current book information."""
        if not self.current_book:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'name': self.current_book_name,
            'hash': self.current_book_hash,
            'size': len(self.current_book),
            'pages': len(self.current_book.split('\n\n'))
        }
