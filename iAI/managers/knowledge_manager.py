"""Knowledge Base Manager - RAG System for iAI"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import hashlib
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False
    print("[Knowledge] WARNING: chromadb not installed. Run: pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("[Knowledge] WARNING: sentence-transformers not installed")

try:
    import PyPDF2
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False


class KnowledgeManager:
    """Manages document upload, processing, and retrieval."""
    
    def __init__(self, data_dir: str = "data/knowledge_base"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.data_dir / "metadata.json"
        self.documents = self._load_metadata()
        
        # Initialize ChromaDB - LAZY LOADING
        self._client = None
        self._collection = None
        self._embedder = None
        self._initialized = False
        
        print("[Knowledge] [OK] Manager initialized (ChromaDB lazy-loaded)")
    
    def _ensure_initialized(self):
        """Initialize ChromaDB on first use (lazy loading)."""
        if self._initialized:
            return
        
        self._initialized = True
        
        if not HAS_CHROMA or not HAS_EMBEDDINGS:
            print("[Knowledge] RAG system disabled (missing dependencies)")
            return
        
        try:
            self._client = chromadb.PersistentClient(
                path=str(self.data_dir / "chroma_db")
            )
            self._collection = self._client.get_or_create_collection(
                name="knowledge_base",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Load embedding model (small and fast)
            print("[Knowledge] Loading embedder model...")
            self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("[Knowledge] [OK] RAG system ready")
        except Exception as e:
            print(f"[Knowledge] Error initializing: {e}")
    
    @property
    def client(self):
        """Get ChromaDB client (lazy loaded)."""
        self._ensure_initialized()
        return self._client
    
    @property
    def collection(self):
        """Get ChromaDB collection (lazy loaded)."""
        self._ensure_initialized()
        return self._collection
    
    @property
    def embedder(self):
        """Get embedder model (lazy loaded)."""
        self._ensure_initialized()
        return self._embedder
    
    def _load_metadata(self) -> Dict:
        """Load document metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_metadata(self):
        """Save document metadata."""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.documents, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Knowledge] Error saving metadata: {e}")
    
    def is_available(self) -> bool:
        """Check if RAG system is available."""
        return HAS_CHROMA and HAS_EMBEDDINGS and self.collection is not None
    
    def get_file_hash(self, file_path: str) -> str:
        """Generate hash for file content."""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from various file formats."""
        path = Path(file_path)
        ext = path.suffix.lower()
        
        try:
            # TXT
            if ext == '.txt':
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            # PDF
            elif ext == '.pdf' and HAS_PDF:
                text = []
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text.append(page.extract_text())
                return '\n'.join(text)
            
            # DOCX
            elif ext in ['.docx', '.doc'] and HAS_DOCX:
                doc = DocxDocument(file_path)
                return '\n'.join([para.text for para in doc.paragraphs])
            
            else:
                return None
        
        except Exception as e:
            print(f"[Knowledge] Error extracting {ext}: {e}")
            return None
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def add_document(self, file_path: str, model_name: str = "general") -> Dict:
        """
        Process and add document to knowledge base.
        
        Returns:
            Dict with status and statistics
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "RAG system not available"
            }
        
        path = Path(file_path)
        file_hash = self.get_file_hash(file_path)
        
        # Check if already processed
        if file_hash in self.documents:
            return {
                "success": False,
                "error": "Document already exists"
            }
        
        # Extract text
        print(f"[Knowledge] Extracting text from {path.name}...")
        text = self.extract_text(file_path)
        
        if not text:
            return {
                "success": False,
                "error": f"Cannot extract text from {path.suffix}"
            }
        
        # Split into chunks
        print(f"[Knowledge] Chunking text...")
        chunks = self.chunk_text(text)
        
        if not chunks:
            return {
                "success": False,
                "error": "No text content found"
            }
        
        # Generate embeddings and store
        print(f"[Knowledge] Generating embeddings for {len(chunks)} chunks...")
        try:
            embeddings = self.embedder.encode(chunks, show_progress_bar=True)
            
            # Store in ChromaDB
            ids = [f"{file_hash}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "filename": path.name,
                    "file_hash": file_hash,
                    "chunk_id": i,
                    "model": model_name,
                    "timestamp": datetime.now().isoformat()
                }
                for i in range(len(chunks))
            ]
            
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=chunks,
                metadatas=metadatas
            )
            
            # Save metadata
            self.documents[file_hash] = {
                "filename": path.name,
                "model": model_name,
                "chunks": len(chunks),
                "size": path.stat().st_size,
                "added_at": datetime.now().isoformat()
            }
            self._save_metadata()
            
            print(f"[Knowledge] [OK] Added {path.name} ({len(chunks)} chunks)")
            
            return {
                "success": True,
                "filename": path.name,
                "chunks": len(chunks),
                "size": path.stat().st_size
            }
        
        except Exception as e:
            print(f"[Knowledge] Error processing document: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }
    
    def search(self, query: str, model_name: str = "general", top_k: int = 3, 
              timeout: float = None) -> List[str]:
        """
        Search knowledge base for relevant chunks.
        
        Args:
            query: Search query
            model_name: Filter by model (default "general")
            top_k: Number of results (default 3 for speed)
            timeout: Ignored (kept for compatibility)
            
        Returns:
            List of relevant text chunks
        """
        if not self.is_available():
            return []
        
        try:
            # Generate query embedding (fast model)
            query_embedding = self.embedder.encode([query])[0]
            
            # Search with limit for speed
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=min(top_k, 5),  # Cap at 5 for speed
                where={"model": model_name} if model_name != "general" else None
            )
            
            if results and results['documents']:
                return results['documents'][0]
            
            return []
        
        except Exception as e:
            print(f"[Knowledge] Search error: {e}")
            return []
    
    def remove_document(self, file_hash: str) -> bool:
        """Remove document from knowledge base."""
        if file_hash not in self.documents:
            return False
        
        try:
            # Get all chunk IDs for this document
            results = self.collection.get(
                where={"file_hash": file_hash}
            )
            
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
            
            # Remove metadata
            filename = self.documents[file_hash]['filename']
            del self.documents[file_hash]
            self._save_metadata()
            
            print(f"[Knowledge] [OK] Removed {filename}")
            return True
        
        except Exception as e:
            print(f"[Knowledge] Error removing document: {e}")
            return False
    
    def list_documents(self) -> List[Dict]:
        """Get list of all documents."""
        docs = []
        for file_hash, info in self.documents.items():
            docs.append({
                "hash": file_hash,
                "filename": info['filename'],
                "model": info.get('model', 'general'),
                "chunks": info['chunks'],
                "size": info['size'],
                "added_at": info['added_at']
            })
        return sorted(docs, key=lambda x: x['added_at'], reverse=True)
    
    def get_stats(self) -> Dict:
        """Get knowledge base statistics."""
        total_docs = len(self.documents)
        total_chunks = sum(doc['chunks'] for doc in self.documents.values())
        total_size = sum(doc['size'] for doc in self.documents.values())
        
        return {
            "documents": total_docs,
            "chunks": total_chunks,
            "size_mb": round(total_size / (1024 * 1024), 2),
            "available": self.is_available()
        }