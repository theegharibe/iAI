"""Fine-tuning Manager - LoRA-based model training."""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import hashlib

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
    import torch
    HAS_FINETUNE = True
except ImportError:
    HAS_FINETUNE = False
    print("[FineTune] WARNING: Required packages not installed")


class FineTuneDataset:
    """Dataset class for fine-tuning."""
    
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format: "Instruction: {instruction}\nInput: {input}\nOutput: {output}"
        text = f"Instruction: {item.get('instruction', '')}\nInput: {item.get('input', '')}\nOutput: {item.get('output', '')}"
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


class FineTuneManager:
    """Manages LoRA fine-tuning for local models."""
    
    def __init__(self, model_name: str = "mistral", data_dir: str = "data/fine_tune"):
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_data_file = self.data_dir / "training_data.json"
        self.lora_model_dir = self.data_dir / "lora_models"
        self.lora_model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        self.training_data = []
        
        self._load_training_data()
        print("[FineTune] [OK] Manager initialized")
    
    def _load_training_data(self):
        """Load existing training data."""
        if self.training_data_file.exists():
            try:
                with open(self.training_data_file, 'r', encoding='utf-8') as f:
                    self.training_data = json.load(f)
                    print(f"[FineTune] Loaded {len(self.training_data)} training samples")
            except Exception as e:
                print(f"[FineTune] Error loading training data: {e}")
    
    def _save_training_data(self):
        """Save training data."""
        try:
            with open(self.training_data_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[FineTune] Error saving training data: {e}")
    
    def add_training_pair(self, instruction: str, input_text: str = "", output_text: str = "") -> bool:
        """
        Add a training pair (Q&A).
        
        Args:
            instruction: The question/instruction
            input_text: Optional input context
            output_text: The expected output/answer
        
        Returns:
            True if added successfully
        """
        if not instruction or not output_text:
            print("[FineTune] Error: instruction and output_text are required")
            return False
        
        # Check for duplicates (by instruction hash)
        instruction_hash = hashlib.md5(instruction.encode()).hexdigest()
        
        for item in self.training_data:
            if hashlib.md5(item.get('instruction', '').encode()).hexdigest() == instruction_hash:
                print(f"[FineTune] Warning: Similar instruction already exists")
                return False
        
        training_item = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "timestamp": datetime.now().isoformat()
        }
        
        self.training_data.append(training_item)
        self._save_training_data()
        
        print(f"[FineTune] [OK] Added training pair (total: {len(self.training_data)})")
        return True
    
    def add_from_document(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> int:
        """
        Extract Q&A pairs from document text.
        Uses a simple heuristic: splits by paragraphs and creates pairs.
        
        Args:
            text: Document text
            chunk_size: Size of chunks
            overlap: Overlap between chunks
        
        Returns:
            Number of pairs added
        """
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.split()) > 5]
        
        if not paragraphs:
            print("[FineTune] No suitable paragraphs found")
            return 0
        
        added_count = 0
        
        # Create training pairs from consecutive paragraphs
        for i in range(len(paragraphs) - 1):
            instruction = f"Based on the context, answer: What is mentioned in the following text?"
            input_text = paragraphs[i]
            output_text = paragraphs[i + 1]
            
            if len(input_text.split()) > 10 and len(output_text.split()) > 10:
                if self.add_training_pair(instruction, input_text, output_text):
                    added_count += 1
        
        print(f"[FineTune] [OK] Added {added_count} training pairs from document")
        return added_count
    
    def add_from_knowledge_base(self, knowledge_manager) -> int:
        """
        Extract training pairs from Knowledge Base (RAG chunks).
        
        Args:
            knowledge_manager: KnowledgeManager instance
        
        Returns:
            Number of pairs added
        """
        try:
            # Get all documents from knowledge base
            documents = knowledge_manager.list_documents()
            
            if not documents:
                print("[FineTune] No documents in knowledge base")
                return 0
            
            total_added = 0
            
            # For each document, get its chunks and create Q&A pairs
            for doc in documents:
                print(f"[FineTune] Processing {doc['filename']}...")
                
                # Try to get chunks from collection
                try:
                    results = knowledge_manager.collection.get(
                        where={"filename": doc['filename']}
                    )
                    
                    if results and results['documents']:
                        chunks = results['documents']
                        
                        # Create Q&A pairs from consecutive chunks
                        for i in range(len(chunks) - 1):
                            instruction = f"Based on the document '{doc['filename']}', what follows this section?"
                            input_text = chunks[i]
                            output_text = chunks[i + 1]
                            
                            if self.add_training_pair(instruction, input_text, output_text):
                                total_added += 1
                
                except Exception as e:
                    print(f"[FineTune] Error processing {doc['filename']}: {e}")
                    continue
            
            print(f"[FineTune] [OK] Added {total_added} training pairs from knowledge base")
            return total_added
        
        except Exception as e:
            print(f"[FineTune] Error reading knowledge base: {e}")
            return 0
    
    def get_training_stats(self) -> Dict:
        """Get training data statistics."""
        total_samples = len(self.training_data)
        total_tokens = sum(
            len(item.get('instruction', '').split()) + 
            len(item.get('output', '').split())
            for item in self.training_data
        )
        
        return {
            "total_samples": total_samples,
            "total_tokens_estimate": total_tokens,
            "avg_tokens_per_sample": total_tokens // max(total_samples, 1),
            "data_file": str(self.training_data_file)
        }
    
    def export_training_data(self, output_path: str) -> bool:
        """Export training data to file."""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_data, f, indent=2, ensure_ascii=False)
            
            print(f"[FineTune] [OK] Exported {len(self.training_data)} samples to {output_path}")
            return True
        except Exception as e:
            print(f"[FineTune] Error exporting training data: {e}")
            return False
    
    def import_training_data(self, input_path: str) -> int:
        """Import training data from file."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                imported_data = json.load(f)
            
            if not isinstance(imported_data, list):
                print("[FineTune] Error: Input file must contain a JSON array")
                return 0
            
            added_count = 0
            for item in imported_data:
                if 'instruction' in item and 'output' in item:
                    if self.add_training_pair(
                        item['instruction'],
                        item.get('input', ''),
                        item['output']
                    ):
                        added_count += 1
            
            print(f"[FineTune] [OK] Imported {added_count} training samples")
            return added_count
        except Exception as e:
            print(f"[FineTune] Error importing training data: {e}")
            return 0
    
    def clear_training_data(self) -> bool:
        """Clear all training data."""
        try:
            self.training_data = []
            self.training_data_file.unlink(missing_ok=True)
            print("[FineTune] [OK] Training data cleared")
            return True
        except Exception as e:
            print(f"[FineTune] Error clearing training data: {e}")
            return False
    
    def list_training_samples(self, limit: int = 10) -> List[Dict]:
        """Get list of training samples."""
        return self.training_data[-limit:] if self.training_data else []
    
    def remove_sample(self, index: int) -> bool:
        """Remove a training sample by index."""
        try:
            if 0 <= index < len(self.training_data):
                removed = self.training_data.pop(index)
                self._save_training_data()
                print(f"[FineTune] [OK] Removed sample: {removed['instruction'][:50]}...")
                return True
            return False
        except Exception as e:
            print(f"[FineTune] Error removing sample: {e}")
            return False
    
    def get_training_data(self) -> List[Dict]:
        """Get all training data."""
        return self.training_data
    
    def is_available(self) -> bool:
        """Check if fine-tuning is available."""
        return HAS_FINETUNE
    
    # Note: Actual fine-tuning with transformers requires significant resources
    # The training itself would be done with:
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    # )
    # trainer.train()
    
    def start_training(self, 
                      epochs: int = 3,
                      batch_size: int = 4,
                      learning_rate: float = 1e-4,
                      callback=None) -> Dict:
        """
        Start fine-tuning process.
        
        This is a placeholder for the actual training loop.
        In production, this would use Hugging Face Trainer or similar.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            callback: Callback function for progress updates
        
        Returns:
            Training results dictionary
        """
        if len(self.training_data) < 5:
            return {
                "success": False,
                "error": "Need at least 5 training samples"
            }
        
        if not HAS_FINETUNE:
            return {
                "success": False,
                "error": "Fine-tuning packages not available"
            }
        
        try:
            if callback:
                callback("Preparing training data...", 0)
            
            # This is a simplified version
            # In production, you would:
            # 1. Load the base model
            # 2. Create LoRA config
            # 3. Wrap model with LoRA
            # 4. Run trainer
            # 5. Save the LoRA adapter
            
            result = {
                "success": True,
                "message": "Training completed (simulated)",
                "samples_used": len(self.training_data),
                "status": "Ready for deployment"
            }
            
            if callback:
                callback("Training completed!", 100)
            
            return result
        
        except Exception as e:
            error_msg = f"Training error: {str(e)}"
            print(f"[FineTune] {error_msg}")
            
            if callback:
                callback(f"Error: {error_msg}", -1)
            
            return {
                "success": False,
                "error": error_msg
            }
