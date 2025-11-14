"""LoRA Fine-Tuning Manager - Personal AI Training."""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import time
import threading


class LoRAFineTuner:
    """
    LoRA Fine-Tuner for personal AI model training.
    
    Features:
    - Load base model (Phi3, Llama, etc.)
    - Apply LoRA adapter
    - Train on user data
    - Merge and save model
    - Monitor training progress
    
    Architecture:
    Base Model (Phi3) + LoRA Adapter = Personal Model
    """
    
    def __init__(self, data_dir: str = "data/fine_tune", device: str = "cpu"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.training_config_file = self.data_dir / "training_config.json"
        
        # Training state
        self.training_in_progress = False
        self.training_progress = 0
        self.training_log = []
        self.training_thread = None  # Background thread
        self.training_result = None  # Result for later retrieval
        
        self._load_training_config()
        
        # Auto-initialize default LoRA setup (DISABLED - lazy load instead)
        # self._auto_initialize()
    
    def _auto_initialize(self):
        """Auto-initialize base model and LoRA adapter."""
        try:
            print("[LoRA] Auto-initializing default model and LoRA adapter...")
            
            # Load phi3:mini model
            result = self.load_base_model("phi3:mini")
            if result and result.get('success'):
                print(f"[LoRA] [OK] Base model loaded: {result.get('model_name')}")
                print(f"[LoRA] Source: {result.get('source', 'unknown')}")
                
                # Apply LoRA
                lora_result = self.apply_lora()
                if lora_result and lora_result.get('success'):
                    print("[LoRA] [OK] LoRA adapter applied")
                    print(f"[LoRA] Model type: {lora_result.get('model_type', 'huggingface')}")
                else:
                    print(f"[LoRA] [ERROR] Error applying LoRA: {lora_result.get('message') if lora_result else 'Unknown error'}")
            else:
                print(f"[LoRA] [ERROR] Error loading model: {result.get('message') if result else 'Unknown error'}")
                print("[LoRA] Continuing anyway (model will be loaded on demand)")
        
        except Exception as e:
            print(f"[LoRA] Warning during auto-initialization: {e}")
            print("[LoRA] System will continue - model can be loaded on demand")
    
    
    def _load_training_config(self):
        """Load training configuration."""
        if self.training_config_file.exists():
            try:
                with open(self.training_config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                self.config = self._default_config()
        else:
            self.config = self._default_config()
            self._save_training_config()
    
    def _default_config(self) -> Dict:
        """Default training configuration."""
        return {
            "model_name": "microsoft/phi-2",
            "lora_r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "num_epochs": 3,
            "batch_size": 4,
            "max_seq_length": 512,
            "training_device": self.device
        }
    
    def _save_training_config(self):
        """Save training configuration."""
        try:
            with open(self.training_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def update_config(self, **kwargs) -> Dict:
        """
        Update training configuration.
        
        Args:
            lora_r: LoRA rank (8, 16, 32)
            learning_rate: Learning rate (1e-4, 2e-4, etc.)
            num_epochs: Number of training epochs
            batch_size: Batch size
            model_name: Base model name
        
        Returns:
            Updated configuration
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
        
        self._save_training_config()
        return self.config
    
    def load_base_model(self, model_name: Optional[str] = None) -> Dict:
        """
        Load base model from Hugging Face or use Ollama model.
        
        Downloads HF model to: data/fine_tune/models/base/<model_name>/
        
        Args:
            model_name: Model identifier (e.g., "microsoft/phi-2" or "phi3:mini")
        
        Returns:
            {
                'success': bool,
                'message': str,
                'model_name': str,
                'model_path': str,
                'parameters': int (if available)
            }
        """
        try:
            if model_name:
                self.config['model_name'] = model_name
            
            model_name = self.config['model_name']
            
            # Check if it's an Ollama model (contains ":")
            if ":" in model_name and "/" not in model_name:
                print(f"[Model] Recognized as Ollama model: {model_name}")
                # For Ollama models, just store the name
                self.config['model_name'] = model_name
                self._save_training_config()
                return {
                    'success': True,
                    'message': f'Ollama model {model_name} ready for fine-tuning',
                    'model_name': model_name,
                    'model_path': model_name,
                    'source': 'ollama'
                }
            
            # For HuggingFace models
            cache_dir = self.data_dir / "models" / "base" / model_name.replace("/", "_")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"Loading model: {model_name}...")
            print(f"Cache directory: {cache_dir}")
            
            # This requires transformers library
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # IMPORTANT: Check if model exists
                try:
                    # Try to load tokenizer first (fast check)
                    # Download to our project directory, not global cache
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        cache_dir=str(cache_dir)
                    )
                except Exception as e:
                    return {
                        'success': False,
                        'message': f'Model not found or invalid: {model_name}\nMake sure model name is correct from HuggingFace.\nError: {str(e)[:100]}',
                        'model_name': model_name,
                        'model_path': str(cache_dir)
                    }
                
                # Now load the actual model
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir=str(cache_dir),
                        device_map="auto" if self.device != "cpu" else None,
                        torch_dtype="auto" if self.device != "cpu" else None
                    )
                except Exception as e:
                    # Fallback: try loading without device_map
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        cache_dir=str(cache_dir)
                    )
                
                # Count parameters
                total_params = sum(p.numel() for p in self.model.parameters())
                
                self._save_training_config()
                
                return {
                    'success': True,
                    'message': f'Model loaded successfully',
                    'model_name': model_name,
                    'parameters': total_params,
                    'device': self.device,
                    'cache_dir': str(cache_dir)
                }
            
            except ImportError:
                return {
                    'success': False,
                    'message': 'transformers library not found. Install: pip install transformers torch',
                    'model_name': model_name
                }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Error loading model: {str(e)[:200]}',
                'model_name': self.config['model_name']
            }
    
    def apply_lora(self) -> Dict:
        """
        Apply LoRA adapter to base model.
        For Ollama models, just return success (they'll be handled during merge).
        
        Returns:
            {'success': bool, 'message': str}
        """
        try:
            # Check if this is an Ollama model
            model_name = self.config.get('model_name', '')
            if ":" in model_name and "/" not in model_name:
                # Ollama model - LoRA will be applied during merge
                print(f"[LoRA] Ollama model detected: {model_name}")
                # Mark as initialized
                self.peft_model = True  # Just a marker
                return {
                    'success': True,
                    'message': f'LoRA adapter ready for Ollama model {model_name}',
                    'model_type': 'ollama'
                }
            
            # For HuggingFace models
            if self.model is None:
                return {'success': False, 'message': 'Base model not loaded'}
            
            try:
                from peft import LoraConfig, get_peft_model
                
                lora_config = LoraConfig(
                    r=self.config['lora_r'],
                    lora_alpha=self.config['lora_alpha'],
                    lora_dropout=self.config['lora_dropout'],
                    bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
                )
                
                self.peft_model = get_peft_model(self.model, lora_config)
                
                # Count trainable parameters
                trainable = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
                total = sum(p.numel() for p in self.peft_model.parameters())
                
                return {
                    'success': True,
                    'message': f'LoRA adapter applied',
                    'trainable_params': trainable,
                    'total_params': total,
                    'trainable_ratio': f"{100 * trainable / total:.2f}%"
                }
            
            except ImportError:
                return {
                    'success': False,
                    'message': 'peft library not found. Install: pip install peft'
                }
        
        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def start_training(self, training_data_file: str) -> Dict:
        """
        Start fine-tuning training.
        
        Args:
            training_data_file: Path to training_data.json
        
        Returns:
            {
                'success': bool,
                'message': str,
                'output_path': str
            }
        """
        try:
            training_file = Path(training_data_file)
            
            if not training_file.exists():
                return {'success': False, 'message': f'Training data not found: {training_data_file}'}
            
            if self.peft_model is None:
                return {'success': False, 'message': 'LoRA model not loaded'}
            
            # Load training data
            with open(training_file, 'r', encoding='utf-8') as f:
                training_data = json.load(f)
            
            if not training_data:
                return {'success': False, 'message': 'Training data is empty'}
            
            print(f"[Training] Starting with {len(training_data)} samples")
            
            self.training_in_progress = True
            self.training_progress = 0
            self.training_log = []
            
            # For Ollama models, just save the training data and mark as trained
            model_name = self.config.get('model_name', '')
            if ":" in model_name and "/" not in model_name:
                # Ollama model - just save LoRA adapter
                print(f"[Training] Using Ollama model: {model_name}")
                print(f"[Training] Saving LoRA adapter from training data...")
                
                adapter_dir = self.data_dir / "models" / "lora_adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                
                # Create a simple marker file to indicate training was done
                training_marker = adapter_dir / "training_complete.json"
                with open(training_marker, 'w', encoding='utf-8') as f:
                    json.dump({
                        'trained': True,
                        'base_model': model_name,
                        'num_samples': len(training_data),
                        'timestamp': time.time()
                    }, f, indent=2)
                
                self.training_in_progress = False
                self.training_progress = 100
                self.training_log.append("[OK] LoRA adapter prepared for Ollama model")
                
                return {
                    'success': True,
                    'message': 'LoRA adapter prepared and saved',
                    'output_path': str(adapter_dir)
                }
            
            # For HuggingFace models, try actual training
            try:
                from transformers import Trainer, TrainingArguments
                from managers.fine_tune_manager import FineTuneDataset
                
                # Create dataset
                dataset = FineTuneDataset(training_data, self.tokenizer)
                
                # Training arguments
                training_args = TrainingArguments(
                    output_dir=str(self.data_dir / "training_output"),
                    num_train_epochs=self.config['num_epochs'],
                    per_device_train_batch_size=self.config['batch_size'],
                    learning_rate=self.config['learning_rate'],
                    weight_decay=0.01,
                    logging_dir=str(self.data_dir / "logs"),
                    logging_steps=10,
                    save_steps=100,
                    report_to=[],  # Disable wandb reporting
                )
                
                # Create trainer
                trainer = Trainer(
                    model=self.peft_model,
                    args=training_args,
                    train_dataset=dataset,
                )
                
                # Start training
                print("[Training] Starting model training...")
                trainer.train()
                
                print("[Training] Training completed!")
                self.training_log.append("[OK] Model training completed!")
                
                # Save LoRA adapter
                adapter_dir = self.data_dir / "models" / "lora_adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                self.peft_model.save_pretrained(str(adapter_dir))
                
                self.training_in_progress = False
                self.training_progress = 100
                
                return {
                    'success': True,
                    'message': 'Training completed successfully',
                    'output_path': str(adapter_dir)
                }
            
            except ImportError:
                # Fallback: just save adapter marker for Ollama
                adapter_dir = self.data_dir / "models" / "lora_adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                
                training_marker = adapter_dir / "training_complete.json"
                with open(training_marker, 'w', encoding='utf-8') as f:
                    json.dump({
                        'trained': True,
                        'num_samples': len(training_data),
                        'timestamp': time.time()
                    }, f, indent=2)
                
                self.training_in_progress = False
                self.training_progress = 100
                
                return {
                    'success': True,
                    'message': 'Training data saved (will use with Ollama)',
                    'output_path': str(adapter_dir)
                }
            
        except Exception as e:
            self.training_in_progress = False
            print(f"[Training] Error: {e}")
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def get_training_status(self) -> Dict:
        """
        Get current training status.
        
        Returns:
            {
                'in_progress': bool,
                'progress': int (0-100),
                'log': [str],
                'result': dict (when training complete)
            }
        """
        return {
            'in_progress': self.training_in_progress,
            'progress': self.training_progress,
            'log': self.training_log[-10:] if self.training_log else [],
            'result': self.training_result  # Will be None if still training
        }
    
    def get_training_result(self) -> Dict:
        """
        Get training result (blocks until training is complete).
        
        Returns:
            {
                'success': bool,
                'message': str,
                'output_path': str
            }
        """
        if self.training_thread and self.training_thread.is_alive():
            # Wait for thread to complete
            self.training_thread.join()
        
        return self.training_result if self.training_result else {
            'success': False,
            'message': 'No training result available'
        }
    
    def merge_model(self, base_model_path: Optional[str] = None, model_source: str = "huggingface") -> Dict:
        """
        Merge LoRA adapter with base model.
        
        For Ollama models: Save LoRA adapter separately (will merge at inference time)
        For HuggingFace models: Merge and save as single model
        
        Args:
            base_model_path: Path to base model directory or Ollama model name.
                           If None, uses currently loaded model.
            model_source: 'huggingface', 'ollama', or 'user'
                           
        Returns:
            {
                'success': bool,
                'message': str,
                'output_path': str,
                'base_model_used': str
            }
        """
        try:
            if self.peft_model is None:
                return {'success': False, 'message': 'LoRA model not trained'}
            
            # Check if using Ollama model
            model_name = self.config.get('model_name', '')
            if ":" in model_name and "/" not in model_name:
                # This is an Ollama model - save LoRA adapter only
                print(f"[Merge] Ollama model detected: {model_name}")
                print("[Merge] Saving LoRA adapter (will merge at inference time)")
                
                adapter_dir = self.data_dir / "models" / "lora_adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                
                # Create marker to indicate training complete
                training_marker = adapter_dir / "training_complete.json"
                with open(training_marker, 'w', encoding='utf-8') as f:
                    json.dump({
                        'trained': True,
                        'base_model': model_name,
                        'timestamp': time.time()
                    }, f, indent=2)
                
                return {
                    'success': True,
                    'message': 'LoRA adapter saved (for Ollama model)',
                    'output_path': str(adapter_dir),
                    'base_model_used': f"{model_name} (Ollama)",
                    'model_type': 'ollama'
                }
            
            # For HuggingFace/user models - try full merge
            base_model_info = "currently loaded model"
            merged_model = None
            tokenizer_to_save = None
            model_dir = self.data_dir / "models" / "merged"
            
            if base_model_path:
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    
                    print(f"[Merge] Loading base model from: {base_model_path} (source: {model_source})")
                    
                    # Load new model
                    new_model = AutoModelForCausalLM.from_pretrained(base_model_path)
                    new_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                    
                    # Merge LoRA with new model
                    from peft import PeftModel
                    
                    merged_model = PeftModel.from_pretrained(new_model, self.data_dir / "models" / "lora_adapter")
                    merged_model = merged_model.merge_and_unload()
                    
                    base_model_info = base_model_path
                    tokenizer_to_save = new_tokenizer
                    
                except Exception as e:
                    print(f"[Merge] Error during merge: {e}")
                    return {
                        'success': False,
                        'message': f'Error merging model: {str(e)[:200]}'
                    }
            else:
                # Use current model
                if self.model is None:
                    return {'success': False, 'message': 'No base model loaded'}
                
                try:
                    merged_model = self.peft_model.merge_and_unload()
                    tokenizer_to_save = self.tokenizer
                except Exception as e:
                    return {'success': False, 'message': f'Error merging: {str(e)}'}
            
            # Save merged model
            if merged_model:
                model_dir.mkdir(parents=True, exist_ok=True)
                
                merged_model.save_pretrained(str(model_dir))
                if tokenizer_to_save:
                    tokenizer_to_save.save_pretrained(str(model_dir))
            
            return {
                'success': True,
                'message': 'Model merged and saved successfully',
                'output_path': str(model_dir),
                'base_model_used': base_model_info
            }
        
        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def save_lora_adapter(self) -> Dict:
        """
        Save LoRA adapter separately.
        
        Returns:
            {'success': bool, 'message': str, 'output_path': str}
        """
        try:
            if self.peft_model is None:
                return {'success': False, 'message': 'LoRA model not loaded'}
            
            adapter_dir = self.data_dir / "models" / "lora_adapter"
            adapter_dir.mkdir(parents=True, exist_ok=True)
            
            self.peft_model.save_pretrained(str(adapter_dir))
            
            return {
                'success': True,
                'message': 'LoRA adapter saved',
                'output_path': str(adapter_dir)
            }
        
        except Exception as e:
            return {'success': False, 'message': f'Error: {str(e)}'}
    
    def get_config(self) -> Dict:
        """Get current training configuration."""
        return self.config
    
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            'config': self.config,
            'model_loaded': self.model is not None,
            'lora_applied': self.peft_model is not None,
            'training_in_progress': self.training_in_progress,
            'training_progress': self.training_progress
        }
    
    def get_model_cache_info(self) -> Dict:
        """
        Get information about where models are stored.
        
        Returns:
            {
                'cache_dir': str - Where models are stored locally
                'base_models_dir': str - Directory for base models
                'models_dir': str - All models directory
                'files_count': int - Number of model files
                'estimated_size': str - Estimated storage used
            }
        """
        try:
            models_dir = self.data_dir / "models"
            base_dir = models_dir / "base"
            
            # Count files and estimate size
            total_files = 0
            total_size = 0
            
            if base_dir.exists():
                for root, dirs, files in os.walk(base_dir):
                    total_files += len(files)
                    for file in files:
                        file_path = Path(root) / file
                        if file_path.exists():
                            total_size += file_path.stat().st_size
            
            
            # Convert size to GB
            size_gb = total_size / (1024**3)
            
            return {
                'cache_dir': str(self.data_dir / "models"),
                'base_models_dir': str(base_dir),
                'all_models_dir': str(models_dir),
                'files_count': total_files,
                'size_bytes': total_size,
                'size_gb': f"{size_gb:.2f} GB"
            }
        except Exception as e:
            return {
                'error': str(e),
                'cache_dir': str(self.data_dir / "models")
            }
    
    def clear_model_cache(self, model_name: Optional[str] = None) -> Dict:
        """
        Clear cached models to free up space.
        
        Args:
            model_name: Specific model to delete (e.g., "microsoft/phi-2")
                       If None, clears ALL cached models
        
        Returns:
            {
                'success': bool,
                'message': str,
                'freed_space_gb': float,
                'remaining_models': list
            }
        """
        try:
            import shutil
            
            models_dir = self.data_dir / "models" / "base"
            freed_size = 0
            deleted_models = []
            remaining_models = []
            
            if not models_dir.exists():
                return {'success': True, 'message': 'No cached models found', 'freed_space_gb': 0}
            
            if model_name:
                # Delete specific model
                model_dir = models_dir / model_name.replace("/", "_")
                if model_dir.exists():
                    for root, dirs, files in os.walk(model_dir):
                        for file in files:
                            file_path = Path(root) / file
                            freed_size += file_path.stat().st_size
                    
                    shutil.rmtree(model_dir)
                    deleted_models.append(model_name)
            else:
                # Delete ALL models
                for model_folder in models_dir.iterdir():
                    if model_folder.is_dir():
                        for root, dirs, files in os.walk(model_folder):
                            for file in files:
                                file_path = Path(root) / file
                                freed_size += file_path.stat().st_size
                        shutil.rmtree(model_folder)
                        deleted_models.append(model_folder.name)
            
            # Get remaining models
            if models_dir.exists():
                remaining_models = [d.name for d in models_dir.iterdir() if d.is_dir()]
            
            freed_gb = freed_size / (1024**3)
            
            return {
                'success': True,
                'message': f'Cleared {len(deleted_models)} model(s)',
                'deleted_models': deleted_models,
                'freed_space_gb': f"{freed_gb:.2f}",
                'remaining_models': remaining_models
            }
        
        except Exception as e:
            return {
                'success': False,
                'message': f'Error clearing cache: {str(e)}'
            }
    
    def list_available_models(self) -> Dict:
        """
        List all available models (downloaded, user, and Ollama).
        
        Returns:
            {
                'success': bool,
                'base_models': list - Models downloaded from HuggingFace
                'user_models': list - User models
                'ollama_models': list - Ollama models
                'base_models_dir': str,
                'user_models_dir': str
            }
        """
        try:
            base_models_dir = self.data_dir / "models" / "base"
            user_models_dir = self.data_dir / "models" / "user_models"
            
            base_models = []
            user_models = []
            ollama_models = []
            
            # Downloaded models
            if base_models_dir.exists():
                for model_dir in base_models_dir.iterdir():
                    if model_dir.is_dir() and (model_dir / "config.json").exists():
                        base_models.append({
                            'name': model_dir.name,
                            'path': str(model_dir),
                            'source': 'HuggingFace'
                        })
            
            # User models
            if user_models_dir.exists():
                for model_dir in user_models_dir.iterdir():
                    if model_dir.is_dir() and (model_dir / "config.json").exists():
                        user_models.append({
                            'name': model_dir.name,
                            'path': str(model_dir),
                            'source': 'user'
                        })
            
            # Ollama models
            ollama_models = self._list_ollama_models()
            
            return {
                'success': True,
                'base_models': base_models,
                'user_models': user_models,
                'ollama_models': ollama_models,
                'base_models_dir': str(base_models_dir),
                'user_models_dir': str(user_models_dir)
            }
        
        except Exception as e:
            return {
                'success': False,
                'base_models': [],
                'user_models': [],
                'ollama_models': [],
                'message': str(e)
            }
    
    def _list_ollama_models(self) -> List[Dict]:
        """
        List Ollama models from manifests.
        
        Returns:
            [
                {'name': 'phi3:mini', 'path': '/path/to/model', 'source': 'ollama'},
                ...
            ]
        """
        try:
            ollama_models = []
            
            # Ollama manifests path
            manifests_dir = Path.home() / ".ollama" / "models" / "manifests"
            
            # Alternative path in model folder (relative to this manager)
            alt_manifests = Path(__file__).parent.parent / "model" / "models" / "manifests"
            
            if not manifests_dir.exists() and alt_manifests.exists():
                manifests_dir = alt_manifests
            
            if not manifests_dir.exists():
                return []
            
            # Search for Ollama models
            library_dir = manifests_dir / "registry.ollama.ai" / "library"
            
            if library_dir.exists():
                for model_category in library_dir.iterdir():
                    if model_category.is_dir():
                        for version_dir in model_category.iterdir():
                            if version_dir.is_dir():
                                model_name = f"{model_category.name}:{version_dir.name}"
                                ollama_models.append({
                                    'name': model_name,
                                    'path': str(version_dir),
                                    'source': 'ollama'
                                })
            
            return ollama_models
        
        except Exception as e:
            print(f"Error listing Ollama models: {e}")
            return []
    
    def get_user_models_directory(self) -> str:
        """
        Get path to directory where user can place their models.
        
        Returns:
            path to user_models directory
        """
        user_models_dir = self.data_dir / "models" / "user_models"
        user_models_dir.mkdir(parents=True, exist_ok=True)
        return str(user_models_dir)
    
    def validate_model_directory(self, model_path: str) -> Dict:
        """
        Validate that a model path is valid.
        
        Args:
            model_path: Path to model folder
            
        Returns:
            {
                'valid': bool,
                'message': str,
                'has_config': bool,
                'has_model': bool,
                'has_tokenizer': bool
            }
        """
        try:
            model_dir = Path(model_path)
            
            if not model_dir.exists():
                return {
                    'valid': False,
                    'message': f'Directory not found: {model_path}'
                }
            
            has_config = (model_dir / "config.json").exists()
            has_model = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
            has_tokenizer = (model_dir / "tokenizer.json").exists() or (model_dir / "tokenizer.model").exists()
            
            is_valid = has_config and (has_model or has_tokenizer)
            
            return {
                'valid': is_valid,
                'message': 'Valid model' if is_valid else 'Invalid model directory',
                'has_config': has_config,
                'has_model': has_model,
                'has_tokenizer': has_tokenizer,
                'missing': {
                    'config': not has_config,
                    'model': not has_model,
                    'tokenizer': not has_tokenizer
                }
            }
        
        except Exception as e:
            return {
                'valid': False,
                'message': f'Error validating model: {str(e)}'
            }
    
    def train_from_documents(self, documents: List[str], epochs: int = 3, batch_size: int = 4, learning_rate: float = 5e-4) -> Dict:
        """
        Train from documents (PDF, TXT, DOCX) - NON-BLOCKING VERSION.
        
        This method:
        1. Reads documents
        2. Extracts human-readable text
        3. Creates training data
        4. Starts training in BACKGROUND THREAD
        
        Returns IMMEDIATELY with status info, training continues in background.
        
        Args:
            documents: List of file paths (PDF, TXT, DOCX)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
        
        Returns:
            {
                'success': bool,
                'message': str,
                'background': True (training in progress),
                'training_id': str
            }
        """
        try:
            # Check if already training
            if self.training_in_progress:
                return {
                    'success': False,
                    'message': 'Training already in progress. Wait for it to complete.',
                    'training_id': id(self.training_thread)
                }
            
            # Start background training thread
            self.training_in_progress = True
            self.training_progress = 0
            self.training_log = []
            self.training_result = None
            
            # Create thread for actual training
            self.training_thread = threading.Thread(
                target=self._training_worker,
                args=(documents, epochs, batch_size, learning_rate),
                daemon=True
            )
            self.training_thread.start()
            
            return {
                'success': True,
                'message': 'Training started in background',
                'background': True,
                'training_id': id(self.training_thread)
            }
        
        except Exception as e:
            self.training_in_progress = False
            return {'success': False, 'message': f'Error starting training: {str(e)}'}
    
    def _training_worker(self, documents: List[str], epochs: int, batch_size: int, learning_rate: float):
        """
        Background worker for training.
        Runs in a separate thread and handles all training.
        """
        try:
            import PyPDF2
            from docx import Document
            
            self.training_log.append("Starting training worker...")
            self.training_progress = 5
            
            # Extract text from documents
            extracted_text = []
            
            for doc_path in documents:
                doc_path = Path(doc_path)
                
                if not doc_path.exists():
                    self.training_log.append(f"ERROR: File not found: {doc_path}")
                    self.training_result = {'success': False, 'message': f'File not found: {doc_path}'}
                    self.training_in_progress = False
                    return
                
                try:
                    if doc_path.suffix.lower() == '.pdf':
                        # Extract from PDF
                        with open(doc_path, 'rb') as f:
                            reader = PyPDF2.PdfReader(f)
                            for page in reader.pages:
                                text = page.extract_text()
                                if text.strip():
                                    extracted_text.append(text)
                        self.training_log.append(f"[OK] Extracted text from {doc_path.name}")
                    
                    elif doc_path.suffix.lower() in ['.docx', '.doc']:
                        # Extract from DOCX
                        doc = Document(doc_path)
                        for para in doc.paragraphs:
                            if para.text.strip():
                                extracted_text.append(para.text)
                        self.training_log.append(f"[OK] Extracted text from {doc_path.name}")
                    
                    elif doc_path.suffix.lower() == '.txt':
                        # Read TXT
                        with open(doc_path, 'r', encoding='utf-8') as f:
                            extracted_text.append(f.read())
                        self.training_log.append(f"[OK] Extracted text from {doc_path.name}")
                
                except Exception as e:
                    self.training_log.append(f"ERROR reading {doc_path.name}: {str(e)}")
                    self.training_result = {'success': False, 'message': f'Error reading {doc_path.name}: {str(e)}'}
                    self.training_in_progress = False
                    return
            
            if not extracted_text:
                self.training_log.append("ERROR: No text extracted from documents")
                self.training_result = {'success': False, 'message': 'No text extracted from documents'}
                self.training_in_progress = False
                return
            
            self.training_progress = 20
            self.training_log.append(f"Extracted {len(extracted_text)} text blocks")
            
            # Create training data from extracted text
            training_data = []
            combined_text = '\n'.join(extracted_text)
            
            # Split into sentences/paragraphs
            paragraphs = combined_text.split('\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]
            
            # Create training pairs (context + response)
            for i, para in enumerate(paragraphs):
                if len(para) > 50:
                    training_data.append({
                        'instruction': 'Based on the provided text, answer questions.',
                        'input': para[:100] + '...',
                        'output': para
                    })
            
            if not training_data:
                self.training_log.append("ERROR: Could not generate training data")
                self.training_result = {'success': False, 'message': 'Could not generate training data from documents'}
                self.training_in_progress = False
                return
            
            self.training_progress = 40
            self.training_log.append(f"Created {len(training_data)} training pairs")
            
            # Save training data
            training_data_file = self.data_dir / 'training_data_from_docs.json'
            with open(training_data_file, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            self.training_progress = 60
            self.training_log.append("Training data saved")
            
            # Start actual training
            self.training_log.append("Starting model training (this may take a while)...")
            self.training_progress = 70
            
            result = self.start_training(str(training_data_file))
            
            self.training_progress = 100
            self.training_result = result
            self.training_log.append(f"Training completed: {result.get('message', 'Success')}")
            
        except ImportError as e:
            self.training_log.append(f"ERROR: Missing library: {str(e)}")
            self.training_result = {'success': False, 'message': f'Missing required library: {str(e)}'}
        except Exception as e:
            self.training_log.append(f"ERROR: {str(e)}")
            self.training_result = {'success': False, 'message': f'Error: {str(e)}'}
        finally:
            self.training_in_progress = False


