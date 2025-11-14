"""Ollama service manager."""

import subprocess
import os
import requests
import requests.adapters
import json
import time
import traceback


class OllamaManager:
    """Manages Ollama AI service."""
    
    def __init__(self, host: str = "http://localhost:11434", ollama_path: str = None):
        self.host = host
        self.process = None
        self._debug = True  # Enable debug logging
        self.ollama_path = ollama_path
    
    def _log(self, msg: str):
        """Debug logging."""
        if self._debug:
            print(f"[Ollama] {msg}")
    
    def is_running(self) -> bool:
        """Check if Ollama is running."""
        try:
            # No timeout - wait for Ollama response
            response = requests.get(f"{self.host}/api/tags", timeout=None)
            return response.status_code == 200
        except requests.Timeout:
            self._log("Timeout while checking if running - giving more time...")
            return False
        except Exception as e:
            self._log(f"Error checking if running: {e}")
            return False
    
    def start(self) -> bool:
        """Start Ollama service in background."""
        if self.is_running():
            self._log("Already running")
            return True
        
        try:
            self._log("Attempting to start service...")
            
            # Check if ollama executable exists
            from shutil import which
            # If an explicit path was provided on the manager, try that first
            ollama_path = None
            try:
                if hasattr(self, 'ollama_path') and self.ollama_path:
                    if os.path.exists(self.ollama_path):
                        ollama_path = self.ollama_path
            except Exception:
                pass

            if not ollama_path:
                ollama_path = which('ollama')

            # If not in PATH, check common install locations on Windows and Program Files
            if not ollama_path and os.name == 'nt':
                candidates = []
                local_app = os.environ.get('LOCALAPPDATA') or ''
                program_files = os.environ.get('PROGRAMFILES') or ''
                program_files_x86 = os.environ.get('PROGRAMFILES(X86)') or ''
                # Common Ollama install locations
                candidates.append(os.path.join(local_app, 'Programs', 'Ollama', 'ollama.EXE'))
                candidates.append(os.path.join(program_files, 'Ollama', 'ollama.EXE'))
                candidates.append(os.path.join(program_files_x86, 'Ollama', 'ollama.EXE'))
                # Also try user's AppData Local directly
                candidates.append(os.path.join(os.path.expanduser('~'), 'AppData', 'Local', 'Programs', 'Ollama', 'ollama.EXE'))
                for p in candidates:
                    try:
                        if p and os.path.exists(p):
                            ollama_path = p
                            break
                    except Exception:
                        continue

            if not ollama_path:
                self._log("Error: ollama executable not found (not in PATH or common locations)")
                return False
            
            self._log(f"Found ollama at: {ollama_path}")
            
            if os.name == 'nt':  # Windows
                # Launch detached so the app doesn't need a manual serve window open
                CREATE_NO_WINDOW = 0x08000000
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                creationflags = CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

                try:
                    self.process = subprocess.Popen(
                        [ollama_path, 'serve'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=creationflags
                    )
                except TypeError:
                    # Fallback for platforms that may not accept creationflags
                    self.process = subprocess.Popen(
                        [ollama_path, 'serve'],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
            else:  # Linux/Mac
                # Use nohup-like behavior
                self.process = subprocess.Popen(
                    [ollama_path, 'serve'],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            
            # Wait up to 30 seconds for the server to become available
            self._log("Waiting for service to start...")
            for i in range(30):
                if self.is_running():
                    self._log(f"Service started successfully after {i+1} seconds")
                    return True
                time.sleep(1)
            
            # If we get here, service didn't start
            self._log("Service failed to start after 15 seconds")
            if self.process:
                out, err = self.process.communicate()
                self._log(f"stdout: {out.decode() if out else 'None'}")
                self._log(f"stderr: {err.decode() if err else 'None'}")
            return False
        
        except Exception as e:
            self._log(f"Error starting service: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def stop(self):
        """Stop Ollama service."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except:
                try:
                    self.process.kill()
                except:
                    pass
    
    def get_models(self) -> list:
        """Get list of available models."""
        if not self.is_running():
            return []
        
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=None)
            if response.status_code == 200:
                data = response.json()
                models = [model['name'] for model in data.get('models', [])]
                self._log(f"Found models: {models}")
                return models
        except Exception as e:
            self._log(f"Error getting models: {e}")
        
        return []
    
    def chat(self, model: str, message: str, history: list = None, 
             temperature: float = 0.7, context_length: int = 4096) -> str:
        """Send chat message to Ollama."""
        if not self.is_running():
            raise Exception("Ollama is not running")
        
        self._log(f"Chat request - Model: {model}")
        self._log(f"Message: {message[:100]}...")
        
        # Build messages list with deduplication
        messages = []
        seen_messages = set()
        
        if history:
            for h in history:
                msg_content = h['content'].strip()
                if msg_content not in seen_messages:
                    messages.append({"role": h['role'], "content": msg_content})
                    seen_messages.add(msg_content)
                    self._log(f"History message - Role: {h['role']}, Content: {msg_content[:50]}...")
        
        # Add current message if not duplicate
        msg_content = message.strip()
        if msg_content not in seen_messages:
            messages.append({"role": "user", "content": msg_content})
        self._log(f"Total unique messages in context: {len(messages)}")
        
        # Build request options
        options = {
            "temperature": temperature,
            "num_thread": 4,
            "num_ctx": context_length,
            "top_k": 40,
            "top_p": 0.9,
            "repeat_penalty": 1.1
        }
        
        request_body = {
            "model": model,
            "messages": messages,
            "options": options
        }
        
        self._log("Full request body:")
        self._log(json.dumps(request_body, indent=2, ensure_ascii=False))
        
        session = None
        response = None
        
        try:
            # Check model availability
            self._log("Checking model availability...")
            model_list = self.get_models()
            if model not in model_list:
                self._log(f"Warning: Model {model} not found in available models: {model_list}")
            
            # Use requests directly without streaming for initial response
            self._log("Making API call...")
            response = requests.post(
                f"{self.host}/api/chat",
                json=request_body,
                timeout=None  # No timeout - wait for full response
            )
            
            self._log(f"Got response with status: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = f"Error: HTTP {response.status_code} - {response.text}"
                self._log(error_msg)
                return error_msg
            
            # Process response with robust JSON/NDJSON handling
            text = response.text
            self._log(f"Raw response length: {len(text)}")
            # Try simple JSON first
            try:
                data = json.loads(text)
                if isinstance(data, dict) and 'message' in data and 'content' in data['message']:
                    content = data['message']['content']
                    self._log(f"Received response: {content[:200]}...")
                    return content
                # If it's a list of messages, try to extract
                if isinstance(data, list):
                    parts = []
                    for item in data:
                        if isinstance(item, dict) and 'message' in item and 'content' in item['message']:
                            parts.append(item['message']['content'])
                    if parts:
                        joined = ''.join(parts)
                        self._log(f"Joined response from list: {len(joined)} chars")
                        return joined
                # fallback
                self._log(f"JSON parsed but unexpected structure: {type(data)}")
            except json.JSONDecodeError as e:
                self._log(f"Primary JSON parse failed: {e}")

            # Try newline-delimited JSON (NDJSON)
            parts = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and 'message' in item and 'content' in item['message']:
                        parts.append(item['message']['content'])
                except Exception:
                    # ignore lines that aren't JSON
                    continue

            if parts:
                joined = ''.join(parts)
                self._log(f"Joined NDJSON response: {len(joined)} chars")
                return joined

            # As a last resort, try incremental JSON decoding to handle concatenated JSON objects
            try:
                decoder = json.JSONDecoder()
                s = text.strip()
                idx = 0
                decoded_parts = []
                while idx < len(s):
                    s = s.lstrip()
                    try:
                        obj, offset = decoder.raw_decode(s)
                    except Exception:
                        break
                    decoded_parts.append(obj)
                    s = s[offset:]

                parts = []
                for obj in decoded_parts:
                    if isinstance(obj, dict) and 'message' in obj and 'content' in obj['message']:
                        parts.append(obj['message']['content'])

                if parts:
                    joined = ''.join(parts)
                    self._log(f"Joined concatenated JSON response: {len(joined)} chars")
                    return joined
            except Exception as ex:
                self._log(f"Incremental decode failed: {ex}")

            # If nothing worked, return the raw text as a fallback (avoid JSON error)
            if text.strip():
                self._log("Returning raw response text as fallback")
                return text
            return "Error: Failed to parse response from model"
            
        except requests.Timeout as e:
            error_msg = (
                "⚠️ Request timed out. Try:\n"
                "1. Using a smaller model\n"
                "2. Reducing context length\n"
                "3. Asking shorter questions"
            )
            self._log(f"Timeout error: {e}")
            return error_msg
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self._log(f"Unexpected error: {e}")
            traceback.print_exc()
            return error_msg
            
        finally:
            if response:
                try:
                    response.close()
                except:
                    pass
            if session:
                try:
                    session.close()
                except:
                    pass
