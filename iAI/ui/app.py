"""Main application UI logic."""

import flet as ft
import threading
from managers import OllamaManager, ConfigManager, AssetManager, KnowledgeManager, FineTuneManager, QueryManager, LoRAFineTuner
from managers.book_trainer import BookTrainer
from managers.book_query_optimizer import BookQueryOptimizer
from .theme import ThemeManager
from .components import UIComponents
from .train_personal_ai import TrainPersonalAI
from .use_trained_model import UseTrainedModel


class LocalAIChatApp:
    """Main Flet application."""
    
    def __init__(self, page: ft.Page):
        self.page = page
        
        # Initialize managers
        self.config = ConfigManager()
        self.ollama = OllamaManager(ollama_path=self.config.get('ollama_path', None))
        self.assets = AssetManager()
        self.knowledge = KnowledgeManager()
        self.fine_tune = FineTuneManager()
        self.query = QueryManager()
        self.lora_trainer = LoRAFineTuner()
        self.book_trainer = BookTrainer(
            knowledge_manager=self.knowledge,
            lora_trainer=self.lora_trainer
        )
        self.query_optimizer = BookQueryOptimizer(
            knowledge_manager=self.knowledge,
            book_trainer=self.book_trainer
        )

        # State
        self.history = []
        self.chat_histories = [{"id": 1, "history": []}]
        self.active_chat_id = 1
        self.is_chat_view = False
        self.current_model = self.config.get('model')
        self.temperature = self.config.get('temperature', 0.7)
        self.context_length = self.config.get('context_length', 4096)
        self.theme = self.config.get('theme', 'dark')
        self._model_loaded = False
        
        # Ensure auto_start_ollama is enabled by default
        if not self.config.get('auto_start_ollama'):
            self.config.set('auto_start_ollama', True)
            self.config.save()
        
        # UI Controls (initialized later)
        self.chat_view = None
        self.input_field = None
        self.model_dropdown = None
        self.status_text = None
        self.bg_image_container = None
        self.model_icon_container = None
        self.toolbar_items = None
        self.chat_items = None
        
        # Configure page and window
        self.page.title = "iAI"
        self.page.padding = 0
        self.page.window_width = 1000
        self.page.window_height = 700
        self.page.window_resizable = True
        
        # Set window icon - use relative path from assets_dir
        # Flet expects path relative to assets_dir specified in ft.app()
        self.page.window_icon_path = "icons/logo.png"
        print(f"[App] Window icon set to: icons/logo.png")
        
        # Build UI
        self.build_ui()
        
        # Initialize Ollama in background
        threading.Thread(target=self.initialize_ollama, daemon=True).start()
        
        # Handle window close and resize
        self.page.on_close = self.on_close
        self.page.on_resize = self.on_resize
    
    # ========================================================================
    # UI BUILDING
    # ========================================================================
    
    def build_ui(self):
        """Build the complete UI - FAST first pass without heavy assets."""
        # Create UI controls first (without waiting for assets)
        self.chat_view = ft.ListView(
            expand=True,
            spacing=10,
            padding=20,
            auto_scroll=True
        )
        
        self.input_field = ft.TextField(
            hint_text="Type ...",
            multiline=False,
            expand=True,
            on_submit=self.send_message,
            border_radius=25,
            filled=True,
            content_padding=ft.padding.symmetric(horizontal=20, vertical=15)
        )
        
        self.model_dropdown = ft.Dropdown(
            hint_text="Select Model",
            options=[ft.dropdown.Option("Loading...")],
            on_change=self.on_model_change,
            width=250
        )
        
        self.status_text = ft.Text(
            "Initializing...",
            size=12,
            color=ft.Colors.GREY_500
        )
        
        # [OK] DO NOT load background here - show UI first!
        # bg_image will be set later in background thread
        bg_image = None
        
        # Build layout WITHOUT background initially
        main_content = ft.Row(
            controls=[
                UIComponents.create_sidebar(self, self.theme),
                ft.VerticalDivider(width=1),
                ft.Column(
                    controls=[
                        UIComponents.create_top_bar(self, self.theme),
                        UIComponents.create_chat_area(self),
                        UIComponents.create_input_area(self, self.theme),
                    ],
                    expand=True,
                    spacing=0
                )
            ],
            expand=True,
            spacing=0
        )
        
        # Simple layout initially (NO background)
        self.main_layout = main_content
        self.page.add(self.main_layout)
        
        # Add initial message
        self.add_message("assistant", "Hello! I'm iAI, your local AI assistant.")
        
        # [OK] Load theme & background in BACKGROUND THREAD
        threading.Thread(target=self._load_theme_async, daemon=True).start()
    
    def _load_theme_async(self):
        """Load theme and background asynchronously."""
        try:
            print("[App] Loading theme and background...")
            # Apply theme (this loads background)
            bg_image = ThemeManager.apply_theme(self.page, self.theme, self.assets)
            
            if bg_image:
                print("[App] Background loaded, updating UI...")
                # Create new layout with background
                main_content = self.main_layout
                
                new_layout = ft.Stack(
                    controls=[
                        bg_image,
                        ft.Container(
                            content=main_content,
                            expand=True,
                            bgcolor=ft.Colors.TRANSPARENT
                        )
                    ],
                    expand=True
                )
                
                # Update UI on main thread
                self.page.clean()
                self.page.add(new_layout)
                self.page.update()
                print("[App] [OK] Theme and background applied!")
        except Exception as e:
            print(f"[App] Error loading theme: {e}")
    
    # ========================================================================
    # THEME MANAGEMENT
    # ========================================================================
    
    def toggle_theme(self, e=None):
        """Toggle between light and dark theme."""
        self.theme = 'light' if self.theme == 'dark' else 'dark'
        self.config.set('theme', self.theme)
        self.config.save()
        
        # Rebuild UI with new theme
        prev_status = self.status_text.value if self.status_text else None
        prev_model = self.current_model

        self.page.controls.clear()
        self.build_ui()

        # Restore chat history
        temp_history = self.history.copy()
        self.history = []
        for msg in temp_history:
            self.add_message(msg['role'], msg['content'])

        # Refresh available models
        try:
            self.refresh_models()
        except Exception:
            pass

        # Check if models are available
        try:
            option_values = [opt.value for opt in self.model_dropdown.options]
            has_models = any(v and v != 'No models' and v != 'Loading...' for v in option_values)
        except Exception:
            has_models = False

        if has_models:
            try:
                if prev_model and prev_model in [opt.value for opt in self.model_dropdown.options]:
                    self.model_dropdown.value = prev_model
                    self.current_model = prev_model
                    self.model_dropdown.update()
            except Exception:
                pass

            if prev_status:
                self.update_status(prev_status)
            else:
                self.update_status("Ready")
        else:
            self.update_status("Not running")

        self.page.update()
    
    # ========================================================================
    # OLLAMA OPERATIONS
    # ========================================================================
    
    def initialize_ollama(self):
        """Initialize Ollama service."""
        try:
            self.update_status("Checking Service...")
            print("[App] Checking Ollama status...")
            
            if self.ollama.is_running():
                print("[App] Ollama is already running")
                self.update_status("Ready")
                self.refresh_models()
                return
            
            if self.config.get('auto_start_ollama', True):
                print("[App] Attempting to start Service...")
                self.add_message("system", "Starting Service...")
                self.update_status("Starting...")
                
                if self.ollama.start():
                    print("[App] Service started successfully")
                    self.add_message("system", "Service started!")
                    self.update_status("Ready")
                    
                    import time
                    time.sleep(2)
                    self.refresh_models()
                else:
                    error_msg = "Could not start Service. Please make sure it's installed and try running 'ollama serve' manually."
                    print(f"[App] {error_msg}")
                    self.add_message("error", f" {error_msg}")
                    self.update_status("Not running")
            else:
                error_msg = "Service is not running and auto-start is disabled."
                print(f"[App] {error_msg}")
                self.add_message("error", f"{error_msg}")
                self.update_status("Not running")
        
        except Exception as e:
            error_msg = f"Error initializing Service: {str(e)}"
            print(f"[App] {error_msg}")
            import traceback
            traceback.print_exc()
            self.add_message("error", f"{error_msg}")
            self.update_status("Error")
    
    def refresh_models(self):
        """Refresh available models."""
        models = self.ollama.get_models()
        
        if models:
            self.model_dropdown.options = [ft.dropdown.Option(m) for m in models]
            
            if self.current_model not in models:
                self.current_model = models[0]
                self.model_dropdown.value = models[0]
            else:
                self.model_dropdown.value = self.current_model
            
            self.model_dropdown.update()
            
            if self.model_icon_container:
                suffix = ".light" if self.theme == "light" else ""
                model_icon = self.assets.get_icon(f"model_two{suffix}", 32)
                if model_icon:
                    self.model_icon_container.content = ft.Image(
                        src_base64=model_icon,
                        width=32,
                        height=32
                    )
                    self.model_icon_container.update()
            
            self.show_snackbar(f"{len(models)} model(s) found")
        else:
            self.model_dropdown.options = [ft.dropdown.Option("No models")]
            self.model_dropdown.update()
            self.show_snackbar("No models found")
    
    # ========================================================================
    # USER ACTIONS
    # ========================================================================
    
    def send_message(self, e):
        """Send user message."""
        message = self.input_field.value.strip()
        
        if not message:
            return
        
        if not self.ollama.is_running():
            self.show_snackbar("Service not running!")
            return
        
        if not self.current_model or self.current_model == 'Loading...':
            self.show_snackbar("Please select a model!")
            return
        
        # Clear input
        self.input_field.value = ""
        self.page.update()
        
        # Display user message
        self.add_message("user", message)
        
        # Query in background
        self.update_status("Thinking...")
        threading.Thread(
            target=self._query_ollama,
            args=(self.current_model, message),
            daemon=True
        ).start()
    
    def _query_ollama(self, model: str, message: str):
        """Query Ollama (background thread) with RAG and book training support."""
        try:
            print(f"[Debug] Using settings - Temperature: {self.temperature}, Context Length: {self.context_length}")
            
            # Add user message to history
            self.history.append({'role': 'user', 'content': message})
            
            # Check if book was trained and use appropriate method
            book_info = self.book_trainer.get_book_info()
            trained_method = self.book_trainer.training_method
            
            original_message = message
            
            # ========== TRAINED BOOK SUPPORT WITH OPTIMIZED INFERENCE ==========
            if book_info.get('loaded') and trained_method:
                print(f"[App] Using trained book with method: {trained_method}")
                
                if trained_method == 'rag' or trained_method == 'hybrid':
                    try:
                        # Use optimizer for fast, timeout-protected retrieval
                        optimized_message, metadata = self.query_optimizer.get_optimized_prompt(
                            original_message,
                            max_context_tokens=512
                        )
                        
                        if metadata.get('chunks_used', 0) > 0:
                            print(f"[App] Retrieved {metadata['chunks_used']} chunks in {metadata.get('retrieval_time', 0):.3f}s")
                            message = (
                                f"Reference material from the book '{book_info.get('name')}':\n\n"
                                f"{optimized_message}"
                            )
                        else:
                            print(f"[App] No relevant context found or retrieval timed out - using base model")
                            # Falls back to original message if no context
                            
                    except Exception as opt_error:
                        print(f"[App] Optimizer error: {opt_error} - using original message")
                        # Fall back to original message on any error
                        pass
            else:
                # ========== STANDARD RAG: Search knowledge base ==========
                try:
                    context_docs = self.knowledge.search(
                        message, 
                        model_name=model, 
                        top_k=3,
                        timeout=None
                    )
                    
                    if context_docs:
                        print(f"[App] Found {len(context_docs)} relevant document chunks")
                        context_text = "\n\n".join([
                            f"[Knowledge Base - Context {i+1}]:\n{doc}" 
                            for i, doc in enumerate(context_docs)
                        ])
                        
                        message = (
                            f"You have access to the following information from the user's knowledge base. "
                            f"Use this context to answer the question accurately:\n\n"
                            f"{context_text}\n\n"
                            f"User's Question: {original_message}"
                        )
                        
                        print(f"[App] Enhanced message with knowledge base context")
                    else:
                        print("[App] No relevant knowledge base context found")
                        
                except Exception as rag_error:
                    print(f"[App] RAG error: {rag_error} - using original message")
                    # Fall back to original message on timeout/error
            # ================================================
            
            try:
                response = self.ollama.chat(
                    model=model,
                    message=message,
                    history=self.history[-10:],
                    temperature=self.temperature,
                    context_length=min(4096, self.context_length)
                )
                
                # Add response to history
                self.history.append({'role': 'assistant', 'content': response})
                
                # Display response
                self.add_message("assistant", response)
                self.update_status("Ready")
                
            except Exception as api_error:
                if "Read timed out" in str(api_error):
                    print("[Debug] Retrying with reduced context...")
                    response = self.ollama.chat(
                        model=model,
                        message=message,
                        history=self.history[-5:],
                        temperature=self.temperature,
                        context_length=2048
                    )
                    
                    self.history.append({'role': 'assistant', 'content': response})
                    self.add_message("assistant", response)
                    self.update_status("Ready")
                    
                    self.show_snackbar("Response was slow, reduced context for better performance")
                else:
                    raise api_error
        
        except Exception as e:
            error_msg = str(e)
            if "Read timed out" in error_msg:
                self.add_message("error", "Response took too long. Try reducing the context length in settings.")
            else:
                self.add_message("error", f"Error: {error_msg}")
            self.update_status("Error")
    
    # ========================================================================
    # KNOWLEDGE BASE
    # ========================================================================
    
    def show_knowledge_page(self, e):
        """Show knowledge base management page."""
        from .knowledge_components import KnowledgeUI
        
        def close_page(e):
            """Close knowledge page."""
            if knowledge_popup in self.page.overlay:
                self.page.overlay.remove(knowledge_popup)
            if overlay in self.page.overlay:
                self.page.overlay.remove(overlay)
            self.page.update()
        
        def close_on_overlay_click(e):
            """Only close if clicking on the overlay itself, not on content."""
            # Only close if target is the overlay container itself
            if e.target == overlay:
                close_page(e)
        
        # Create semi-transparent overlay
        overlay = ft.Container(
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.BLACK),
            on_click=close_on_overlay_click
        )
        
        # Create knowledge page popup with scrollable content - CENTERED
        knowledge_content = ft.Card(
            content=KnowledgeUI.create_knowledge_page(self, on_close=close_page)
        )
        
        # Use simpler layout - just center the content
        knowledge_popup = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click  # Close when clicking on left spacer
                    ),
                    ft.Container(
                        content=knowledge_content,
                        width=1200,
                        height=800,
                        bgcolor=ft.Colors.with_opacity(0.98, ft.Colors.ON_SURFACE_VARIANT),
                        border_radius=10,
                        animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT)
                    ),
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click  # Close when clicking on right spacer
                    ),
                ],
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER
            ),
            expand=True
        )
        
        # Add to page overlay
        self.page.overlay.extend([overlay, knowledge_popup])
        self.page.update()
    
    def show_fine_tune_page(self, e):
        """Show fine-tuning management page."""
        from .fine_tune_components import FineTuneUI
        
        def close_page(e):
            """Close fine-tune page."""
            if fine_tune_popup in self.page.overlay:
                self.page.overlay.remove(fine_tune_popup)
            if overlay in self.page.overlay:
                self.page.overlay.remove(overlay)
            self.page.update()
        
        def close_on_overlay_click(e):
            """Only close if clicking on the overlay itself, not on content."""
            # Only close if target is the overlay container itself
            if e.target == overlay:
                close_page(e)
        
        # Create semi-transparent overlay
        overlay = ft.Container(
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.BLACK),
            on_click=close_on_overlay_click
        )
        
        # Create fine-tune page popup with scrollable content - CENTERED
        fine_tune_content = ft.Card(
            content=FineTuneUI.create_fine_tune_page(self, on_close=close_page)
        )
        
        # Use simpler layout - just center the content
        fine_tune_popup = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click  # Close when clicking on left spacer
                    ),
                    ft.Container(
                        content=fine_tune_content,
                        width=1200,
                        height=800,
                        bgcolor=ft.Colors.with_opacity(0.98, ft.Colors.ON_SURFACE_VARIANT),
                        border_radius=10,
                        animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT)
                    ),
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click  # Close when clicking on right spacer
                    ),
                ],
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER
            ),
            expand=True
        )
        
        # Add to page overlay
        self.page.overlay.extend([overlay, fine_tune_popup])
        self.page.update()
    
    def show_query_page(self, e):
        """Show document query page."""
        from .query_components import QueryUI
        
        # Create file picker at page level
        file_picker = ft.FilePicker()
        self.page.overlay.append(file_picker)
        
        def close_page(e):
            """Close query page."""
            if query_popup in self.page.overlay:
                self.page.overlay.remove(query_popup)
            if overlay in self.page.overlay:
                self.page.overlay.remove(overlay)
            if file_picker in self.page.overlay:
                self.page.overlay.remove(file_picker)
            self.page.update()
        
        def close_on_overlay_click(e):
            """Only close if clicking on the overlay itself."""
            if e.target == overlay:
                close_page(e)
        
        # Create semi-transparent overlay
        overlay = ft.Container(
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.BLACK),
            on_click=close_on_overlay_click
        )
        
        # Create query page popup with scrollable content
        query_content = ft.Card(
            content=QueryUI.create_query_page(self, on_close=close_page, file_picker=file_picker)
        )
        
        # Center the content
        query_popup = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click
                    ),
                    ft.Container(
                        content=query_content,
                        width=1200,
                        height=800,
                        bgcolor=ft.Colors.with_opacity(0.98, ft.Colors.ON_SURFACE_VARIANT),
                        border_radius=10,
                        animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT)
                    ),
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click
                    ),
                ],
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER
            ),
            expand=True
        )
        
        # Add to page overlay
        self.page.overlay.extend([overlay, query_popup])
        self.page.update()
    
    def show_training_page(self, e):
        """Show model training page."""
        from .training_components import TrainingUI
        
        def close_page(e):
            """Close training page."""
            if training_popup in self.page.overlay:
                self.page.overlay.remove(training_popup)
            if overlay in self.page.overlay:
                self.page.overlay.remove(overlay)
            self.page.update()
        
        def close_on_overlay_click(e):
            """Only close if clicking on the overlay itself."""
            if e.target == overlay:
                close_page(e)
        
        # Create semi-transparent overlay
        overlay = ft.Container(
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.BLACK),
            on_click=close_on_overlay_click
        )
        
        # Create training page popup
        training_content = ft.Card(
            content=TrainingUI.create_training_page(self, on_close=close_page)
        )
        
        # Center the content
        training_popup = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click
                    ),
                    ft.Container(
                        content=training_content,
                        width=1200,
                        height=800,
                        bgcolor=ft.Colors.with_opacity(0.98, ft.Colors.ON_SURFACE_VARIANT),
                        border_radius=10,
                        animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT)
                    ),
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click
                    ),
                ],
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER
            ),
            expand=True
        )
        
        # Add to page overlay
        self.page.overlay.extend([overlay, training_popup])
        self.page.update()
    
    def show_personal_ai_page(self, e):
        """Show Personal AI Training page (جدید)."""
        
        def close_page():
            """Close personal AI page."""
            if personal_ai_popup in self.page.overlay:
                self.page.overlay.remove(personal_ai_popup)
            if overlay in self.page.overlay:
                self.page.overlay.remove(overlay)
            self.page.update()
        
        def close_on_overlay_click(e):
            """Only close if clicking on the overlay itself."""
            if e.target == overlay:
                close_page()
        
        # Create semi-transparent overlay
        overlay = ft.Container(
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.BLACK),
            on_click=close_on_overlay_click
        )
        
        # Create personal AI page
        personal_ai_content = ft.Card(
            content=TrainPersonalAI.create_page(self, on_close=close_page)
        )
        
        # Center the content
        personal_ai_popup = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click
                    ),
                    ft.Container(
                        content=personal_ai_content,
                        width=900,
                        height=800,
                        bgcolor=ft.Colors.with_opacity(0.98, ft.Colors.ON_SURFACE_VARIANT),
                        border_radius=10,
                        animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT)
                    ),
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click
                    ),
                ],
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER
            ),
            expand=True
        )
        
        # Add to page overlay
        self.page.overlay.extend([overlay, personal_ai_popup])
        self.page.update()
    
    def show_use_model_page(self, e):
        """Show Use Trained Model page."""
        
        def close_page():
            """Close model usage page."""
            if use_model_popup in self.page.overlay:
                self.page.overlay.remove(use_model_popup)
            if overlay in self.page.overlay:
                self.page.overlay.remove(overlay)
            self.page.update()
        
        def close_on_overlay_click(e):
            """Only close if clicking on the overlay itself."""
            if e.target == overlay:
                close_page()
        
        # Create semi-transparent overlay
        overlay = ft.Container(
            expand=True,
            bgcolor=ft.Colors.with_opacity(0.7, ft.Colors.BLACK),
            on_click=close_on_overlay_click
        )
        
        # Create use model page
        use_model_content = ft.Card(
            content=UseTrainedModel.create_page(self, on_close=close_page)
        )
        
        # Center the content
        use_model_popup = ft.Container(
            content=ft.Row(
                controls=[
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click
                    ),
                    ft.Container(
                        content=use_model_content,
                        width=900,
                        height=800,
                        bgcolor=ft.Colors.with_opacity(0.98, ft.Colors.ON_SURFACE_VARIANT),
                        border_radius=10,
                        animate=ft.Animation(300, ft.AnimationCurve.EASE_OUT)
                    ),
                    ft.Container(
                        expand=True,
                        on_click=close_on_overlay_click
                    ),
                ],
                expand=True,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER
            ),
            expand=True
        )
        
        # Add to page overlay
        self.page.overlay.extend([overlay, use_model_popup])
        self.page.update()
    
    
    # ========================================================================
    # CHAT MANAGEMENT
    # ========================================================================
    
    def toggle_chat_view(self, e=None):
        """Toggle between main toolbar and chat management view."""
        self.is_chat_view = not self.is_chat_view
        
        if hasattr(self, 'chat_toolbar'):
            self.chat_toolbar.visible = self.is_chat_view
            self.chat_toolbar.update()

            main_layout = self.page.controls[0]
            if hasattr(main_layout, 'controls') and len(main_layout.controls) > 1 and isinstance(main_layout, ft.Stack):
                main_content_container = main_layout.controls[1]
            else:
                main_content_container = main_layout

            if hasattr(main_content_container, 'content') and isinstance(main_content_container.content, ft.Row):
                main_row = main_content_container.content
                if len(main_row.controls) > 2 and isinstance(main_row.controls[2], ft.Column):
                    main_column = main_row.controls[2]
                    if len(main_column.controls) > 0 and isinstance(main_column.controls[0], ft.Container):
                        top_bar_container = main_column.controls[0]
                        if hasattr(top_bar_container, 'content') and isinstance(top_bar_container.content, ft.Row):
                            top_bar = top_bar_container
                            if self.is_chat_view:
                                menu_b64 = self.assets.get_icon('menu', 32)
                                if menu_b64:
                                    menu_btn = ft.IconButton(
                                        content=ft.Image(src_base64=menu_b64, width=32, height=32),
                                        tooltip="Show Main Menu",
                                        on_click=self.toggle_chat_view
                                    )
                                else:
                                    menu_btn = ft.IconButton(
                                        icon=ft.Icons.MENU,
                                        icon_size=32,
                                        tooltip="Show Main Menu",
                                        on_click=self.toggle_chat_view
                                    )

                                top_bar.content.controls = [
                                    menu_btn,
                                    ft.Container(expand=True),
                                    ft.Text(f"Chats: {len(self.chat_histories)}/3", 
                                           size=12, 
                                           color=ft.Colors.GREY_500)
                                ]
                            else:
                                chatmenu_b64 = self.assets.get_icon('chatmenu', 32)
                                if chatmenu_b64:
                                    manage_btn = ft.IconButton(
                                        content=ft.Image(src_base64=chatmenu_b64, width=32, height=32),
                                        tooltip="Manage Chats",
                                        on_click=self.toggle_chat_view
                                    )
                                else:
                                    manage_btn = ft.IconButton(
                                        icon=ft.Icons.CHAT,
                                        icon_size=32,
                                        tooltip="Manage Chats",
                                        on_click=self.toggle_chat_view
                                    )

                                top_bar.content.controls = [
                                    self.model_icon_container,
                                    self.model_dropdown,
                                    ft.Container(expand=True),
                                    manage_btn,
                                    self.status_text
                                ]
                            top_bar.update()

        self.page.update()
    
    def new_chat(self, e=None):
        """Start new chat."""
        if len(self.chat_histories) >= 3:
            self.show_snackbar("Maximum number of chats reached (3)")
            return
            
        if self.history:
            for chat in self.chat_histories:
                if chat["id"] == self.active_chat_id:
                    chat["history"] = self.history.copy()
                    break
        
        new_chat_id = len(self.chat_histories) + 1
        welcome_msg = {"role": "assistant", "content": "Hello! I'm iAI, your local AI assistant."}
        self.chat_histories.append({"id": new_chat_id, "history": []})
        
        self.chat_view.controls.clear()
        self.update_chat_toolbar()
        self.switch_to_chat(new_chat_id)
        self.add_message("assistant", welcome_msg["content"])
        
        if not self.is_chat_view:
            self.toggle_chat_view()
        
        self.page.update()
    
    def clear_chat(self, e):
        """Clear chat history."""
        self.chat_view.controls.clear()
        self.history = []
        self.add_message("system", " Chat cleared! ")
        self.page.update()
    
    def on_model_change(self, e):
        """Handle model selection change."""
        self.current_model = e.control.value
        self.config.set('model', self.current_model)
        
        if self.model_icon_container:
            suffix = ".light" if self.theme == "light" else ""
            icon_name = "model_two" if self.current_model not in [None, "Loading...", "No models"] else "model_one"
            model_icon = self.assets.get_icon(f"{icon_name}{suffix}", 32)
            if model_icon:
                self.model_icon_container.content = ft.Image(
                    src_base64=model_icon,
                    width=32,
                    height=32
                )
                self.model_icon_container.update()
        
        self.add_message("system", f"Switched to: {self.current_model}")
    
    # ========================================================================
    # DIALOGS
    # ========================================================================
    
    def open_settings(self, e):
        """Open settings popup overlay."""
        try:
            overlay, popup = UIComponents.create_settings_popup(self)
            self.page.overlay.extend([overlay, popup])
            self.page.update()
        except Exception as err:
            print(f"[App] Error opening settings: {err}")
            import traceback
            traceback.print_exc()
    
    def show_about(self, e):
        """Show about popup."""
        overlay, popup = UIComponents.create_about_popup(self)
        self.page.overlay.extend([overlay, popup])
        self.page.update()
    
    def save_settings(self):
        """Save settings."""
        print(f"[Debug] Saving settings - Temperature: {self.temperature}, Context Length: {self.context_length}")
        self.config.set('temperature', self.temperature)
        self.config.set('context_length', self.context_length)
        self.config.set('theme', self.theme)
        self.config.save()
        
        self.page.controls.clear()
        self.build_ui()
        
        temp_history = self.history.copy()
        self.history = []
        for msg in temp_history:
            self.add_message(msg['role'], msg['content'])
        
        self.show_snackbar("Settings saved!")
        if self.ollama.is_running():
            try:
                self.refresh_models()
            except Exception:
                pass

        self.page.update()
    
    # ========================================================================
    # HELPERS
    # ========================================================================
    
    def switch_to_chat(self, chat_id: int):
        """Switch to a different chat."""
        if self.active_chat_id == chat_id:
            return
            
        if self.history:
            for chat in self.chat_histories:
                if chat["id"] == self.active_chat_id:
                    chat["history"] = self.history.copy()
                    break
        
        self.active_chat_id = chat_id
        self.history = []
        
        current_chat = next((chat for chat in self.chat_histories if chat["id"] == chat_id), None)
        if current_chat:
            self.history = current_chat["history"].copy() if current_chat["history"] else []
        
        self.chat_view.controls.clear()
        
        if self.history:
            for msg in self.history:
                self.add_message(msg["role"], msg["content"], update_page=False)
        
        self.update_chat_toolbar()
        self.page.update()
    
    def update_chat_toolbar(self):
        """Update the chat toolbar UI."""
        if not hasattr(self, 'chat_toolbar'):
            return
            
        self.chat_toolbar.controls.clear()
        
        for chat in self.chat_histories:
            chat_id = chat["id"]
            suffix = ".light" if self.theme == "light" else ""
            
            chat_icon = self.assets.get_icon(f"chat{chat_id}{suffix}", 32)
            if not chat_icon:
                chat_icon = self.assets.get_icon(f"chat{suffix}", 32)
            
            button = ft.IconButton(
                content=ft.Image(src_base64=chat_icon, width=32, height=32) if chat_icon else None,
                icon=ft.Icons.CHAT if not chat_icon else None,
                icon_size=32,
                tooltip=f"Chat {chat_id}",
                data=str(chat_id),
                on_click=lambda e: self.switch_to_chat(int(e.control.data)),
                style=ft.ButtonStyle(
                    color=ft.Colors.BLUE if chat_id == self.active_chat_id else None,
                    overlay_color=ft.Colors.with_opacity(0.1, ft.Colors.BLUE) if chat_id == self.active_chat_id else None
                )
            )
            self.chat_toolbar.controls.append(button)
        
        self.chat_toolbar.visible = self.is_chat_view
        self.chat_toolbar.update()

    def add_message(self, role: str, text: str, update_page=True):
        """Add message to chat."""
        color, text_color, alignment = ThemeManager.get_message_colors(role, self.theme)
        
        bubble = ft.Container(
            content=ft.Text(text, color=text_color, size=14, selectable=True),
            padding=15,
            bgcolor=color,
            border_radius=15,
            width=600
        )
        
        self.chat_view.controls.append(
            ft.Row(controls=[bubble], alignment=alignment)
        )
        
        if update_page:
            self.page.update()
    
    def update_status(self, status: str):
        """Update status text."""
        if self.status_text:
            self.status_text.value = status
            self.page.update()
    
    def show_snackbar(self, message: str):
        """Show snackbar notification."""
        self.page.snack_bar = ft.SnackBar(content=ft.Text(message))
        self.page.snack_bar.open = True
        self.page.update()
    
    def on_resize(self, e):
        """Handle window resize event."""
        if self.page.controls and isinstance(self.page.controls[0], ft.Stack):
            main_stack = self.page.controls[0]
            if len(main_stack.controls) > 1:
                main_stack.controls[0].update()
    
    def on_close(self, e):
        """Handle window close."""
        self.config.save()
        if self.ollama.process:
            self.ollama.stop()