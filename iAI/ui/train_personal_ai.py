"""Personal AI training page - new and simple."""

import flet as ft
from pathlib import Path
import threading


class TrainPersonalAI:
    """Personal AI training interface."""
    
    @staticmethod
    def create_page(app, on_close=None):
        """Main training page."""
        
        lora_trainer = app.lora_trainer
        
        # Colors
        is_light = app.theme == "light"
        text_primary = ft.Colors.BLACK if is_light else ft.Colors.WHITE
        text_secondary = ft.Colors.GREY_700 if is_light else ft.Colors.GREY_300
        bg_primary = ft.Colors.WHITE if is_light else ft.Colors.SURFACE
        bg_hover = ft.Colors.GREY_100 if is_light else ft.Colors.GREY_900
        
        # State
        state = {
            'selected_model': None,
            'model_path': None,
            'model_source': None,  # 'ollama', 'huggingface', 'user'
            'documents': [],
            'training': False,
            'download_progress': 0
        }
        
        # ==================== HEADER ====================
        def close_handler(e):
            if on_close:
                on_close()
        
        suffix = ".light" if app.theme == "light" else ""
        close_icon = app.assets.get_icon(f"close{suffix}", 24)
        
        close_btn = None
        if close_icon:
            close_btn = ft.IconButton(
                content=ft.Image(src_base64=close_icon, width=24, height=24),
                on_click=close_handler,
                tooltip="Close"
            )
        else:
            close_btn = ft.IconButton(ft.Icons.CLOSE, on_click=close_handler)
        
        header = ft.Container(
            content=ft.Row([
                ft.Text("🤖 Train Personal AI", size=24, weight="bold", color=text_primary),
                ft.Container(expand=True),
                close_btn
            ]),
            padding=20,
            bgcolor=bg_hover
        )
        
        # ==================== SECTION 1: SELECT MODEL ====================
        
        model_section_title = ft.Text("🤖 AI Model: phi3:mini", size=16, weight="bold", color=text_primary)
        
        # Selected model display
        selected_model_display = ft.Text(
            "[OK] phi3:mini (Ollama)",
            size=12,
            color=ft.Colors.GREEN,
            italic=True
        )
        
        # Auto-select phi3:mini
        state['selected_model'] = 'phi3:mini'
        state['model_source'] = 'ollama'
        state['model_path'] = 'phi3:mini'
        
        # Model section - very simple, just show selected model
        model_section = ft.Container(
            content=ft.Column([
                model_section_title,
                selected_model_display,
                ft.Divider(),
                ft.Text("This model will be used for fine-tuning.", size=11, color=text_secondary, italic=True)
            ], spacing=10),
            padding=15,
            bgcolor=bg_primary,
            border_radius=8
        )
        
        # ==================== SECTION 2: UPLOAD DOCUMENTS ====================
        
        docs_section_title = ft.Text("2️⃣ Upload Documents", size=16, weight="bold", color=text_primary)
        
        documents_display = ft.Text("No documents uploaded", size=11, color=text_secondary, italic=True)
        
        file_picker = ft.FilePicker()
        
        def on_files_selected(e):
            """When files are selected."""
            if e.files and len(e.files) > 0:
                state['documents'] = [f.path for f in e.files]
                doc_names = [Path(f.path).name for f in e.files]
                doc_list = ", ".join(doc_names[:3])
                if len(doc_names) > 3:
                    doc_list += f" (+{len(doc_names)-3} more)"
                documents_display.value = f"[OK] {len(doc_names)} files: {doc_list}"
                documents_display.color = ft.Colors.GREEN
            else:
                documents_display.value = "No documents uploaded"
                documents_display.color = ft.Colors.GREY_500
                state['documents'] = []
            app.page.update()
        
        file_picker.on_result = on_files_selected
        
        select_docs_btn = ft.ElevatedButton(
            text="📄 Select Documents (PDF, TXT, DOCX)",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=lambda e: file_picker.pick_files(
                allowed_extensions=['pdf', 'txt', 'docx'],
                allow_multiple=True
            ),
            expand=True,
            height=45
        )
        
        docs_section = ft.Container(
            content=ft.Column([
                docs_section_title,
                documents_display,
                select_docs_btn
            ], spacing=10),
            padding=15,
            bgcolor=bg_primary,
            border_radius=8
        )
        
        # ==================== SECTION 3: TRAINING ====================
        
        train_section_title = ft.Text("3️⃣ Train AI", size=16, weight="bold", color=text_primary)
        
        training_status = ft.Text("Ready to train", size=11, color=text_secondary)
        training_progress = ft.ProgressBar(value=0, visible=False)
        training_log = ft.Text("", size=10, color=text_secondary)
        
        def start_training(e):
            """Start training."""
            if not state['selected_model']:
                training_status.value = "No model selected"
                training_status.color = ft.Colors.RED
                app.page.update()
                return
            
            if not state['documents']:
                training_status.value = "No documents uploaded"
                training_status.color = ft.Colors.RED
                app.page.update()
                return
            
            state['training'] = True
            training_status.value = "Starting training in background..."
            training_status.color = ft.Colors.ORANGE
            training_progress.visible = True
            training_progress.value = 0
            train_btn.disabled = True
            app.page.update()
            
            def training_ui_thread():
                """
                UI thread that monitors training progress.
                No timeout, just displays updates.
                """
                try:
                    # Start training in background
                    start_result = lora_trainer.train_from_documents(
                        documents=state['documents'],
                        epochs=3,
                        batch_size=4,
                        learning_rate=5e-4
                    )
                    
                    if not start_result.get('success'):
                        training_status.value = f"Failed to start: {start_result.get('message', 'Unknown error')}"
                        training_status.color = ft.Colors.RED
                        app.page.update()
                        return
                    
                    # Training started, now monitor
                    training_status.value = "Training in background (no timeout)..."
                    training_status.color = ft.Colors.ORANGE
                    app.page.update()
                    
                    # Monitor progress
                    import time
                    last_log_count = 0
                    
                    while lora_trainer.training_in_progress:
                        # Get current status (this is fast, no blocking)
                        status = lora_trainer.get_training_status()
                        
                        training_progress.value = status['progress'] / 100
                        
                        # Update log if new entries
                        if len(status['log']) > last_log_count:
                            training_log.value = '\n'.join(status['log'])
                            last_log_count = len(status['log'])
                        
                        app.page.update()
                        time.sleep(0.5)  # Check every 500ms
                    
                    # Training done
                    final_result = lora_trainer.training_result
                    
                    if final_result and final_result.get('success'):
                        training_status.value = "Training completed!"
                        training_status.color = ft.Colors.GREEN
                        training_log.value += f"\nOutput: {final_result.get('output_path', 'Unknown')}"
                        
                        # Try to merge model
                        try:
                            merge_result = lora_trainer.merge_model(
                                base_model_path=state['model_path'],
                                model_source=state.get('model_source', 'huggingface')
                            )
                            
                            if merge_result['success']:
                                training_status.value += "\nModel merged and ready!"
                                training_log.value += f"\nMerged: {merge_result.get('output_path', 'Unknown')}"
                            else:
                                training_log.value += f"\nMerge error: {merge_result['message']}"
                        except Exception as merge_err:
                            training_log.value += f"\nMerge error: {str(merge_err)}"
                        
                        # Add documents to QueryManager
                        try:
                            query_mgr = app.query
                            for doc_path in state['documents']:
                                query_mgr.add_document(doc_path)
                            training_log.value += "\nDocuments indexed for querying!"
                        except Exception as q_err:
                            training_log.value += f"\nIndexing note: {str(q_err)[:50]}"
                    else:
                        training_status.value = "Training failed"
                        training_status.color = ft.Colors.RED
                        training_log.value = final_result.get('message', 'Unknown error') if final_result else 'No result'
                    
                    training_progress.value = 1.0
                    
                except Exception as err:
                    training_status.value = f"Error: {str(err)[:50]}"
                    training_status.color = ft.Colors.RED
                    training_log.value = str(err)
                finally:
                    training_progress.visible = False
                    train_btn.disabled = False
                    state['training'] = False
                    app.page.update()
            
            threading.Thread(target=training_ui_thread, daemon=True).start()
        
        train_btn = ft.ElevatedButton(
            text="🚀 Start Training",
            on_click=start_training,
            expand=True,
            height=50,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN)
        )
        
        train_section = ft.Container(
            content=ft.Column([
                train_section_title,
                training_status,
                training_progress,
                training_log,
                train_btn
            ], spacing=10),
            padding=15,
            bgcolor=bg_primary,
            border_radius=8
        )
        
        # ==================== MAIN LAYOUT ====================
        
        content = ft.Column([
            header,
            ft.Container(
                content=ft.Column([
                    model_section,
                    docs_section,
                    train_section
                ], spacing=15, scroll=ft.ScrollMode.AUTO),
                expand=True,
                padding=15
            )
        ], expand=True)
        
        page = ft.Container(
            content=content,
            expand=True,
            bgcolor=app.page.bgcolor
        )
        
        # Add file picker to overlay
        app.page.overlay.append(file_picker)
        
        return page
