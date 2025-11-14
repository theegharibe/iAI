"""Book Training UI Component - Upload book and select training method."""

import flet as ft
import threading
from pathlib import Path


class BookTrainingUI:
    """UI for book-based training (RAG/Fine-tuning/Hybrid)."""
    
    @staticmethod
    def create_book_training_section(app, on_close=None):
        """
        Create book training UI section.
        
        Args:
            app: Application instance (has book_trainer)
            on_close: Callback when done
            
        Returns:
            ft.Container with book training UI
        """
        
        # Initialize book_trainer if not exists
        if not hasattr(app, 'book_trainer') or app.book_trainer is None:
            from managers.book_trainer import BookTrainer
            app.book_trainer = BookTrainer(
                knowledge_manager=app.knowledge if hasattr(app, 'knowledge') else None,
                lora_trainer=app.lora_trainer if hasattr(app, 'lora_trainer') else None
            )
        
        book_trainer = app.book_trainer
        
        # Colors
        is_light = app.theme == "light"
        text_primary = ft.Colors.BLACK if is_light else ft.Colors.WHITE
        text_secondary = ft.Colors.GREY_700 if is_light else ft.Colors.GREY_300
        bg_primary = ft.Colors.WHITE if is_light else ft.Colors.SURFACE
        bg_hover = ft.Colors.GREY_100 if is_light else ft.Colors.GREY_900
        
        # State
        state = {
            'book_loaded': False,
            'book_name': '',
            'method': 'rag',
            'training': False
        }
        
        # ==================== FILE PICKER ====================
        file_picker = ft.FilePicker()
        app.page.overlay.append(file_picker)
        
        # ==================== STEP 1: UPLOAD BOOK ====================
        upload_status = ft.Text("No book loaded", size=12, color=text_secondary)
        
        def on_book_selected(e):
            """Handle book file selection."""
            if not e.files:
                return
            
            file_path = e.files[0].path
            upload_status.value = "Uploading..."
            upload_status.color = ft.Colors.ORANGE
            app.page.update()
            
            def upload_thread():
                try:
                    result = book_trainer.upload_book(file_path)
                    
                    if result['success']:
                        state['book_loaded'] = True
                        state['book_name'] = result['book_name']
                        upload_status.value = (
                            f"[OK] {result['book_name']} ({result['pages']} pages, "
                            f"{result['text_length']//1000}KB)"
                        )
                        upload_status.color = ft.Colors.GREEN
                        method_radio.disabled = False
                        train_btn.disabled = False
                    else:
                        upload_status.value = f"[ERROR] {result['message']}"
                        upload_status.color = ft.Colors.RED
                
                except Exception as ex:
                    upload_status.value = f"[ERROR] Error: {str(ex)[:50]}"
                    upload_status.color = ft.Colors.RED
                
                app.page.update()
            
            threading.Thread(target=upload_thread, daemon=True).start()
        
        file_picker.on_result = on_book_selected
        
        upload_btn = ft.ElevatedButton(
            "📚 Select PDF Book",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=lambda e: file_picker.pick_files(allowed_extensions=['pdf']),
            expand=True,
            height=45
        )
        
        upload_section = ft.Container(
            content=ft.Column([
                ft.Text("Step 1: Upload Book", size=14, weight="bold", color=text_primary),
                ft.Row([upload_btn], spacing=10),
                upload_status
            ], spacing=10),
            padding=15,
            bgcolor=bg_primary,
            border_radius=8
        )
        
        # ==================== STEP 2: SELECT METHOD ====================
        method_radio = ft.RadioGroup(value='rag')
        
        rag_radio = ft.Radio(
            value='rag',
            label='RAG Only (Fast - 2 min)',
            label_position=ft.LabelPosition.RIGHT,
            fill_color=ft.Colors.BLUE
        )
        
        ft_radio = ft.Radio(
            value='finetuning',
            label='Fine-tuning (Accurate - 20 min)',
            label_position=ft.LabelPosition.RIGHT,
            fill_color=ft.Colors.BLUE
        )
        
        hybrid_radio = ft.Radio(
            value='hybrid',
            label='Hybrid (Best - 25 min)',
            label_position=ft.LabelPosition.RIGHT,
            fill_color=ft.Colors.BLUE
        )
        
        method_radio.controls = [rag_radio, ft_radio, hybrid_radio]
        method_radio.disabled = True
        
        def on_method_change(e):
            state['method'] = method_radio.value
            book_trainer.set_training_method(state['method'])
        
        method_radio.on_change = on_method_change
        
        method_section = ft.Container(
            content=ft.Column([
                ft.Text("Step 2: Select Training Method", size=14, weight="bold", color=text_primary),
                ft.Text(
                    "RAG: Fast, search-based | Fine-tuning: Accurate, model learns | "
                    "Hybrid: Both",
                    size=10,
                    color=text_secondary,
                    italic=True
                ),
                method_radio
            ], spacing=10),
            padding=15,
            bgcolor=bg_primary,
            border_radius=8
        )
        
        # ==================== STEP 3: PROGRESS ====================
        progress_bar = ft.ProgressBar(value=0, visible=False)
        progress_text = ft.Text("Ready to train", size=11, color=text_secondary)
        progress_log = ft.Column(scroll=ft.ScrollMode.AUTO, height=100, spacing=5)
        
        def update_progress(progress, message):
            """Update progress display."""
            progress_bar.value = progress / 100
            progress_text.value = message
            
            # Add to log (max 10 items)
            if len(progress_log.controls) >= 10:
                progress_log.controls.pop(0)
            
            progress_log.controls.append(
                ft.Text(f"• {message}", size=9, color=text_secondary)
            )
            
            app.page.update()
        
        progress_section = ft.Container(
            content=ft.Column([
                ft.Text("Training Progress", size=14, weight="bold", color=text_primary),
                progress_bar,
                progress_text,
                ft.Divider(height=10),
                progress_log
            ], spacing=10),
            padding=15,
            bgcolor=bg_primary,
            border_radius=8,
            visible=False
        )
        
        # ==================== STEP 4: TRAIN BUTTON ====================
        def start_training(e):
            """Start training."""
            if not state['book_loaded']:
                app.show_snackbar("[ERROR] Please load a book first")
                return
            
            state['training'] = True
            train_btn.disabled = True
            progress_section.visible = True
            progress_bar.value = 0.01
            progress_bar.visible = True
            progress_text.value = "Starting..."
            progress_log.controls.clear()
            app.page.update()
            
            def training_thread():
                try:
                    method = state['method']
                    
                    if method == 'rag':
                        result = book_trainer.train_with_rag(
                            progress_callback=update_progress
                        )
                    elif method == 'finetuning':
                        result = book_trainer.train_with_finetuning(
                            progress_callback=update_progress
                        )
                    else:  # hybrid
                        result = book_trainer.train_hybrid(
                            progress_callback=update_progress
                        )
                    
                    if result['success']:
                        progress_text.value = f"[OK] {result['message']}"
                        progress_text.color = ft.Colors.GREEN
                        app.show_snackbar(f"[OK] Training complete!")
                    else:
                        progress_text.value = f"[ERROR] {result['message']}"
                        progress_text.color = ft.Colors.RED
                        app.show_snackbar(f"[ERROR] {result['message']}")
                
                except Exception as ex:
                    progress_text.value = f"[ERROR] Error: {str(ex)[:50]}"
                    progress_text.color = ft.Colors.RED
                    app.show_snackbar(f"[ERROR] {str(ex)[:50]}")
                
                finally:
                    state['training'] = False
                    train_btn.disabled = False
                    app.page.update()
            
            threading.Thread(target=training_thread, daemon=True).start()
        
        train_btn = ft.ElevatedButton(
            "🚀 Start Training",
            icon=ft.Icons.PLAY_ARROW,
            on_click=start_training,
            expand=True,
            height=50,
            disabled=True,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN)
        )
        
        train_section = ft.Container(
            content=ft.Column([
                ft.Text("Step 3: Start Training", size=14, weight="bold", color=text_primary),
                train_btn
            ], spacing=10),
            padding=15,
            bgcolor=bg_primary,
            border_radius=8
        )
        
        # ==================== MAIN LAYOUT ====================
        def close_handler(e=None):
            if on_close:
                on_close()
        
        close_btn = ft.IconButton(ft.Icons.CLOSE, on_click=close_handler)
        
        header = ft.Container(
            content=ft.Row([
                ft.Text("📖 Train with Book", size=24, weight="bold", color=text_primary),
                ft.Container(expand=True),
                close_btn
            ]),
            padding=20,
            bgcolor=bg_hover
        )
        
        main_content = ft.Column([
            header,
            ft.Container(
                content=ft.Column([
                    upload_section,
                    method_section,
                    train_section,
                    progress_section
                ], spacing=15, scroll=ft.ScrollMode.AUTO),
                expand=True,
                padding=15
            )
        ], expand=True)
        
        return ft.Container(
            content=main_content,
            expand=True,
            bgcolor=app.page.bgcolor
        )
