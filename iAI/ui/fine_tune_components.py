"""Fine-tuning UI components - SIMPLIFIED VERSION."""

import flet as ft
from pathlib import Path


class FineTuneUI:
    """UI components for fine-tuning."""
    
    @staticmethod
    def create_fine_tune_page(app, on_close=None):
        """
        Create fine-tuning management page.
        
        Usage Guide:
        ============
        1. Add training data first:
           - Question (Instruction)
           - Answer (Output)
           - Click "Add Pair"
        
        2. Add at least 5 examples
        
        3. Click "Start Training"
        
        Buttons:
        ========
        - "Add Pair": Add a new question-answer pair
        - "Refresh": Refresh the samples list
        - "From Knowledge Base": Import KB data
        - "Export Data": Save existing data
        - "Clear All": Delete all data (with confirmation)
        - "Start Training": Start model training (requires 5+ examples)
        """
        fine_tune = app.fine_tune
        
        # Colors
        is_light = app.theme == "light"
        text_primary = ft.Colors.BLACK if is_light else ft.Colors.WHITE
        text_secondary = ft.Colors.GREY_700 if is_light else ft.Colors.GREY_300
        
        # ========================= HEADER =========================
        def close_handler(e):
            if on_close:
                on_close(e)
        
        title = ft.Text("Fine-Tuning Manager", size=24, weight="bold", color=text_primary)
        close_btn = ft.IconButton(ft.Icons.CLOSE, tooltip="Close", on_click=close_handler)
        header = ft.Container(
            content=ft.Row([title, ft.Container(expand=True), close_btn]),
            padding=20
        )
        
        # ========================= STATS =========================
        stats = fine_tune.get_training_stats()
        
        stat_text = f"""
📊 Statistics:
  • Total Samples: {stats['total_samples']}
  • Total Tokens: {stats['total_tokens_estimate']}
  • Avg Tokens/Sample: {stats['avg_tokens_per_sample']}
        """
        
        stats_display = ft.Text(stat_text, size=12, color=text_secondary)
        
        # ========================= SAMPLES LIST =========================
        samples_list = ft.Column(scroll=ft.ScrollMode.AUTO, spacing=10, expand=True)
        
        def refresh_samples_list():
            """Refresh the samples display."""
            samples_list.controls.clear()
            samples = fine_tune.list_training_samples(limit=20)
            
            if not samples:
                samples_list.controls.append(
                    ft.Text("No training examples yet", color=text_secondary, size=12)
                )
            else:
                for i, sample in enumerate(samples):
                    def make_delete_func(idx):
                        def delete_sample(e):
                            fine_tune.remove_sample(idx)
                            refresh_samples_list()
                            app.page.update()
                            app.show_snackbar(f"[OK] Example #{idx+1} deleted")
                        return delete_sample
                    
                    sample_text = f"Q: {sample['instruction'][:40]}...\nA: {sample['output'][:40]}..."
                    sample_row = ft.Row([
                        ft.Text(sample_text, size=11, color=text_primary, expand=True),
                        ft.IconButton(ft.Icons.DELETE, icon_size=20, 
                                    on_click=make_delete_func(i))
                    ], spacing=10)
                    
                    samples_list.controls.append(ft.Card(
                        content=ft.Container(sample_row, padding=10)
                    ))
        
        refresh_samples_list()
        
        # ========================= ADD PAIR SECTION =========================
        instruction_input = ft.TextField(
            label="Question",
            hint_text="e.g. What is 2+2?",
            min_lines=2,
            max_lines=4
        )
        
        output_input = ft.TextField(
            label="Answer",
            hint_text="e.g. The answer is 4",
            min_lines=2,
            max_lines=4
        )
        
        def add_pair_handler(e):
            """Handle adding a new pair."""
            instruction = instruction_input.value.strip()
            output = output_input.value.strip()
            
            if not instruction or not output:
                app.show_snackbar("[ERROR] Please fill both question and answer!")
                return
            
            if fine_tune.add_training_pair(instruction, "", output):
                app.show_snackbar(f"[OK] Example added successfully!")
                instruction_input.value = ""
                output_input.value = ""
                refresh_samples_list()
                # Update stats
                new_stats = fine_tune.get_training_stats()
                stats_display.value = f"""
📊 Statistics:
  • Total Samples: {new_stats['total_samples']}
  • Total Tokens: {new_stats['total_tokens_estimate']}
  • Avg Tokens/Sample: {new_stats['avg_tokens_per_sample']}
                """
                app.page.update()
            else:
                app.show_snackbar("[ERROR] Error adding example")
        
        add_btn = ft.ElevatedButton(
            "Add Pair",
            icon=ft.Icons.ADD,
            on_click=add_pair_handler,
            expand=True
        )
        
        refresh_btn = ft.IconButton(
            ft.Icons.REFRESH,
            tooltip="Refresh",
            on_click=lambda e: (refresh_samples_list(), app.page.update())
        )
        
        add_section = ft.Column([
            ft.Text("➕ Add New Training Pair", size=14, weight="bold", color=text_primary),
            instruction_input,
            output_input,
            ft.Row([add_btn, refresh_btn], spacing=10)
        ], spacing=10)
        
        # ========================= DATA MANAGEMENT =========================
        def export_handler(e):
            """Export training data."""
            try:
                path = "data/fine_tune/exported_data.json"
                if fine_tune.export_training_data(path):
                    app.show_snackbar(f"[OK] Data saved to: {path}")
                else:
                    app.show_snackbar("[ERROR] Error saving data")
            except Exception as ex:
                app.show_snackbar(f"[ERROR] {str(ex)}")
        
        def clear_handler(e):
            """Clear all data with confirmation."""
            def on_confirm(e=None):
                fine_tune.clear_training_data()
                refresh_samples_list()
                app.show_snackbar("[OK] All data cleared")
                app.page.update()
            
            def close_dlg(e=None):
                dlg.open = False
                app.page.update()
            
            dlg = ft.AlertDialog(
                title=ft.Text("Delete All Data?"),
                content=ft.Text("This action cannot be undone!"),
                actions=[
                    ft.TextButton("Cancel", on_click=close_dlg),
                    ft.TextButton("Delete", on_click=lambda e: (on_confirm(), close_dlg()))
                ]
            )
            app.page.dialog = dlg
            dlg.open = True
            app.page.update()
        
        def extract_kb_handler(e):
            """Extract from knowledge base."""
            try:
                if not app.knowledge.is_available():
                    app.show_snackbar("[ERROR] Knowledge Base not available")
                    return
                
                docs = app.knowledge.list_documents()
                if not docs:
                    app.show_snackbar("[ERROR] No documents in Knowledge Base")
                    return
                
                app.show_snackbar("🔄 Extracting...")
                added = fine_tune.add_from_knowledge_base(app.knowledge)
                refresh_samples_list()
                app.show_snackbar(f"[OK] {added} examples added!")
                app.page.update()
            except Exception as ex:
                app.show_snackbar(f"[ERROR] {str(ex)}")
        
        def upload_file_handler(e):
            """Handle file upload (JSON, PDF, TXT)."""
            import json
            import os
            
            if not e.files:
                app.show_snackbar("[ERROR] No file selected")
                return
            
            try:
                file_path = e.files[0].path
                file_name = os.path.basename(file_path)
                file_ext = os.path.splitext(file_name)[1].lower()
                
                app.show_snackbar(f"🔄 Processing: {file_name}")
                
                # ==================== JSON ====================
                if file_ext == '.json':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if isinstance(data, list):
                        pairs = data
                    elif isinstance(data, dict) and 'data' in data:
                        pairs = data['data']
                    else:
                        pairs = [data]
                    
                    added_count = 0
                    for item in pairs:
                        instruction = item.get('instruction', '') or item.get('question', '')
                        output = item.get('output', '') or item.get('answer', '')
                        
                        if instruction and output:
                            fine_tune.add_training_pair(instruction, "", output)
                            added_count += 1
                
                # ==================== TXT ====================
                elif file_ext == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Split by double newlines (paragraphs)
                    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
                    
                    # Create Q&A from paragraphs
                    added_count = 0
                    for i, para in enumerate(paragraphs):
                        if len(para) > 20:  # Only add if paragraph is meaningful
                            # Use first sentence as question
                            sentences = para.split('.')
                            question = sentences[0].strip()
                            answer = para.strip()
                            
                            if question and answer and len(question) > 5:
                                fine_tune.add_training_pair(question + "?", "", answer)
                                added_count += 1
                
                # ==================== PDF ====================
                elif file_ext == '.pdf':
                    try:
                        import PyPDF2
                        added_count = 0
                        
                        with open(file_path, 'rb') as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            
                            for page_num, page in enumerate(pdf_reader.pages):
                                text = page.extract_text()
                                
                                # Split into paragraphs
                                paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
                                
                                for para in paragraphs:
                                    if len(para) > 20:
                                        # Create Q&A from text
                                        sentences = para.split('.')
                                        question = sentences[0].strip()
                                        answer = para.strip()
                                        
                                        if question and answer and len(question) > 5:
                                            fine_tune.add_training_pair(question + "?", "", answer)
                                            added_count += 1
                        
                    except ImportError:
                        app.show_snackbar("[ERROR] PyPDF2 not installed. Install: pip install PyPDF2")
                        return
                
                # ==================== DOCX ====================
                elif file_ext == '.docx':
                    try:
                        from docx import Document
                        added_count = 0
                        
                        doc = Document(file_path)
                        
                        for para in doc.paragraphs:
                            text = para.text.strip()
                            
                            if len(text) > 20:
                                sentences = text.split('.')
                                question = sentences[0].strip()
                                answer = text.strip()
                                
                                if question and answer and len(question) > 5:
                                    fine_tune.add_training_pair(question + "?", "", answer)
                                    added_count += 1
                    
                    except ImportError:
                        app.show_snackbar("[ERROR] python-docx not installed. Install: pip install python-docx")
                        return
                
                else:
                    app.show_snackbar(f"[ERROR] Unsupported file type: {file_ext}\nSupported: JSON, TXT, PDF, DOCX")
                    return
                
                # Update display
                refresh_samples_list()
                new_stats = fine_tune.get_training_stats()
                stats_display.value = f"""
📊 Statistics:
  • Total Samples: {new_stats['total_samples']}
  • Total Tokens: {new_stats['total_tokens_estimate']}
  • Avg Tokens/Sample: {new_stats['avg_tokens_per_sample']}
                """
                app.show_snackbar(f"[OK] {added_count} examples extracted and added!")
                app.page.update()
                
            except json.JSONDecodeError:
                app.show_snackbar("[ERROR] Invalid JSON format")
            except Exception as ex:
                app.show_snackbar(f"[ERROR] Error: {str(ex)}")
        
        # File picker - allow multiple file types
        file_picker = ft.FilePicker(on_result=upload_file_handler)
        app.page.overlay.append(file_picker)
        
        # Theme-aware custom icons for data management buttons
        suffix = ".light" if app.theme == "light" else ""
        kb_icon = app.assets.get_icon(f"knowledge_base{suffix}", 24)
        export_icon = app.assets.get_icon(f"export{suffix}", 24)
        delete_icon = app.assets.get_icon(f"delete_all{suffix}", 24)
        
        export_btn = ft.ElevatedButton(
            "Export Data",
            icon=ft.Icons.DOWNLOAD,
            on_click=export_handler,
            expand=True
        )
        
        clear_btn = ft.ElevatedButton(
            "Clear All",
            icon=ft.Icons.DELETE_FOREVER,
            on_click=clear_handler,
            expand=True
        )
        
        kb_btn = ft.ElevatedButton(
            "From KB",
            icon=ft.Icons.LIBRARY_BOOKS,
            on_click=extract_kb_handler,
            expand=True
        )
        
        upload_btn = ft.ElevatedButton(
            "Upload File",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=lambda e: file_picker.pick_files(
                allowed_extensions=['json', 'txt', 'pdf', 'docx']
            ),
            expand=True
        )
        
        data_section = ft.Column([
            ft.Text("📁 Data Management", size=14, weight="bold", color=text_primary),
            ft.Text("Supported: JSON, TXT, PDF, DOCX", size=10, color=text_secondary),
            ft.Row([kb_btn, export_btn, clear_btn], spacing=10),
            ft.Row([upload_btn], spacing=10)
        ], spacing=10)
        
        # ========================= TRAINING =========================
        def train_handler(e):
            """Start training."""
            current_stats = fine_tune.get_training_stats()
            
            if current_stats['total_samples'] < 5:
                app.show_snackbar(f"[ERROR] Need at least 5 examples (current: {current_stats['total_samples']})")
                return
            
            app.show_snackbar("🚀 Training started... (this may take a few minutes)")
            
            try:
                result = fine_tune.start_training(epochs=3, batch_size=4, learning_rate=1e-4)
                if result['success']:
                    app.show_snackbar(f"[OK] Training completed! {result.get('message', '')}")
                else:
                    app.show_snackbar(f"[ERROR] {result.get('error', 'Training failed')}")
            except Exception as ex:
                app.show_snackbar(f"[ERROR] {str(ex)}")
        
        train_btn = ft.ElevatedButton(
            "🚀 Start Training",
            on_click=train_handler,
            style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN, color=ft.Colors.WHITE),
            expand=True
        )
        
        training_section = ft.Column([
            ft.Text("🤖 Fine-Tune Model", size=14, weight="bold", color=text_primary),
            ft.Text("Minimum 5 examples required to start training", size=11, color=text_secondary),
            train_btn
        ], spacing=10)
        
        # ========================= MAIN LAYOUT =========================
        main_content = ft.Column([
            header,
            ft.Divider(),
            ft.Text("📊 Statistics", size=14, weight="bold", color=text_primary),
            stats_display,
            ft.Divider(),
            ft.Text("📚 Training Samples", size=14, weight="bold", color=text_primary),
            samples_list,
            ft.Divider(),
            add_section,
            ft.Divider(),
            data_section,
            ft.Divider(),
            training_section
        ], expand=True, spacing=15, scroll=ft.ScrollMode.AUTO)
        
        return main_content
