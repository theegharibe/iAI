"""Query components for document Q&A interface."""

import flet as ft
from pathlib import Path


class QueryUI:
    """Document Query UI for asking questions from uploaded documents."""
    
    @staticmethod
    def create_query_page(app, on_close, file_picker=None):
        """Create the document query page.
        
        Args:
            app: The main LocalAIChatApp instance
            on_close: Callback function when page is closed
            
        Returns:
            ft.Column: The query page UI
        """
        
        # State variables
        uploaded_docs = []
        
        def handle_file_pick(e):
            """Handle file selection for upload."""
            if e.files:
                for file in e.files:
                    try:
                        file_path = file.path
                        file_name = Path(file_path).name
                        
                        # Check if file exists
                        if not Path(file_path).exists():
                            if hasattr(app, 'show_snackbar'):
                                app.show_snackbar(f"[ERROR] File not found: {file_name}")
                            continue
                        
                        # Copy file to documents folder for persistence
                        import shutil
                        doc_dir = Path("data/fine_tune/documents")
                        doc_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Create persistent copy
                        persistent_path = doc_dir / file_name
                        shutil.copy2(file_path, persistent_path)
                        
                        # Add to list with persistent path
                        uploaded_docs.append({
                            'path': str(persistent_path),
                            'name': file_name
                        })
                        
                        # Add to UI list
                        doc_item = ft.Row(
                            controls=[
                                ft.Icon(ft.Icons.DESCRIPTION, color=ft.Colors.BLUE),
                                ft.Text(file_name, expand=True),
                                ft.IconButton(
                                    icon=ft.Icons.DELETE,
                                    icon_color=ft.Colors.RED,
                                    on_click=lambda e, idx=len(uploaded_docs)-1: remove_doc(idx)
                                )
                            ]
                        )
                        docs_list.controls.append(doc_item)
                        
                        if hasattr(app, 'show_snackbar'):
                            app.show_snackbar(f"[OK] {file_name} added and indexed")
                    except Exception as ex:
                        if hasattr(app, 'show_snackbar'):
                            app.show_snackbar(f"[ERROR] Error adding file: {str(ex)}")
                
                docs_list.update()
        
        def remove_doc(index):
            """Remove a document from the list."""
            if 0 <= index < len(uploaded_docs):
                uploaded_docs.pop(index)
                docs_list.controls.pop(index)
                docs_list.update()
        
        def handle_query(e):
            """Handle user query submission."""
            query_text = query_input.value.strip()
            
            if not query_text:
                show_error("Please enter a query")
                return
            
            if not uploaded_docs:
                show_error("Please upload at least one document")
                return
            
            # Show loading state
            query_button.disabled = True
            query_button.text = "Processing..."
            response_area.value = "Processing your query..."
            page_update()
            
            try:
                # Process query through app's query manager
                if hasattr(app, 'query'):
                    # Add documents if not already added
                    for doc in uploaded_docs:
                        if not any(d.get('path') == doc['path'] for d in getattr(app, '_indexed_docs', [])):
                            result = app.query.add_document(doc['path'], doc['name'])
                            if not result.get('success'):
                                show_error(f"Failed to index {doc['name']}: {result.get('message')}")
                                return
                    
                    # Query the documents
                    result = app.query.query(query_text, top_k=5)
                    
                    if result.get('success'):
                        # Format response
                        answers = result.get('answers', [])
                        if answers:
                            response_text = f"Question: {query_text}\n\n"
                            response_text += f"Found Answers: {len(answers)}\n"
                            response_text += f"Time: {result.get('time_seconds', 0):.3f}s\n"
                            response_text += "=" * 50 + "\n\n"
                            
                            for i, answer in enumerate(answers, 1):
                                similarity = answer.get('similarity', 0)
                                # Only show answers with meaningful relevance
                                if similarity > 0.05:
                                    response_text += f"Answer {i}:\n"
                                    response_text += f"{answer.get('text', 'No text')}\n"
                                    response_text += f"Source: {answer.get('doc', 'Unknown')}\n"
                                    response_text += f"Relevance: {similarity*100:.1f}%\n"
                                    response_text += f"Position: {answer.get('source', 'Unknown')}\n\n"
                            
                            if not any(a.get('similarity', 0) > 0.05 for a in answers):
                                response_text += "No relevant answers found. Please revise your question.\n"
                        else:
                            response_text = f"No answers found for question: '{query_text}'\n\n"
                            response_text += "Tips:\n"
                            response_text += "- Check your uploaded documents\n"
                            response_text += "- Rephrase your question with more details\n"
                            response_text += "- Try using different words\n"
                        
                        response_area.value = response_text
                    else:
                        show_error(result.get('message', 'Query failed'))
                else:
                    show_error("Query manager not available")
            
            except Exception as ex:
                show_error(f"Error processing query: {str(ex)}")
            
            finally:
                # Reset button state
                query_button.disabled = False
                query_button.text = "Ask Question"
                page_update()
        
        def show_error(message):
            """Show error message."""
            response_area.value = f"Error: {message}"
            page_update()
        
        def page_update():
            """Update the page."""
            if hasattr(app, 'page'):
                app.page.update()
        
        # File picker - use passed one or create new one
        if file_picker is None:
            file_picker = ft.FilePicker(on_result=handle_file_pick)
            # Only add to overlay if we created it here
            if hasattr(app, 'page') and file_picker not in app.page.overlay:
                app.page.overlay.append(file_picker)
                app.page.update()
        else:
            # Update the callback for the passed file_picker
            file_picker.on_result = handle_file_pick
        
        # Document list
        docs_list = ft.Column(
            controls=[],
            spacing=10
        )
        
        # Query input
        query_input = ft.TextField(
            label="Ask your question",
            hint_text="What would you like to know?",
            multiline=True,
            min_lines=3,
            max_lines=5,
            expand=True
        )
        
        # Response area - using Markdown for proper text display
        response_area = ft.Markdown(
            value="Your response will appear here...",
            selectable=True,
            expand=True
        )
        
        # Query button
        query_button = ft.ElevatedButton(
            text="Ask Question",
            on_click=handle_query,
            width=200
        )
        
        # Upload button
        upload_button = ft.ElevatedButton(
            text="Upload Document",
            icon=ft.Icons.UPLOAD,
            on_click=lambda e: file_picker.pick_files(allowed_extensions=['txt', 'pdf', 'docx', 'json'])
        )
        
        # Main layout
        return ft.Column(
            controls=[
                # Header
                ft.Row(
                    controls=[
                        ft.Text("Document Q&A", size=20, weight="bold"),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE,
                            tooltip="Close",
                            on_click=on_close,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN
                ),
                
                ft.Divider(),
                
                # Documents section
                ft.Text("Uploaded Documents:", weight="bold"),
                ft.Container(
                    content=docs_list,
                    border=ft.border.all(1, ft.Colors.OUTLINE),
                    border_radius=5,
                    padding=10,
                    height=150,
                    expand=True
                ),
                upload_button,
                
                ft.Divider(),
                
                # Query section
                ft.Text("Your Question:", weight="bold"),
                query_input,
                query_button,
                
                ft.Divider(),
                
                # Response section
                ft.Text("Response:", weight="bold"),
                response_area,
            ],
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            expand=True
        )
