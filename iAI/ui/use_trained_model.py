"""Page to use trained models."""

import flet as ft
import threading
from pathlib import Path
import requests


class UseTrainedModel:
    """Use trained model page."""
    
    @staticmethod
    def create_page(app, on_close=None):
        """Main page for using trained model."""
        
        lora_trainer = app.lora_trainer
        
        # Colors
        is_light = app.theme == "light"
        text_primary = ft.Colors.BLACK if is_light else ft.Colors.WHITE
        text_secondary = ft.Colors.GREY_700 if is_light else ft.Colors.GREY_300
        bg_primary = ft.Colors.WHITE if is_light else ft.Colors.SURFACE
        bg_hover = ft.Colors.GREY_100 if is_light else ft.Colors.GREY_900
        
        # State
        state = {
            'model_ready': False,
            'model_info': None,
            'response': '',
            'waiting': False
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
                ft.Text("AI Model with Training Data", size=24, weight="bold", color=text_primary),
                ft.Container(expand=True),
                close_btn
            ]),
            padding=20,
            bgcolor=bg_hover
        )
        
        # ==================== CHECK MODEL ====================
        
        model_status = ft.Text("Checking model...", size=12, color=text_secondary)
        model_details = ft.Text("", size=11, color=text_secondary)
        
        def check_model_status(e=None):
            """Check trained model status."""
            model_status.value = "Checking..."
            app.page.update()
            
            def check_thread():
                try:
                    # Check if LoRA adapter exists
                    adapter_dir = Path("data/fine_tune/models/lora_adapter")
                    
                    if adapter_dir.exists() and list(adapter_dir.glob("*")):
                        files = list(adapter_dir.glob("*"))
                        state['model_ready'] = True
                        state['model_info'] = {
                            'type': 'LoRA Adapter',
                            'base_model': 'phi3:mini (Ollama)',
                            'files': len(files),
                            'path': str(adapter_dir)
                        }
                        
                        model_status.value = "Ready: Model trained with your documents"
                        model_status.color = ft.Colors.GREEN
                        
                        model_details.value = (
                            f"Base: phi3:mini\n"
                            f"Type: LoRA Fine-tuned\n"
                            f"Location: {adapter_dir}\n"
                            f"Files: {len(files)}\n\n"
                            f"This model has learned from your documents!"
                        )
                    else:
                        state['model_ready'] = False
                        model_status.value = "Not trained yet"
                        model_status.color = ft.Colors.RED
                        model_details.value = "Go to 'Train Personal AI' first."
                
                except Exception as err:
                    model_status.value = f"Error: {str(err)[:50]}"
                    model_status.color = ft.Colors.RED
                
                app.page.update()
            
            threading.Thread(target=check_thread, daemon=True).start()
        
        # Initial check
        check_model_status()
        
        status_section = ft.Container(
            content=ft.Column([
                ft.Text("Model Status", size=14, weight="bold", color=text_primary),
                model_status,
                model_details,
                ft.ElevatedButton(
                    text="Refresh",
                    on_click=check_model_status,
                    expand=True
                )
            ], spacing=10),
            padding=15,
            bgcolor=bg_primary,
            border_radius=8
        )
        
        # ==================== USE MODEL ====================
        
        query_input = ft.TextField(
            label="Ask your model about the documents",
            multiline=True,
            min_lines=3,
            max_lines=6,
            filled=True,
            expand=True
        )
        
        response_display = ft.Text(
            "Response will appear here...",
            size=12,
            color=text_secondary,
            selectable=True
        )
        
        response_scroll = ft.Container(
            content=response_display,
            padding=15,
            bgcolor=bg_hover,
            border_radius=8,
            height=200,
            expand=True
        )
        
        waiting_indicator = ft.ProgressRing(visible=False)
        
        def send_query(e):
            """Send query to trained model."""
            question = query_input.value.strip()
            
            if not question:
                response_display.value = "Error: Please enter a question"
                response_display.color = ft.Colors.RED
                app.page.update()
                return
            
            if not state['model_ready']:
                response_display.value = "Error: Model not trained. Train first in AI Training section."
                response_display.color = ft.Colors.RED
                app.page.update()
                return
            
            state['waiting'] = True
            waiting_indicator.visible = True
            response_display.value = "Thinking..."
            response_display.color = ft.Colors.ORANGE
            send_btn.disabled = True
            app.page.update()
            
            def query_thread():
                try:
                    # Get training data context
                    training_data_file = Path("data/fine_tune/training_data_from_docs.json")
                    context = ""
                    
                    if training_data_file.exists():
                        import json
                        with open(training_data_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            context_parts = [item.get('output', '') for item in data[:10]]
                            context = "\n".join(context_parts)
                    
                    # Call phi3:mini with context about training data
                    try:
                        prompt = f"""You have been trained with this information:
{context[:1000]}

Question: {question}

Answer based on what you learned:"""
                        
                        response = requests.post(
                            'http://localhost:11434/api/generate',
                            json={
                                'model': 'phi3:mini',
                                'prompt': prompt,
                                'stream': False,
                                'temperature': 0.7
                            },
                            timeout=None
                        )
                        
                        if response.status_code == 200:
                            answer = response.json().get('response', 'No response')
                            response_display.value = f"AI Response:\n\n{answer}"
                            response_display.color = ft.Colors.GREEN
                        else:
                            response_display.value = f"Ollama error: {response.status_code}"
                            response_display.color = ft.Colors.RED
                    
                    except requests.exceptions.Timeout:
                        response_display.value = (
                            "Timeout: Model is taking too long.\n\n"
                            "Make sure Ollama is running:\n"
                            "ollama serve"
                        )
                        response_display.color = ft.Colors.RED
                    except requests.exceptions.ConnectionError:
                        response_display.value = (
                            "Connection Error: Ollama not accessible.\n\n"
                            "Start Ollama:\n"
                            "ollama serve"
                        )
                        response_display.color = ft.Colors.RED
                    except Exception as err:
                        response_display.value = f"Error: {str(err)[:100]}"
                        response_display.color = ft.Colors.RED
                
                except Exception as err:
                    response_display.value = f"Error: {str(err)[:150]}"
                    response_display.color = ft.Colors.RED
                finally:
                    state['waiting'] = False
                    waiting_indicator.visible = False
                    send_btn.disabled = False
                    app.page.update()
            
            threading.Thread(target=query_thread, daemon=True).start()
        
        send_btn = ft.ElevatedButton(
            text="Ask Model",
            on_click=send_query,
            expand=True,
            height=45,
            style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE)
        )
        
        use_section = ft.Container(
            content=ft.Column([
                ft.Text("Ask Your Trained Model", size=14, weight="bold", color=text_primary),
                query_input,
                ft.Row([waiting_indicator, send_btn], spacing=10),
                ft.Text("Response:", size=11, weight="bold", color=text_primary),
                response_scroll
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
                    status_section,
                    use_section
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
        
        return page
