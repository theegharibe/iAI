"""Training components for model fine-tuning and book training."""

import flet as ft
from .book_training_ui import BookTrainingUI


class TrainingUI:
    """Training interface combining personal AI training and book training."""
    
    @staticmethod
    def create_training_page(app, on_close):
        """Create the main training page with tabs for different training methods.
        
        Args:
            app: The main LocalAIChatApp instance
            on_close: Callback function when training page is closed
        
        Returns:
            ft.Column: The training page UI
        """
        
        # Create tabs for different training methods
        training_tabs = ft.Tabs(
            selected_index=0,
            tabs=[
                ft.Tab(
                    text="📚 Book Training",
                    content=BookTrainingUI.create_book_training_section(app),
                ),
            ],
            expand=True,
        )
        
        # Main container
        return ft.Column(
            controls=[
                # Header
                ft.Container(
                    content=ft.Row(
                        controls=[
                            ft.Text(
                                "🎓 AI Training Center",
                                size=24,
                                weight="bold",
                            ),
                            ft.IconButton(
                                icon=ft.Icons.CLOSE,
                                tooltip="Close",
                                on_click=on_close,
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    ),
                    padding=20,
                    border_radius=8,
                ),
                ft.Divider(),
                
                # Training tabs
                training_tabs,
                
                # Footer with info
                ft.Container(
                    content=ft.Text(
                        "💡 Tip: Train with books to enhance your AI's knowledge with specific content",
                        size=12,
                        color=ft.Colors.GREY_700,
                    ),
                    padding=15,
                ),
            ],
            expand=True,
            spacing=10,
            padding=0,
        )
