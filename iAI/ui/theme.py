"""Theme management for UI."""

import flet as ft


class ThemeManager:
    """Manages application themes and colors."""
    
    @staticmethod
    def apply_theme(page: ft.Page, theme: str, asset_manager):
        """Apply theme to page with background."""
        # Set theme mode ONLY (fast operation)
        page.theme_mode = ft.ThemeMode.DARK if theme == 'dark' else ft.ThemeMode.LIGHT
        page.window_maximized = True
        page.window_maximizable = True
        page.window_resizable = True
        
        # [OK] Set fallback background color immediately (no wait)
        page.bgcolor = ft.Colors.GREY_900 if theme == 'dark' else ft.Colors.GREY_100
        
        # Try to load background (this may take time)
        bg_data = asset_manager.get_background(
            theme, 
            target_width=int(page.window_width or 1920),
            target_height=int(page.window_height or 1080)
        )
        
        if bg_data:
            print(f"[Theme] Setting background image for {theme} theme")
            # Set background image
            page.bgcolor = None  # Remove solid color
            page.window_bgcolor = ft.Colors.TRANSPARENT
            
            # Create background image overlay
            bg_image = ft.Container(
                expand=True,
                content=ft.Stack([
                    ft.Image(
                        src_base64=bg_data,
                        width=4000,  # Oversized to ensure coverage
                        height=3000, # Oversized to ensure coverage
                        fit=ft.ImageFit.COVER,
                        repeat=ft.ImageRepeat.NO_REPEAT,
                        opacity=0.15 if theme == 'dark' else 0.1,
                    ),
                ]),
                clip_behavior=ft.ClipBehavior.ANTI_ALIAS,
            )
            return bg_image
        else:
            print(f"[Theme] Using solid color for {theme} theme")
            # Already set bgcolor above
            return None
    
    @staticmethod
    def get_message_colors(role: str, theme: str) -> tuple:
        """
        Get colors for message bubbles.
        
        Returns:
            (bubble_color, text_color, alignment)
        """
        if role == "user":
            color = ft.Colors.BLACK if theme == 'dark' else ft.Colors.BLACK
            text_color = ft.Colors.WHITE
            alignment = ft.MainAxisAlignment.END
        elif role == "assistant":
            color = ft.Colors.WHITE if theme == 'dark' else ft.Colors.WHITE
            text_color = ft.Colors.BLACK
            alignment = ft.MainAxisAlignment.START
        elif role == "system":
            color = ft.Colors.GREY_700 if theme == 'dark' else ft.Colors.GREY_400
            text_color = ft.Colors.WHITE if theme == 'dark' else ft.Colors.BLACK
            alignment = ft.MainAxisAlignment.CENTER
        else:  # error
            color = ft.Colors.RED_600
            text_color = ft.Colors.WHITE
            alignment = ft.MainAxisAlignment.START
        
        return color, text_color, alignment