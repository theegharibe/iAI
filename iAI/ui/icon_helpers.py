"""
Icon Helper Utilities for Theme-Aware PNG Icons

Provides reusable functions for creating buttons and icons with
custom PNG variants and Flet fallbacks.

Features:
- Theme-aware icon loading
- Automatic light/dark variant selection
- Graceful fallback to Material Icons
- Anonymous error handling
"""

import flet as ft
from typing import Optional, Callable


class IconHelpers:
    """Helper utilities for icon management."""
    
    @staticmethod
    def get_theme_suffix(theme: str) -> str:
        """
        Get icon filename suffix based on theme.
        
        Args:
            theme: 'light' or 'dark' (any other value defaults to dark)
            
        Returns:
            ".light" for light theme, "" for dark theme
        """
        return ".light" if theme == "light" else ""
    
    @staticmethod
    def create_icon_button(
        app,
        icon_name: str,
        tooltip: str,
        on_click: Callable,
        fallback_icon: str = None,
        size: int = 32,
        icon_color: str = None
    ) -> Optional[ft.IconButton]:
        """
        Create an icon button with custom PNG or Flet fallback.
        
        This is the main pattern used throughout the app for sidebar
        and toolbar icons.
        
        Args:
            app: Application instance (has assets and theme)
            icon_name: Name of icon without extension (e.g., "question")
            tooltip: Tooltip text for the button
            on_click: Click handler callback
            fallback_icon: Flet icon to use if PNG not found
            size: Icon size in pixels (default 32)
            icon_color: Icon color (optional)
            
        Returns:
            ft.IconButton with custom PNG or fallback
            
        Example:
            >>> btn = IconHelpers.create_icon_button(
            ...     app, "question", "Ask Questions",
            ...     app.show_query_page, ft.Icons.QUESTION_ANSWER
            ... )
        """
        theme_suffix = IconHelpers.get_theme_suffix(app.theme)
        icon_data = app.assets.get_icon(f"{icon_name}{theme_suffix}", size)
        
        if icon_data:
            # Use custom PNG icon
            return ft.IconButton(
                content=ft.Image(
                    src_base64=icon_data,
                    width=size,
                    height=size,
                    fit=ft.ImageFit.CONTAIN
                ),
                tooltip=tooltip,
                on_click=on_click,
                icon_color=icon_color
            )
        elif fallback_icon:
            # Fall back to Flet Material Icon
            return ft.IconButton(
                icon=fallback_icon,
                icon_size=size,
                tooltip=tooltip,
                on_click=on_click,
                icon_color=icon_color
            )
        else:
            return None
    
    @staticmethod
    def create_button_with_icon(
        app,
        icon_name: str,
        button_text: str,
        on_click: Callable,
        fallback_icon: str = None,
        size: int = 24,
        expand: bool = True,
        style: ft.ButtonStyle = None
    ) -> ft.ElevatedButton:
        """
        Create a button with custom PNG icon or fallback.
        
        Used for action buttons (Export, Clear All, From KB, etc.)
        
        Args:
            app: Application instance
            icon_name: Icon filename without extension
            button_text: Text displayed on button
            on_click: Click handler
            fallback_icon: Flet icon fallback
            size: Icon size in pixels
            expand: Whether button should expand to fill space
            style: Optional button style
            
        Returns:
            ft.ElevatedButton with icon
            
        Example:
            >>> export_btn = IconHelpers.create_button_with_icon(
            ...     app, "export", "Export Data",
            ...     export_handler, ft.Icons.DOWNLOAD
            ... )
        """
        theme_suffix = IconHelpers.get_theme_suffix(app.theme)
        icon_data = app.assets.get_icon(f"{icon_name}{theme_suffix}", size)
        
        if icon_data:
            # Use custom PNG icon
            return ft.ElevatedButton(
                text=button_text,
                icon_content=ft.Image(
                    src_base64=icon_data,
                    width=size,
                    height=size,
                    fit=ft.ImageFit.CONTAIN
                ),
                on_click=on_click,
                expand=expand,
                style=style
            )
        else:
            # Use Flet icon
            return ft.ElevatedButton(
                text=button_text,
                icon=fallback_icon,
                on_click=on_click,
                expand=expand,
                style=style
            )
    
    @staticmethod
    def create_close_button(
        app,
        on_click: Callable,
        size: int = 24
    ) -> ft.IconButton:
        """
        Create a close button with custom icon.
        
        Convenience method for close buttons in modal pages.
        
        Args:
            app: Application instance
            on_click: Click handler (usually page close)
            size: Icon size (default 24)
            
        Returns:
            ft.IconButton with close icon
            
        Example:
            >>> close_btn = IconHelpers.create_close_button(
            ...     app, on_close_handler
            ... )
        """
        theme_suffix = IconHelpers.get_theme_suffix(app.theme)
        close_icon = app.assets.get_icon(f"close{theme_suffix}", size)
        
        if close_icon:
            return ft.IconButton(
                content=ft.Image(
                    src_base64=close_icon,
                    width=size,
                    height=size,
                    fit=ft.ImageFit.CONTAIN
                ),
                on_click=on_click,
                tooltip="Close"
            )
        else:
            return ft.IconButton(
                icon=ft.Icons.CLOSE,
                icon_size=size,
                on_click=on_click,
                tooltip="Close"
            )
    
    @staticmethod
    def icon_exists(app, icon_name: str) -> bool:
        """
        Check if custom icon exists.
        
        Args:
            app: Application instance
            icon_name: Icon name without extension
            
        Returns:
            True if icon exists for current theme
        """
        theme_suffix = IconHelpers.get_theme_suffix(app.theme)
        return app.assets.icon_exists(f"{icon_name}{theme_suffix}")
    
    @staticmethod
    def batch_load_icons(app, icon_names: list, size: int = 32) -> dict:
        """
        Load multiple icons at once for better performance.
        
        Useful during page initialization to pre-cache icons.
        
        Args:
            app: Application instance
            icon_names: List of icon names without extension
            size: Icon size in pixels
            
        Returns:
            Dictionary mapping icon_name to base64 data or None
            
        Example:
            >>> icons = IconHelpers.batch_load_icons(
            ...     app, ["question", "ai_train", "rocket"]
            ... )
            >>> print(icons["question"])  # base64 string or None
        """
        theme_suffix = IconHelpers.get_theme_suffix(app.theme)
        result = {}
        
        for icon_name in icon_names:
            full_name = f"{icon_name}{theme_suffix}"
            result[icon_name] = app.assets.get_icon(full_name, size)
        
        return result
    
    @staticmethod
    def create_icon_grid(
        app,
        icon_configs: list,
        on_click_map: dict = None
    ) -> ft.Container:
        """
        Create a grid of icon buttons.
        
        Useful for icon galleries or dashboards.
        
        Args:
            app: Application instance
            icon_configs: List of dicts with keys:
                - icon_name: str (required)
                - tooltip: str (required)
                - size: int (optional, default 32)
                - fallback: str (optional fallback icon)
            on_click_map: Dict mapping icon_name to click handler
            
        Returns:
            ft.Container with icon grid
            
        Example:
            >>> configs = [
            ...     {"icon_name": "question", "tooltip": "Ask"},
            ...     {"icon_name": "ai_train", "tooltip": "Train"},
            ... ]
            >>> handlers = {
            ...     "question": app.show_query_page,
            ...     "ai_train": app.show_personal_ai_page
            ... }
            >>> grid = IconHelpers.create_icon_grid(app, configs, handlers)
        """
        on_click_map = on_click_map or {}
        buttons = []
        
        for config in icon_configs:
            icon_name = config.get("icon_name")
            tooltip = config.get("tooltip", "")
            size = config.get("size", 32)
            fallback = config.get("fallback")
            
            on_click = on_click_map.get(icon_name)
            
            btn = IconHelpers.create_icon_button(
                app, icon_name, tooltip, on_click,
                fallback, size
            )
            
            if btn:
                buttons.append(btn)
        
        return ft.Container(
            content=ft.Row(
                controls=buttons,
                wrap=True,
                spacing=10,
                alignment=ft.MainAxisAlignment.CENTER
            ),
            padding=10
        )


# Convenience exports
def get_theme_suffix(app) -> str:
    """Get icon suffix for current theme."""
    return IconHelpers.get_theme_suffix(app.theme)


def create_icon_btn(app, icon_name, tooltip, on_click, 
                    fallback_icon=None, size=32):
    """Shorthand for create_icon_button."""
    return IconHelpers.create_icon_button(
        app, icon_name, tooltip, on_click, fallback_icon, size
    )


def create_btn_with_icon(app, icon_name, text, on_click,
                         fallback_icon=None, size=24, expand=True):
    """Shorthand for create_button_with_icon."""
    return IconHelpers.create_button_with_icon(
        app, icon_name, text, on_click, fallback_icon, size, expand
    )


def create_close_btn(app, on_click, size=24):
    """Shorthand for create_close_button."""
    return IconHelpers.create_close_button(app, on_click, size)
