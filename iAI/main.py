"""
iAI - Local AI Chat Assistant
Entry point for the application.
"""

import flet as ft
import os
from pathlib import Path

# Ensure we're in the correct working directory
script_dir = Path(__file__).parent
os.chdir(script_dir)

from ui import LocalAIChatApp


def main(page: ft.Page):
    """Main entry point."""
    LocalAIChatApp(page)


if __name__ == "__main__":
    print("Starting iAI - Local AI Chat...")
    print(f"Working Directory: {os.getcwd()}")
    print("Assets: assets/")
    print("   - icons/: PNG icons")
    print("   - backgrounds/: light.png, dark.png")
    print()
    
    try:
        ft.app(
            target=main,
            assets_dir="assets",
            upload_dir="assets/temp"
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()