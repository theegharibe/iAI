"""Asset manager for icons and backgrounds."""

import base64
from pathlib import Path
from typing import Optional

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


class AssetManager:
    """Manages loading of icons and backgrounds with fallbacks."""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent
        self.assets_path = self.base_path / "assets"
        self.icons_path = self.assets_path / "icons"
        self.bg_path = self.assets_path / "backgrounds"
        
        # Cache for loaded assets
        self.icon_cache = {}
        self.bg_cache = {}
        
        # Minimize debug output on init (it slows things down)
        print(f"[AssetManager] OK - Initialized (icons: {self.icons_path.exists()}, bg: {self.bg_path.exists()})")
    
    def get_icon(self, name: str, size: int = 24) -> Optional[str]:
        """
        Get icon as base64 string.
        Supports theme-aware icons (e.g., "question.light" for light theme).
        
        Args:
            name: Icon name (without extension, e.g., "question" or "question.light")
            size: Desired size in pixels (default 24)
            
        Returns:
            Base64 encoded PNG or None if not found
        """
        cache_key = f"{name}_{size}"
        
        if cache_key in self.icon_cache:
            return self.icon_cache[cache_key]
        
        icon_path = self.icons_path / f"{name}.png"
        
        if not icon_path.exists():
            # Log warning but don't print externally (maintain anonymity)
            if HAS_PIL:
                # Silently fail - no external logging
                pass
            return None
        
        if not HAS_PIL:
            return None
        
        try:
            with Image.open(icon_path) as img:
                # Resize with high quality (LANCZOS for best results)
                img = img.resize((size, size), Image.Resampling.LANCZOS)
                
                # Convert to base64 PNG
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                b64 = base64.b64encode(buffer.getvalue()).decode()
                
                self.icon_cache[cache_key] = b64
                return b64
        
        except Exception as e:
            # Anonymous error handling - no external calls
            return None
    
    def icon_exists(self, name: str) -> bool:
        """
        Check if an icon exists without loading it.
        Useful for conditional fallbacks.
        
        Args:
            name: Icon name without extension
            
        Returns:
            True if icon exists, False otherwise
        """
        icon_path = self.icons_path / f"{name}.png"
        return icon_path.exists()
    
    def get_background(self, theme: str, target_width: int = 3840, target_height: int = 2160) -> Optional[str]:
        """
        Get background image as base64 string.
        
        Args:
            theme: 'light' or 'dark'
            target_width: Target width for resizing (default 4K resolution)
            target_height: Target height for resizing (default 4K resolution)
            
        Returns:
            Base64 encoded PNG or None if not found
        """
        cache_key = f"{theme}_{target_width}x{target_height}"
        
        if cache_key in self.bg_cache:
            print(f"[AssetManager] Using cached {theme} background")
            return self.bg_cache[cache_key]
        
        bg_path = self.bg_path / f"{theme}.png"
        
        if not bg_path.exists():
            return None
            
        if not HAS_PIL:
            return None
        
        try:
            with Image.open(bg_path) as img:
                # Calculate new dimensions maintaining aspect ratio
                img_width, img_height = img.size
                ratio = min(target_width/img_width, target_height/img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                # Resize maintaining aspect ratio
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                # Convert to RGB if necessary (remove alpha channel)
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (300, 300, 300))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                    img = background
                
                # Save to buffer
                buffer = io.BytesIO()
                img.save(buffer, format='PNG', optimize=True)
                b64 = base64.b64encode(buffer.getvalue()).decode()
                
                self.bg_cache[cache_key] = b64
                print(f"[AssetManager] [OK] Loaded {theme} background")
                return b64
        
        except Exception as e:
            print(f"[AssetManager] [ERROR] Error loading {theme} background: {e}")
            return None