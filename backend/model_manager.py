"""
model_manager.py
Manages model references (not actual files)
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict


@dataclass
class ModelReference:
    """Reference to a model file on disk"""
    name: str           # Display name (e.g., "TinyLlama 1.1B")
    path: str           # Full path to .gguf file
    size_mb: float      # File size in MB
    date_added: str     # When added to manager
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


def get_config_dir():
    """Get the config directory, handling bundled app"""
    if getattr(sys, 'frozen', False):
        # Running in bundled app - use Application Support
        if sys.platform == 'darwin':
            config_dir = Path.home() / 'Library' / 'Application Support' / 'MeMyselfAI'
        else:
            config_dir = Path.home() / '.memyselfai'
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
    else:
        # Running in development - use current directory
        return Path('.')


class ModelManager:
    """Manages model references stored in JSON"""
    
    def __init__(self, config, config_file: str = "models.json"):
        config_dir = get_config_dir()
        self.config_file = config_dir / config_file
        self.models: List[ModelReference] = []
        self.config = config
        self.load()
    
    def load(self):
        """Load model references from JSON"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.models = [ModelReference.from_dict(m) for m in data]
                print(f"✅ [ModelManager] Loaded {len(self.models)} model(s)")
            except Exception as e:
                print(f"⚠️  [ModelManager] Failed to load: {e}")
                self.models = []
        else:
            print(f"ℹ️  [ModelManager] No models.json found, starting fresh")
            self.models = []
    
    def save(self):
        """Save model references to JSON"""
        try:
            data = [m.to_dict() for m in self.models]
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ [ModelManager] Saved {len(self.models)} model(s)")
        except Exception as e:
            print(f"❌ [ModelManager] Failed to save: {e}")

    def sync_models_from_folder(self, folder_path: str):
        """Sync models.json with a folder."""
        model_folder = Path(folder_path)
        if not model_folder.is_dir():
            return

        # Find all .gguf files in the folder
        found_files = {str(p.absolute()) for p in model_folder.glob("*.gguf")}
        
        # Models currently in models.json
        known_files = {m.path for m in self.models}
        
        # Add new models
        for file_path in found_files - known_files:
            self.add_model(file_path)
            
        # Remove old models
        for file_path in known_files - found_files:
            self.remove_model(file_path)

    def add_model(self, path: str, custom_name: Optional[str] = None) -> bool:
        """
        Add a model reference
        
        Args:
            path: Full path to .gguf file
            custom_name: Optional custom display name
            
        Returns:
            True if added successfully
        """
        file_path = Path(path)
        
        # Validate file exists and is .gguf
        if not file_path.exists():
            print(f"❌ [ModelManager] File not found: {path}")
            return False
        
        if file_path.suffix != '.gguf':
            print(f"❌ [ModelManager] Not a .gguf file: {path}")
            return False
        
        # Check if already exists
        if any(m.path == str(file_path) for m in self.models):
            print(f"⚠️  [ModelManager] Model already exists: {path}")
            return False
        
        # Get file size
        size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Create reference
        from datetime import datetime
        model_ref = ModelReference(
            name=custom_name or file_path.stem,
            path=str(file_path.absolute()),
            size_mb=round(size_mb, 1),
            date_added=datetime.now().isoformat()
        )
        
        self.models.append(model_ref)
        self.save()
        
        print(f"✅ [ModelManager] Added: {model_ref.name}")
        return True
    
    def remove_model(self, path: str) -> bool:
        """
        Remove a model reference (does NOT delete file from disk)
        
        Args:
            path: Path of model to remove
            
        Returns:
            True if removed successfully
        """
        original_count = len(self.models)
        self.models = [m for m in self.models if m.path != path]
        
        if len(self.models) < original_count:
            self.save()
            print(f"✅ [ModelManager] Removed reference (file still on disk)")
            return True
        else:
            print(f"⚠️  [ModelManager] Model not found: {path}")
            return False
    
    def get_all_models(self) -> List[ModelReference]:
        """Get all model references"""
        # Sync with the models folder first
        models_folder = self.config.get("models_folder", "Models")
        self.sync_models_from_folder(models_folder)

        # Filter out models whose files no longer exist
        valid_models = []
        for model in self.models:
            if Path(model.path).exists():
                valid_models.append(model)
            else:
                print(f"⚠️  [ModelManager] Skipping missing file: {model.path}")
        
        return valid_models
    
    def get_model_by_path(self, path: str) -> Optional[ModelReference]:
        """Get model reference by path"""
        for model in self.models:
            if model.path == path:
                return model
        return None
    
    def rename_model(self, path: str, new_name: str) -> bool:
        """
        Rename a model (display name only, not file)
        
        Args:
            path: Path of model to rename
            new_name: New display name
            
        Returns:
            True if renamed successfully
        """
        model = self.get_model_by_path(path)
        if model:
            model.name = new_name
            self.save()
            print(f"✅ [ModelManager] Renamed to: {new_name}")
            return True
        return False


if __name__ == "__main__":
    # Test the model manager
    manager = ModelManager("test_models.json")
    
    print("\nTesting ModelManager...")
    print("=" * 60)
    
    # Add a model (use a real path for testing)
    test_model = "/path/to/test.gguf"
    print(f"\n1. Adding model: {test_model}")
    # manager.add_model(test_model, "Test Model")
    
    # List models
    print(f"\n2. All models:")
    for model in manager.get_all_models():
        print(f"   - {model.name} ({model.size_mb}MB)")
        print(f"     Path: {model.path}")
    
    # Remove a model
    # print(f"\n3. Removing model: {test_model}")
    # manager.remove_model(test_model)
    
    print("\n" + "=" * 60)
    
    # Cleanup test file
    Path("test_models.json").unlink(missing_ok=True)
