"""
model_manager_dialog.py
UI for managing model references
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QListWidget, QListWidgetItem, QFileDialog, QMessageBox,
    QInputDialog, QLabel
)
from PyQt6.QtCore import Qt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.model_manager import ModelManager, ModelReference


class ModelManagerDialog(QDialog):
    """Dialog for managing model references"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = ModelManager(parent.config)
        self.init_ui()
        self.load_models()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Model Manager")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        layout = QVBoxLayout(self)
        
        # Info label
        info_label = QLabel(
            "Manage model references (files remain on disk when removed)"
        )
        info_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(info_label)
        
        # Model list
        self.model_list = QListWidget()
        self.model_list.setStyleSheet("""
            QListWidget {
                background: #1C1C1E;
                border: 1px solid #3A3A3C;
                border-radius: 6px;
                color: #EBEBF5;
            }
            QListWidget::item {
                padding: 8px 12px;
                border-bottom: 1px solid #2C2C2E;
            }
            QListWidget::item:selected {
                background: #3A3A3C;
                color: #FFFFFF;
            }
        """)
        self.model_list.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.model_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        add_btn = QPushButton("➕ Add Model...")
        add_btn.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #EBEBF5;
                border: 1px solid #3A3A3C; border-radius: 6px;
                padding: 6px 12px; font-size: 13px;
            }
            QPushButton:hover { 
                background: #3A3A3C; 
                border-color: #e009a7; 
            }
        """)
        add_btn.clicked.connect(self.add_model)
        button_layout.addWidget(add_btn)
        
        self.rename_btn = QPushButton("✏️ Rename")
        self.rename_btn.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #EBEBF5;
                border: 1px solid #3A3A3C; border-radius: 6px;
                padding: 6px 12px; font-size: 13px;
            }
            QPushButton:hover { 
                background: #3A3A3C; 
                border-color: #e009a7; 
            }
            QPushButton:disabled {
                color: #6C6C6E;
            }
        """)
        self.rename_btn.clicked.connect(self.rename_model)
        self.rename_btn.setEnabled(False)
        button_layout.addWidget(self.rename_btn)
        
        self.remove_btn = QPushButton("🗑️ Remove Reference")
        self.remove_btn.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #FF453A;
                border: 1px solid #3A3A3C; border-radius: 6px;
                padding: 6px 12px; font-size: 13px;
            }
            QPushButton:hover { 
                background: #3A3A3C; 
                border-color: #e009a7; 
            }
            QPushButton:disabled {
                color: #6C6C6E;
            }
        """)
        self.remove_btn.clicked.connect(self.remove_model)
        self.remove_btn.setEnabled(False)
        button_layout.addWidget(self.remove_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #EBEBF5;
                border: 1px solid #3A3A3C; border-radius: 6px;
                padding: 6px 12px; font-size: 13px;
            }
            QPushButton:hover { 
                background: #3A3A3C; 
                border-color: #e009a7; 
            }
        """)
        close_btn.clicked.connect(self.accept)
        button_layout.addWidget(close_btn)

        
        layout.addLayout(button_layout)
    
    def load_models(self):
        """Load models into list"""
        self.model_list.clear()
        
        models = self.manager.get_all_models()
        
        if not models:
            item = QListWidgetItem("No models added yet. Click 'Add Model' to get started.")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self.model_list.addItem(item)
            return
        
        for model in models:
            # Format display text
            text = f"{model.name}\n"
            text += f"   Size: {model.size_mb} MB\n"
            text += f"   Path: {model.path}"
            
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, model.path)
            self.model_list.addItem(item)
    
    def on_selection_changed(self):
        """Handle selection change"""
        has_selection = len(self.model_list.selectedItems()) > 0
        self.rename_btn.setEnabled(has_selection)
        self.remove_btn.setEnabled(has_selection)
    
    def add_model(self):
        """Add a new model reference"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "GGUF Models (*.gguf);;All Files (*)"
        )
        
        if not file_path:
            return
        
        # Ask for custom name
        default_name = Path(file_path).stem
        name, ok = QInputDialog.getText(
            self,
            "Model Name",
            "Enter a display name for this model:",
            text=default_name
        )
        
        if not ok or not name.strip():
            name = default_name
        
        # Add to manager
        if self.manager.add_model(file_path, name.strip()):
            self.load_models()
            QMessageBox.information(
                self,
                "Success",
                f"Added model: {name}"
            )
        else:
            QMessageBox.warning(
                self,
                "Failed",
                "Failed to add model. Check console for details."
            )
    
    def rename_model(self):
        """Rename selected model"""
        items = self.model_list.selectedItems()
        if not items:
            return
        
        item = items[0]
        path = item.data(Qt.ItemDataRole.UserRole)
        
        model = self.manager.get_model_by_path(path)
        if not model:
            return
        
        # Ask for new name
        new_name, ok = QInputDialog.getText(
            self,
            "Rename Model",
            "Enter new name:",
            text=model.name
        )
        
        if ok and new_name.strip():
            if self.manager.rename_model(path, new_name.strip()):
                self.load_models()
                QMessageBox.information(
                    self,
                    "Success",
                    f"Renamed to: {new_name}"
                )
    
    def remove_model(self):
        """Remove selected model reference"""
        items = self.model_list.selectedItems()
        if not items:
            return
        
        item = items[0]
        path = item.data(Qt.ItemDataRole.UserRole)
        
        model = self.manager.get_model_by_path(path)
        if not model:
            return
        
        # Confirm removal
        reply = QMessageBox.question(
            self,
            "Remove Model Reference",
            f"Remove reference to '{model.name}'?\n\n"
            f"The file will NOT be deleted from disk:\n{path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.manager.remove_model(path):
                self.load_models()
                QMessageBox.information(
                    self,
                    "Success",
                    "Model reference removed (file still on disk)"
                )