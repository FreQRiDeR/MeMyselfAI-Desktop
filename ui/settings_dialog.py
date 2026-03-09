"""
settings_dialog.py
Settings configuration dialog
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLineEdit, QPushButton, QLabel, QSpinBox,
    QDoubleSpinBox, QFileDialog, QDialogButtonBox,
    QGroupBox, QMessageBox, QFontComboBox, QComboBox,
    QCheckBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from pathlib import Path
import shlex


class SettingsDialog(QDialog):
    """Settings configuration dialog"""
    
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Settings")
        self.setMinimumWidth(600)
        
        layout = QVBoxLayout(self)
        
        # Paths group
        paths_group = QGroupBox("Backend Configuration")
        paths_layout = QFormLayout()
        
        # Backend type selector
        self.backend_combo = QComboBox()
        self.backend_combo.setStyleSheet("""
            QComboBox {
                background: #2C2C2E; color: #EBEBF5;
                border: 1px solid #3A3A3C; border-radius: 6px;
                padding: 4px 8px; font-size: 12px;
            }
            QComboBox:hover {
                border-color: #e009a7;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
                border-left: 1px solid #3A3A3C;
            }
            QComboBox::drop-down:hover {
                background: #3A3A3C;
                border-left: 1px solid #e009a7;
            }
            QComboBox::down-arrow {
                width: 8px;
                height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e009a7;
            }
            QComboBox QAbstractItemView {
                background: #2C2C2E;
                color: #EBEBF5;
                border: 1px solid #3A3A3C;
                border-radius: 6px;
                selection-background-color: #3A3A3C;
            }
        """)
        self.backend_combo.addItem("Local (llama.cpp)", "local")
        self.backend_combo.addItem("Ollama (Local API)", "ollama")
        self.backend_combo.addItem("HuggingFace (Cloud)", "huggingface")
        self.backend_combo.currentIndexChanged.connect(self.on_backend_changed)
        paths_layout.addRow("Backend:", self.backend_combo)
        
        # llama.cpp path (for local backend)
        self.llama_label = QLabel("llama.cpp Binary:")
        llama_layout = QHBoxLayout()
        self.llama_path_input = QLineEdit()
        self.llama_path_input.setPlaceholderText("bundled (or /path/to/llama-server)")
        self.llama_path_input.setStyleSheet("""
            QLineEdit {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #e009a7;
                outline: none;
            }
        """)
        llama_layout.addWidget(self.llama_path_input)
        
        self.browse_llama_btn = QPushButton("Browse...")
        self.browse_llama_btn.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #EBEBF5;
                border: 1px solid #3A3A3C; border-radius: 6px;
                padding: 4px 8px; font-size: 12px;
            }
            QPushButton:hover { 
                background: #3A3A3C; 
                border-color: #e009a7; 
            }
        """)
        self.browse_llama_btn.clicked.connect(self.browse_llama_path)
        llama_layout.addWidget(self.browse_llama_btn)
        
        paths_layout.addRow(self.llama_label, llama_layout)

        # Ollama path (for ollama backend)
        self.ollama_binary_label = QLabel("Ollama Binary:")
        ollama_layout = QHBoxLayout()
        self.ollama_path_input = QLineEdit()
        self.ollama_path_input.setPlaceholderText("bundled (or /path/to/ollama)")
        self.ollama_path_input.setStyleSheet("""
            QLineEdit {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #e009a7;
                outline: none;
            }
        """)
        ollama_layout.addWidget(self.ollama_path_input)
        
        self.browse_ollama_btn = QPushButton("Browse...")
        self.browse_ollama_btn.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #EBEBF5;
                border: 1px solid #3A3A3C; border-radius: 6px;
                padding: 4px 8px; font-size: 12px;
            }
            QPushButton:hover { 
                background: #3A3A3C; 
                border-color: #e009a7; 
            }
        """)
        self.browse_ollama_btn.clicked.connect(self.browse_ollama_path)
        ollama_layout.addWidget(self.browse_ollama_btn)
        
        paths_layout.addRow(self.ollama_binary_label, ollama_layout)
        
        # Ollama URL (for ollama backend)
        self.ollama_label = QLabel("Ollama URL:")
        self.ollama_url_input = QLineEdit()
        self.ollama_url_input.setPlaceholderText("http://localhost:11434")
        self.ollama_url_input.setStyleSheet("""
            QLineEdit {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #e009a7;
                outline: none;
            }
        """)
        paths_layout.addRow(self.ollama_label, self.ollama_url_input)
        
        # HuggingFace API Key (for HF backend)
        self.hf_label = QLabel("HF API Key:")
        self.hf_api_key_input = QLineEdit()
        self.hf_api_key_input.setPlaceholderText("hf_...")
        self.hf_api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.hf_api_key_input.setStyleSheet("""
            QLineEdit {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #e009a7;
                outline: none;
            }
        """)
        paths_layout.addRow(self.hf_label, self.hf_api_key_input)
        
        # Add note about models
        models_note = QLabel("💡 Models are managed via File → Manage Models")
        models_note.setStyleSheet("color: #666; font-style: italic;")
        paths_layout.addRow("", models_note)
        
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # Generation parameters group
        params_group = QGroupBox("Generation Parameters")
        params_layout = QFormLayout()
        
        self.max_tokens_input = QSpinBox()
        self.max_tokens_input.setStyleSheet("""
            QSpinBox {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QSpinBox:focus {
                border-color: #e009a7;
            }
            QSpinBox::up-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QSpinBox::up-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QSpinBox::down-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QSpinBox::down-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QSpinBox::up-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #e009a7;
            }
            QSpinBox::down-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e009a7;
            }
        """)
        self.max_tokens_input.setRange(1, 4096)
        self.max_tokens_input.setSingleStep(64)
        params_layout.addRow("Max Tokens:", self.max_tokens_input)
        
        self.temperature_input = QDoubleSpinBox()
        self.temperature_input.setStyleSheet("""
            QDoubleSpinBox {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QDoubleSpinBox:focus {
                border-color: #e009a7;
            }
            QDoubleSpinBox::up-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QDoubleSpinBox::up-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QDoubleSpinBox::down-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QDoubleSpinBox::down-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QDoubleSpinBox::up-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #e009a7;
            }
            QDoubleSpinBox::down-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e009a7;
            }
        """)
        self.temperature_input.setRange(0.0, 2.0)
        self.temperature_input.setSingleStep(0.1)
        self.temperature_input.setDecimals(2)
        params_layout.addRow("Temperature:", self.temperature_input)
        
        self.context_size_input = QSpinBox()
        self.context_size_input.setStyleSheet("""
            QSpinBox {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QSpinBox:focus {
                border-color: #e009a7;
            }
            QSpinBox::up-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QSpinBox::up-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QSpinBox::down-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QSpinBox::down-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QSpinBox::up-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #e009a7;
            }
            QSpinBox::down-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e009a7;
            }
        """)
        self.context_size_input.setRange(512, 8192)
        self.context_size_input.setSingleStep(512)
        params_layout.addRow("Context Size:", self.context_size_input)
        
        self.threads_input = QSpinBox()
        self.threads_input.setStyleSheet("""
            QSpinBox {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QSpinBox:focus {
                border-color: #e009a7;
            }
            QSpinBox::up-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QSpinBox::up-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QSpinBox::down-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QSpinBox::down-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QSpinBox::up-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #e009a7;
            }
            QSpinBox::down-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e009a7;
            }
        """)
        self.threads_input.setRange(1, 16)
        params_layout.addRow("Threads:", self.threads_input)

        self.timeout_input = QSpinBox()
        self.timeout_input.setStyleSheet("""
            QSpinBox {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QSpinBox:focus {
                border-color: #e009a7;
            }
            QSpinBox::up-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QSpinBox::up-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QSpinBox::down-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QSpinBox::down-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QSpinBox::up-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #e009a7;
            }
            QSpinBox::down-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e009a7;
            }
        """)
        self.timeout_input.setRange(30, 3600)
        self.timeout_input.setSingleStep(30)
        self.timeout_input.setSuffix(" s")
        self.timeout_input.setToolTip("Max seconds to wait for a response from any backend (Ollama, llama-server, HuggingFace)")
        params_layout.addRow("Inference Timeout:", self.timeout_input)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Local llama.cpp tuning parameters
        self.local_tuning_group = QGroupBox("Local llama.cpp Tuning")
        tuning_layout = QFormLayout()

        self.llama_gpu_layers_input = QLineEdit()
        self.llama_gpu_layers_input.setPlaceholderText("auto, all, or integer (e.g. 35)")
        self.llama_gpu_layers_input.setToolTip("Sets -ngl/--gpu-layers")
        self.llama_gpu_layers_input.setStyleSheet("""
            QLineEdit {
                background: #2C2C2E;
                color: #EBEBF5;
                border: 1px solid #3A3A3C;
                border-radius: 6px;
                padding: 6px 8px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #e009a7;
                outline: none;
            }
        """)
        tuning_layout.addRow("GPU Layers (-ngl):", self.llama_gpu_layers_input)

        self.llama_batch_size_input = QSpinBox()
        self.llama_batch_size_input.setRange(1, 65536)
        self.llama_batch_size_input.setSingleStep(64)
        self.llama_batch_size_input.setToolTip("Sets -b/--batch-size (prompt processing throughput)")
        self.llama_batch_size_input.setStyleSheet(self.max_tokens_input.styleSheet())
        tuning_layout.addRow("Batch Size (-b):", self.llama_batch_size_input)

        self.llama_ubatch_size_input = QSpinBox()
        self.llama_ubatch_size_input.setRange(1, 65536)
        self.llama_ubatch_size_input.setSingleStep(64)
        self.llama_ubatch_size_input.setToolTip("Sets -ub/--ubatch-size (physical micro-batch)")
        self.llama_ubatch_size_input.setStyleSheet(self.max_tokens_input.styleSheet())
        tuning_layout.addRow("UBatch Size (-ub):", self.llama_ubatch_size_input)

        self.llama_threads_batch_input = QSpinBox()
        self.llama_threads_batch_input.setRange(0, 256)
        self.llama_threads_batch_input.setSpecialValueText("auto")
        self.llama_threads_batch_input.setToolTip("Sets -tb/--threads-batch (0 = same as Threads)")
        self.llama_threads_batch_input.setStyleSheet(self.max_tokens_input.styleSheet())
        tuning_layout.addRow("Batch Threads (-tb):", self.llama_threads_batch_input)

        self.llama_flash_attn_combo = QComboBox()
        self.llama_flash_attn_combo.addItem("Auto", "auto")
        self.llama_flash_attn_combo.addItem("On", "on")
        self.llama_flash_attn_combo.addItem("Off", "off")
        self.llama_flash_attn_combo.setToolTip("Sets -fa/--flash-attn")
        self.llama_flash_attn_combo.setStyleSheet(self.backend_combo.styleSheet())
        tuning_layout.addRow("Flash Attention (-fa):", self.llama_flash_attn_combo)

        self.llama_kv_offload_combo = QComboBox()
        self.llama_kv_offload_combo.addItem("Enabled", True)
        self.llama_kv_offload_combo.addItem("Disabled", False)
        self.llama_kv_offload_combo.setToolTip("Controls -kvo / -nkvo")
        self.llama_kv_offload_combo.setStyleSheet(self.backend_combo.styleSheet())
        tuning_layout.addRow("KV Offload:", self.llama_kv_offload_combo)

        self.llama_numa_combo = QComboBox()
        self.llama_numa_combo.addItem("Disabled", "disabled")
        self.llama_numa_combo.addItem("Distribute", "distribute")
        self.llama_numa_combo.addItem("Isolate", "isolate")
        self.llama_numa_combo.addItem("Numactl", "numactl")
        self.llama_numa_combo.setToolTip("Sets --numa mode")
        self.llama_numa_combo.setStyleSheet(self.backend_combo.styleSheet())
        tuning_layout.addRow("NUMA Mode:", self.llama_numa_combo)

        self.llama_mmap_combo = QComboBox()
        self.llama_mmap_combo.addItem("Enabled", True)
        self.llama_mmap_combo.addItem("Disabled", False)
        self.llama_mmap_combo.setToolTip("Controls --mmap / --no-mmap")
        self.llama_mmap_combo.setStyleSheet(self.backend_combo.styleSheet())
        tuning_layout.addRow("Memory Map:", self.llama_mmap_combo)

        self.llama_mlock_input = QCheckBox("Keep model locked in RAM (--mlock)")
        self.llama_mlock_input.setStyleSheet("color: #EBEBF5; font-size: 12px;")
        self.llama_mlock_input.setToolTip("Prevents swapping/compression; may require enough RAM")
        tuning_layout.addRow("", self.llama_mlock_input)

        self.llama_priority_input = QSpinBox()
        self.llama_priority_input.setRange(-1, 3)
        self.llama_priority_input.setToolTip("Sets --prio: -1 low, 0 normal, 1 medium, 2 high, 3 realtime")
        self.llama_priority_input.setStyleSheet(self.max_tokens_input.styleSheet())
        tuning_layout.addRow("Priority (--prio):", self.llama_priority_input)

        self.llama_poll_input = QSpinBox()
        self.llama_poll_input.setRange(0, 100)
        self.llama_poll_input.setToolTip("Sets --poll (higher = less latency, more CPU)")
        self.llama_poll_input.setStyleSheet(self.max_tokens_input.styleSheet())
        tuning_layout.addRow("Polling (--poll):", self.llama_poll_input)

        self.llama_extra_args_input = QLineEdit()
        self.llama_extra_args_input.setPlaceholderText("Optional extra flags, e.g. --fit off --cache-type-k q8_0")
        self.llama_extra_args_input.setToolTip("Appended to llama-server command (advanced)")
        self.llama_extra_args_input.setStyleSheet(self.llama_gpu_layers_input.styleSheet())
        tuning_layout.addRow("Extra Args:", self.llama_extra_args_input)

        tuning_note = QLabel("Tip: Context Size (-c) and Threads (-t) are in Generation Parameters above.")
        tuning_note.setStyleSheet("color: #999; font-size: 11px;")
        tuning_layout.addRow("", tuning_note)

        self.local_tuning_group.setLayout(tuning_layout)
        layout.addWidget(self.local_tuning_group)
        
        # Appearance group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout()
        
        self.font_family_input = QFontComboBox()
        self.font_family_input.setStyleSheet("""
            QFontComboBox {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QFontComboBox:focus {
                border-color: #e009a7;
            }
            QFontComboBox::drop-down {
                border: none;
                width: 20px;
                border-left: 1px solid #3A3A3C;
            }
            QFontComboBox::drop-down:hover {
                background: #3A3A3C;
                border-left: 1px solid #e009a7;
            }
            QFontComboBox::down-arrow {
                width: 8px;
                height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e009a7;
            }
        """)
        self.font_family_input.setCurrentFont(QFont("SF Pro"))
        appearance_layout.addRow("Chat Font:", self.font_family_input)
        
        self.font_size_input = QSpinBox()
        self.font_size_input.setStyleSheet("""
            QSpinBox {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QSpinBox:focus {
                border-color: #e009a7;
            }
            QSpinBox::up-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QSpinBox::up-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QSpinBox::down-button {
                border: none;
                background: #3A3A3C;
                width: 16px;
            }
            QSpinBox::down-button:hover { background: #48484A; border-left: 1px solid #e009a7; }
            QSpinBox::up-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-bottom: 5px solid #e009a7;
            }
            QSpinBox::down-arrow {
                width: 8px; height: 5px;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #e009a7;
            }
        """)
        self.font_size_input.setRange(8, 24)
        self.font_size_input.setValue(13)
        self.font_size_input.setSuffix(" pt")
        appearance_layout.addRow("Font Size:", self.font_size_input)
        
        appearance_group.setLayout(appearance_layout)
        layout.addWidget(appearance_group)
        
        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.setStyleSheet("""
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
        button_box.accepted.connect(self.save_and_accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def load_settings(self):
        """Load current settings into UI"""
        # Load backend type
        backend_type = self.config.get("backend_type", "local")
        index = self.backend_combo.findData(backend_type)
        if index >= 0:
            self.backend_combo.setCurrentIndex(index)

        self.ollama_path_input.setText(self.config.get("ollama_path", ""))
        self.llama_path_input.setText(self.config.get("llama_cpp_path", ""))
        self.ollama_url_input.setText(self.config.get("ollama_url", "http://localhost:11434"))
        self.hf_api_key_input.setText(self.config.get("hf_api_key", ""))
        self.max_tokens_input.setValue(self.config.get("max_tokens", 512))
        self.temperature_input.setValue(self.config.get("temperature", 0.7))
        self.context_size_input.setValue(self.config.get("context_size", 2048))
        self.threads_input.setValue(self.config.get("threads", 4))
        self.timeout_input.setValue(self.config.get("inference_timeout", 300))
        self.llama_gpu_layers_input.setText(str(self.config.get("llama_gpu_layers", "auto")))
        self.llama_batch_size_input.setValue(self.config.get("llama_batch_size", 2048))
        self.llama_ubatch_size_input.setValue(self.config.get("llama_ubatch_size", 512))
        self.llama_threads_batch_input.setValue(self.config.get("llama_threads_batch", 0))
        self.llama_priority_input.setValue(self.config.get("llama_priority", 0))
        self.llama_poll_input.setValue(self.config.get("llama_poll", 50))
        self.llama_mlock_input.setChecked(bool(self.config.get("llama_mlock", False)))
        self.llama_extra_args_input.setText(self.config.get("llama_extra_args", ""))

        flash_value = self.config.get("llama_flash_attn", "auto")
        flash_index = self.llama_flash_attn_combo.findData(flash_value)
        if flash_index >= 0:
            self.llama_flash_attn_combo.setCurrentIndex(flash_index)

        kv_offload_value = bool(self.config.get("llama_kv_offload", True))
        kv_offload_index = self.llama_kv_offload_combo.findData(kv_offload_value)
        if kv_offload_index >= 0:
            self.llama_kv_offload_combo.setCurrentIndex(kv_offload_index)

        mmap_value = bool(self.config.get("llama_mmap", True))
        mmap_index = self.llama_mmap_combo.findData(mmap_value)
        if mmap_index >= 0:
            self.llama_mmap_combo.setCurrentIndex(mmap_index)

        numa_value = self.config.get("llama_numa", "disabled")
        numa_index = self.llama_numa_combo.findData(numa_value)
        if numa_index >= 0:
            self.llama_numa_combo.setCurrentIndex(numa_index)
        
        # Appearance
        font_family = self.config.get("font_family", "SF Pro")
        self.font_family_input.setCurrentFont(QFont(font_family))
        self.font_size_input.setValue(self.config.get("font_size", 13))
        
        # Update visibility based on backend
        self.on_backend_changed()
    
    def on_backend_changed(self):
        """Handle backend type change - show/hide relevant fields"""
        backend = self.backend_combo.currentData()
        # Show/hide Ollama binary fields
        is_ollama = backend == "ollama"
        self.ollama_binary_label.setVisible(is_ollama)
        self.ollama_path_input.setVisible(is_ollama)
        self.browse_ollama_btn.setVisible(is_ollama)
        
        # Show/hide llama.cpp fields
        is_local = backend == "local"
        self.llama_label.setVisible(is_local)
        self.llama_path_input.setVisible(is_local)
        self.browse_llama_btn.setVisible(is_local)
        self.local_tuning_group.setVisible(is_local)
        
        # Show/hide Ollama fields
        is_ollama = backend == "ollama"
        self.ollama_label.setVisible(is_ollama)
        self.ollama_url_input.setVisible(is_ollama)

    def browse_ollama_path(self):
        """Browse for Ollama binary"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ollama Binary",
            "",
            "Executables (*);;Ollama (ollama)"
        )
        
        if file_path:
            self.ollama_path_input.setText(file_path)
    
    def browse_llama_path(self):
        """Browse for llama.cpp binary"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select llama.cpp Binary",
            "",
            "Executables (*);;llama-server (llama-server)"
        )
        
        if file_path:
            self.llama_path_input.setText(file_path)
    
    def validate_settings(self) -> bool:
        """Validate settings before saving"""
        backend = self.backend_combo.currentData()
        
        if backend == "local":
            llama_path = self.llama_path_input.text().strip()
            gpu_layers = self.llama_gpu_layers_input.text().strip().lower()
            extra_args = self.llama_extra_args_input.text().strip()
            
            # Allow 'bundled' as valid
            if not llama_path:
                QMessageBox.warning(self, "Invalid Settings", "Please specify llama.cpp binary path or use 'bundled'")
                return False

            # Check if it's a valid llama binary (server, cli, or simple-chat)
            if not Path(llama_path).exists() and llama_path != 'bundled':
                QMessageBox.warning(
                    self,
                    "Invalid Settings",
                    f"llama.cpp binary not found:\n{llama_path}\n\nTip: Use 'bundled' for the built-in version"
                )
                return False

            if gpu_layers not in {"auto", "all"}:
                try:
                    int(gpu_layers)
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Invalid Settings",
                        "GPU Layers must be 'auto', 'all', or an integer."
                    )
                    return False

            if self.llama_ubatch_size_input.value() > self.llama_batch_size_input.value():
                QMessageBox.warning(
                    self,
                    "Invalid Settings",
                    "UBatch Size cannot exceed Batch Size."
                )
                return False

            if extra_args:
                try:
                    shlex.split(extra_args)
                except ValueError as exc:
                    QMessageBox.warning(
                        self,
                        "Invalid Settings",
                        f"Extra Args could not be parsed:\n{exc}"
                    )
                    return False
        
        elif backend == "ollama":
            ollama_url = self.ollama_url_input.text().strip()
            if not ollama_url:
                QMessageBox.warning(self, "Invalid Settings", "Please specify Ollama URL")
                return False
            
            # Test connection
            from backend.unified_backend import UnifiedBackend
            if not UnifiedBackend.test_ollama_connection(ollama_url):
                reply = QMessageBox.question(
                    self,
                    "Ollama Not Running",
                    f"Cannot connect to Ollama at {ollama_url}\n\nSave anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return False
        
        elif backend == "huggingface":
            api_key = self.hf_api_key_input.text().strip()
            if not api_key:
                QMessageBox.warning(self, "Invalid Settings", "Please specify HuggingFace API key")
                return False
            
            if not api_key.startswith('hf_'):
                QMessageBox.warning(self, "Invalid Settings", "HuggingFace API keys should start with 'hf_'")
                return False
        
        return True
    
    def save_and_accept(self):
        """Save settings and close dialog"""
        if not self.validate_settings():
            return
        
        # Save Ollama path
        self.config.set("ollama_path", self.ollama_path_input.text().strip())
        
        # Save all settings
        self.config.set("backend_type", self.backend_combo.currentData())
        self.config.set("llama_cpp_path", self.llama_path_input.text().strip())
        self.config.set("ollama_url", self.ollama_url_input.text().strip())
        self.config.set("hf_api_key", self.hf_api_key_input.text().strip())
        self.config.set("max_tokens", self.max_tokens_input.value())
        self.config.set("temperature", self.temperature_input.value())
        self.config.set("context_size", self.context_size_input.value())
        self.config.set("threads", self.threads_input.value())
        self.config.set("inference_timeout", self.timeout_input.value())
        self.config.set("llama_gpu_layers", self.llama_gpu_layers_input.text().strip() or "auto")
        self.config.set("llama_batch_size", self.llama_batch_size_input.value())
        self.config.set("llama_ubatch_size", self.llama_ubatch_size_input.value())
        self.config.set("llama_threads_batch", self.llama_threads_batch_input.value())
        self.config.set("llama_flash_attn", self.llama_flash_attn_combo.currentData())
        self.config.set("llama_kv_offload", bool(self.llama_kv_offload_combo.currentData()))
        self.config.set("llama_mmap", bool(self.llama_mmap_combo.currentData()))
        self.config.set("llama_mlock", self.llama_mlock_input.isChecked())
        self.config.set("llama_numa", self.llama_numa_combo.currentData())
        self.config.set("llama_priority", self.llama_priority_input.value())
        self.config.set("llama_poll", self.llama_poll_input.value())
        self.config.set("llama_extra_args", self.llama_extra_args_input.text().strip())
        self.config.set("font_family", self.font_family_input.currentFont().family())
        self.config.set("font_size", self.font_size_input.value())
        
        self.accept()
