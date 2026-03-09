"""
main_window.py
Main application window with chat history side pane
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QComboBox,
    QLabel, QMessageBox, QStatusBar, QListWidget,
    QListWidgetItem, QFrame, QFileDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QAction, QFont, QTextCursor, QPixmap, QKeySequence, QShortcut
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.unified_backend import UnifiedBackend, BackendType
from backend.config import Config
from backend.chat_history import ChatHistory, Conversation
from backend.system_prompts import SystemPromptManager


# ─────────────────────────────────────────────
#  Background generation thread
# ─────────────────────────────────────────────
class GenerationThread(QThread):
    token_generated     = pyqtSignal(str)
    generation_complete = pyqtSignal(dict)
    generation_error    = pyqtSignal(str)

    def __init__(self, backend, model, prompt, max_tokens, temperature, system_prompt="", messages=None):
        super().__init__()
        self.backend       = backend
        self.model         = model
        self.prompt        = prompt
        self.max_tokens    = max_tokens
        self.temperature   = temperature
        self.system_prompt = system_prompt
        self.messages      = messages

    def run(self):
        try:
            full_prompt = self.prompt
            if self.system_prompt:
                full_prompt = f"[SYSTEM]: {self.system_prompt}\n\n[USER]: {self.prompt}"
            for token in self.backend.generate_streaming(
                self.model, full_prompt, self.max_tokens, self.temperature,
                messages=self.messages
            ):
                self.token_generated.emit(token)
            stats = {}
            if hasattr(self.backend, "get_last_generation_stats"):
                stats = self.backend.get_last_generation_stats() or {}
            self.generation_complete.emit(stats)
        except Exception as e:
            self.generation_error.emit(str(e))


# ─────────────────────────────────────────────
#  Main window
# ─────────────────────────────────────────────
class MainWindow(QMainWindow):

    HISTORY_WIDTH = 240

    def __init__(self):
        super().__init__()
        self.config             = Config()
        self.backend            = None
        self.generation_thread  = None
        self.current_model      = None
        self.chat_history       = ChatHistory()
        self.current_conversation = None
        self._current_response  = ""
        self._history_open      = False
        self._backend_signature = None
        self.prompt_manager     = SystemPromptManager()
        self.attached_files     = []  # List of file paths attached to next message

        self.init_ui()
        self.load_configuration()
        self._refresh_history_list()

    # ─── UI construction ──────────────────────
    def init_ui(self):
        self.setWindowTitle("MeMyselfAI")
        self.setGeometry(100, 100, 960, 700)
        self.create_menu_bar()

        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Left: history pane (hidden by default) ──
        self.history_pane = QFrame()
        self.history_pane.setMinimumWidth(0)
        self.history_pane.setMaximumWidth(0)
        self.history_pane.setStyleSheet("""
            QFrame {
                background: #1C1C1E;
                border-right: 1px solid #3A3A3C;
            }
        """)
        pane_layout = QVBoxLayout(self.history_pane)
        pane_layout.setContentsMargins(0, 0, 0, 0)
        pane_layout.setSpacing(0)

        # Pane header
        pane_header = QWidget()
        pane_header.setStyleSheet("background: #2C2C2E;")
        pane_header.setFixedHeight(48)
        ph = QHBoxLayout(pane_header)
        ph.setContentsMargins(12, 0, 8, 0)

        pane_title = QLabel("Chats")
        pane_title.setStyleSheet("color: #F2F2F7; font-size: 15px; font-weight: 600;")
        ph.addWidget(pane_title)
        ph.addStretch()

        new_btn = QPushButton("＋")
        new_btn.setFixedSize(28, 28)
        new_btn.setToolTip("New chat")
        new_btn.setStyleSheet("""
            QPushButton {
                background: #3A3A3C; color: #F2F2F7;
                border: none; border-radius: 6px; font-size: 16px;
            }
            QPushButton:hover { background: #48484A; }
        """)
        new_btn.clicked.connect(self.new_chat)
        ph.addWidget(new_btn)
        pane_layout.addWidget(pane_header)

        # Chat list
        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget {
                background: transparent;
                border: none;
                outline: none;
            }
            QListWidget::item {
                padding: 10px 12px;
                border-bottom: 1px solid #2C2C2E;
                color: #EBEBF5;
            }
            QListWidget::item:selected {
                background: #3A3A3C;
                color: #FFFFFF;
                border-radius: 4px;
            }
            QListWidget::item:hover:!selected {
                background: #2C2C2E;
            }
        """)
        self.history_list.itemClicked.connect(self._on_history_item_clicked)
        pane_layout.addWidget(self.history_list, stretch=1)

        # Delete button
        del_btn = QPushButton("🗑  Delete Selected")
        del_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #FF453A;
                border: none; padding: 10px; font-size: 13px;
            }
            QPushButton:hover { background: #2C2C2E; }
        """)
        del_btn.clicked.connect(self._delete_selected_chat)
        pane_layout.addWidget(del_btn)

        root_layout.addWidget(self.history_pane)

        # ── Right: chat area ───────────────────
        chat_area = QWidget()
        chat_layout = QVBoxLayout(chat_area)
        chat_layout.setContentsMargins(10, 10, 10, 10)
        chat_layout.setSpacing(8)

        # Header row: [☰] centred logo+title [gear button]
        header_row = QHBoxLayout()

        self.toggle_btn = QPushButton("☰")
        self.toggle_btn.setFixedSize(36, 36)
        self.toggle_btn.setToolTip("Toggle chat history")
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #e009a7;
                border: 1px solid #3A3A3C; border-radius: 6px;
                font-size: 20px;
            }
            QPushButton:hover { background: #3A3A3C; border-color: #e009a7; }
        """)
        self.toggle_btn.clicked.connect(self.toggle_history_pane)
        header_row.addWidget(self.toggle_btn, alignment=Qt.AlignmentFlag.AlignTop)

        center_col = QVBoxLayout()
        center_col.setSpacing(2)

        logo_label = QLabel()
        logo_path = Path(__file__).parent.parent / "MeMyselfAi.png"
        if logo_path.exists():
            px = QPixmap(str(logo_path)).scaledToHeight(80, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(px)
        else:
            logo_label.setText("🤖")
            logo_label.setFont(QFont("SF Pro", 40))
        logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_col.addWidget(logo_label)

        title_label = QLabel("MeMyselfAI")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setFont(QFont("SF Pro", 22))
        title_label.setStyleSheet("color: #e009a7;")
        center_col.addWidget(title_label)

        header_row.addLayout(center_col, stretch=1)
        
        # Settings gear button (top right)
        settings_btn = QPushButton("⚙️")
        settings_btn.setFixedSize(32, 32)
        settings_btn.setToolTip("Settings")
        settings_btn.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #EBEBF5;
                border: 1px solid #3A3A3C; border-radius: 6px;
                font-size: 16px;
            }
            QPushButton:hover { background: #3A3A3C; border-color: #e009a7; }
        """)
        settings_btn.clicked.connect(self.open_settings)
        header_row.addWidget(settings_btn, alignment=Qt.AlignmentFlag.AlignTop)
        
        chat_layout.addLayout(header_row)

        chat_layout.addLayout(self.create_top_bar())

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Start a conversation…")
        # Apply font from config
        font_family = self.config.get("font_family", "SF Pro")
        font_size = self.config.get("font_size", 13)
        font = QFont(font_family, font_size)
        self.chat_display.setFont(font)
        chat_layout.addWidget(self.chat_display, stretch=1)

        chat_layout.addLayout(self.create_input_area())
        root_layout.addWidget(chat_area, stretch=1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Keyboard shortcuts for font size (Ctrl/Cmd + and -)
        # ZoomIn standard shortcut
        self.zoom_in_shortcut = QShortcut(QKeySequence.StandardKey.ZoomIn, self)
        self.zoom_in_shortcut.activated.connect(self.increase_font_size)
        
        # Also add explicit Cmd/Ctrl+= (without shift)
        self.zoom_in_equals = QShortcut(QKeySequence("Ctrl+="), self)
        self.zoom_in_equals.activated.connect(self.increase_font_size)
        
        # ZoomOut standard shortcut
        self.zoom_out_shortcut = QShortcut(QKeySequence.StandardKey.ZoomOut, self)
        self.zoom_out_shortcut.activated.connect(self.decrease_font_size)

    def create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        for label, shortcut, slot in [
            ("Settings…",        None,     self.open_settings),
            ("Manage Models…",   "Ctrl+M", self.open_model_manager),
            ("System Prompts…",  "Ctrl+P", self.open_system_prompts),
        ]:
            a = QAction(label, self)
            if shortcut:
                a.setShortcut(shortcut)
            a.triggered.connect(slot)
            file_menu.addAction(a)

        file_menu.addSeparator()
        q = QAction("Quit", self)
        q.setShortcut("Ctrl+Q")
        q.triggered.connect(self.close)
        file_menu.addAction(q)

        edit_menu = menubar.addMenu("Edit")
        c = QAction("Clear Chat", self)
        c.setShortcut("Ctrl+K")
        c.triggered.connect(self.clear_chat)
        edit_menu.addAction(c)

        help_menu = menubar.addMenu("Help")
        a = QAction("About", self)
        a.triggered.connect(self.show_about)
        help_menu.addAction(a)

    def create_top_bar(self):
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Model:"))
        
        # Model combo box with consistent styling
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(300)
        self.model_combo.setStyleSheet("""
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
                width: 8px; height: 5px;
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
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        layout.addWidget(self.model_combo, stretch=1)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #EBEBF5;
                border: 1px solid #3A3A3C; border-radius: 6px;
                padding: 4px 12px; font-size: 12px;
            }
            QPushButton:hover { 
                background: #3A3A3C; 
                border-color: #e009a7; 
            }
        """)
        refresh_btn.clicked.connect(self.refresh_models)
        layout.addWidget(refresh_btn)

        # Active system prompt indicator
        self.prompt_indicator = QPushButton()
        self.prompt_indicator.setToolTip("Click to change system prompt (File → System Prompts)")
        self.prompt_indicator.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #EBEBF5;
                border: 1px solid #3A3A3C; border-radius: 6px;
                padding: 4px 10px; font-size: 12px;
            }
            QPushButton:hover { 
                background: #3A3A3C; 
                border-color: #e009a7; 
            }
        """)
        self.prompt_indicator.clicked.connect(self.open_system_prompts)
        layout.addWidget(self.prompt_indicator)
        self._update_prompt_indicator()

        return layout

    def create_input_area(self):
        layout = QHBoxLayout()
        
        # File upload button
        self.upload_button = QPushButton("📎")
        self.upload_button.setFixedSize(36, 36)
        self.upload_button.setToolTip("Attach files")
        self.upload_button.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #e009a7;
                border: 1px solid #3A3A3C; border-radius: 8px;
                font-size: 18px; font-weight: bold;
            }
            QPushButton:hover { 
                background: #3A3A3C; 
                border-color: #e009a7; 
            }
        """)
        self.upload_button.clicked.connect(self.open_file_picker)
        layout.addWidget(self.upload_button)
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Type your message…")
        self.message_input.setFont(QFont("SF Pro", 13))
        self.message_input.setStyleSheet("""
            QLineEdit {
                background: #2C2C2E; 
                color: #EBEBF5;
                border: 1px solid #3A3A3C; 
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #e009a7;
                outline: none;
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)

        layout.addWidget(self.message_input, stretch=1)

        self.send_button = QPushButton("Send")
        self.send_button.setMinimumWidth(80)
        self.send_button.setStyleSheet("""
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
        self.send_button.clicked.connect(self.send_message)
        layout.addWidget(self.send_button)

        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.setMinimumWidth(80)
        self.stop_button.setStyleSheet("""
            QPushButton {
                background: #2C2C2E; color: #FF453A;
                border: 1px solid #3A3A3C; border-radius: 6px;
                padding: 6px 12px; font-size: 13px;
            }
            QPushButton:hover { 
                background: #3A3A3C; 
                border-color: #e009a7; 
            }
        """)
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.setVisible(False)
        layout.addWidget(self.stop_button)
        return layout
    
    def closeEvent(self, event):
        """Clean up when closing the application"""
        # Stop any in-flight generation first so the process isn't mid-use during cleanup
        if self.generation_thread and self.generation_thread.isRunning():
            print("🛑 Stopping generation thread...")
            if self.backend:
                try:
                    self.backend.stop_generation()
                except Exception:
                    pass
            self.generation_thread.quit()
            self.generation_thread.wait(3000)  # 3s timeout

        if self.backend:
            try:
                self.backend.cleanup()
            except Exception as e:
                print(f"⚠️  Cleanup error: {e}")
        super().closeEvent(event)
    

    # ─── History pane ─────────────────────────
    def toggle_history_pane(self):
        self._history_open = not self._history_open
        target = self.HISTORY_WIDTH if self._history_open else 0

        for prop in (b"minimumWidth", b"maximumWidth"):
            anim = QPropertyAnimation(self.history_pane, prop)
            anim.setDuration(220)
            anim.setStartValue(self.history_pane.width())
            anim.setEndValue(target)
            anim.setEasingCurve(QEasingCurve.Type.OutCubic)
            anim.start()
            # keep reference so it isn't garbage-collected
            setattr(self, f"_anim_{prop.decode()}", anim)

    def _refresh_history_list(self):
        self.history_list.clear()
        for conv in self.chat_history.all():
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, conv.id)
            item.setText(f"{conv.title}\n{conv.formatted_date}")
            self.history_list.addItem(item)

    def _on_history_item_clicked(self, item: QListWidgetItem):
        conv_id = item.data(Qt.ItemDataRole.UserRole)
        conv = self.chat_history.load(conv_id)
        if not conv:
            return
        self.current_conversation = conv
        self.chat_display.clear()
        for msg in conv.messages:
            if msg.role == 'user':
                self.append_message("You", msg.content, "#007AFF")
            elif msg.role == 'assistant':
                self.append_message("Assistant", msg.content, "#34C759")
        self.status_bar.showMessage(f"Loaded: {conv.title}")

    def _delete_selected_chat(self):
        item = self.history_list.currentItem()
        if not item:
            return
        title = item.text().split('\n')[0]
        reply = QMessageBox.question(
            self, "Delete Chat", f'Delete "{title}"?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        conv_id = item.data(Qt.ItemDataRole.UserRole)
        self.chat_history.delete(conv_id)
        if self.current_conversation and self.current_conversation.id == conv_id:
            self.current_conversation = None
            self.chat_display.clear()
        self._refresh_history_list()

    def new_chat(self):
        self.current_conversation = None
        self._current_response = ""
        self.chat_display.clear()
        self.message_input.setFocus()
        self.status_bar.showMessage("New conversation")

    # ─── Font Size Control ───────────────────
    def increase_font_size(self):
        """Increase chat font size (CMD/Ctrl +)"""
        current_font = self.chat_display.font()
        current_size = current_font.pointSize()
        if current_size < 24:  # Max size
            new_size = current_size + 1
            current_font.setPointSize(new_size)
            self.chat_display.setFont(current_font)
            self.config.set("font_size", new_size)
            self.status_bar.showMessage(f"Font size: {new_size} pt")
    
    def decrease_font_size(self):
        """Decrease chat font size (CMD/Ctrl -)"""
        current_font = self.chat_display.font()
        current_size = current_font.pointSize()
        if current_size > 8:  # Min size
            new_size = current_size - 1
            current_font.setPointSize(new_size)
            self.chat_display.setFont(current_font)
            self.config.set("font_size", new_size)
            self.status_bar.showMessage(f"Font size: {new_size} pt")

    # ─── Configuration ────────────────────────
    def _local_backend_kwargs(self, llama_path):
        return {
            "llama_cpp_path": llama_path,
            "context_size": self.config.get("context_size", 2048),
            "threads": self.config.get("threads", 4),
            "inference_timeout": self.config.get("inference_timeout", 300),
            "llama_gpu_layers": self.config.get("llama_gpu_layers", "auto"),
            "llama_batch_size": self.config.get("llama_batch_size", 2048),
            "llama_ubatch_size": self.config.get("llama_ubatch_size", 512),
            "llama_threads_batch": self.config.get("llama_threads_batch", 0),
            "llama_flash_attn": self.config.get("llama_flash_attn", "auto"),
            "llama_kv_offload": self.config.get("llama_kv_offload", True),
            "llama_mmap": self.config.get("llama_mmap", True),
            "llama_mlock": self.config.get("llama_mlock", False),
            "llama_numa": self.config.get("llama_numa", "disabled"),
            "llama_priority": self.config.get("llama_priority", 0),
            "llama_poll": self.config.get("llama_poll", 50),
            "llama_extra_args": self.config.get("llama_extra_args", ""),
        }

    def _backend_config_signature(self, backend_type_str):
        if backend_type_str == "local":
            return (
                "local",
                self.config.get("llama_cpp_path", ""),
                self.config.get("context_size", 2048),
                self.config.get("threads", 4),
                self.config.get("inference_timeout", 300),
                self.config.get("llama_gpu_layers", "auto"),
                self.config.get("llama_batch_size", 2048),
                self.config.get("llama_ubatch_size", 512),
                self.config.get("llama_threads_batch", 0),
                self.config.get("llama_flash_attn", "auto"),
                self.config.get("llama_kv_offload", True),
                self.config.get("llama_mmap", True),
                self.config.get("llama_mlock", False),
                self.config.get("llama_numa", "disabled"),
                self.config.get("llama_priority", 0),
                self.config.get("llama_poll", 50),
                self.config.get("llama_extra_args", ""),
            )
        if backend_type_str == "ollama":
            return (
                "ollama",
                self.config.get("ollama_url", "http://localhost:11434"),
                self.config.get("ollama_path", "bundled"),
                self.config.get("inference_timeout", 300),
            )
        if backend_type_str == "huggingface":
            return (
                "huggingface",
                self.config.get("hf_api_key", ""),
                self.config.get("inference_timeout", 300),
            )
        return ("unknown",)

    def load_configuration(self):
        """Load configuration and initialize backend appropriately"""
        backend_type_str = self.config.get("backend_type", "local")
        target_signature = self._backend_config_signature(backend_type_str)

        if not self.config.is_configured():
            self.status_bar.showMessage("⚠️  Please configure backend in Settings")
            QMessageBox.warning(self, "Configuration Required",
                                "Please configure your backend in Settings.")
            self.open_settings()
            return

        # Create backend only if it doesn't exist or type changed
        backend_needs_update = False
        
        if self.backend is None:
            backend_needs_update = True
            print("🔧 Creating new backend instance")
        else:
            # Check if backend type changed
            current_type = getattr(self.backend, 'backend_type', None)
            target_type = {
                "local": BackendType.LOCAL,
                "ollama": BackendType.OLLAMA, 
                "huggingface": BackendType.HUGGINGFACE
            }.get(backend_type_str)
            
            if current_type != target_type:
                backend_needs_update = True
                print("🔧 Backend type changed, recreating")
            elif self._backend_signature != target_signature:
                backend_needs_update = True
                print("🔧 Backend settings changed, recreating")
                
        if backend_needs_update:
            try:
                # Clean up old backend if it exists
                if self.backend is not None:
                    print("🧹 Cleaning up old backend")
                    try:
                        self.backend.cleanup()
                    except Exception as e:
                        print(f"⚠️  Cleanup warning: {e}")
                    self.backend = None
                    
                if backend_type_str == "local":
                    llama_path = self.config.get_llama_cpp_path()
                    self.backend = UnifiedBackend(BackendType.LOCAL, **self._local_backend_kwargs(llama_path))
                    self.status_bar.showMessage(f"✅ Local backend: {llama_path}")
                    self.refresh_models()

                elif backend_type_str == "ollama":
                    ollama_url = self.config.get("ollama_url", "http://localhost:11434")
                    ollama_path = self.config.get("ollama_path", "bundled")
                    self.backend = UnifiedBackend(
                        BackendType.OLLAMA,
                        ollama_url=ollama_url,
                        ollama_path=ollama_path,
                        inference_timeout=self.config.get("inference_timeout", 300),
                    )
                    self.status_bar.showMessage(f"✅ Ollama backend: {ollama_url}")
                    self.refresh_ollama_models()

                elif backend_type_str == "huggingface":
                    api_key = self.config.get("hf_api_key", "")
                    self.backend = UnifiedBackend(
                        BackendType.HUGGINGFACE,
                        api_key=api_key,
                        inference_timeout=self.config.get("inference_timeout", 300),
                    )
                    self.status_bar.showMessage("✅ HuggingFace backend configured")
                    self.model_combo.clear()
                    self.model_combo.addItem("Enter model name (e.g. meta-llama/Llama-2-7b-chat-hf)")

                self._backend_signature = target_signature

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to initialize backend:\n{e}")
        else:
            print("🔄 Reusing existing backend instance")
            self._backend_signature = target_signature
            # Just refresh UI elements
            if backend_type_str == "local":
                llama_path = self.config.get_llama_cpp_path()
                self.status_bar.showMessage(f"✅ Local backend: {llama_path}")
                self.refresh_models()
            elif backend_type_str == "ollama":
                ollama_url = self.config.get("ollama_url", "http://localhost:11434")
                self.status_bar.showMessage(f"✅ Ollama backend: {ollama_url}")
                self.refresh_ollama_models()
            elif backend_type_str == "huggingface":
                self.status_bar.showMessage("✅ HuggingFace backend configured")
                self.model_combo.clear()
                self.model_combo.addItem("Enter model name (e.g. meta-llama/Llama-2-7b-chat-hf)")


    def refresh_models(self):
        from backend.model_manager import ModelManager
        manager = ModelManager()
        models = manager.get_all_models()
        self.model_combo.clear()
        if not models:
            self.model_combo.addItem("No models – use File → Manage Models to add")
            self.status_bar.showMessage("⚠️  No models configured")
            return
        for m in models:
            self.model_combo.addItem(f"{m.name} ({m.size_mb}MB)", m.path)
        self.status_bar.showMessage(f"Found {len(models)} model(s)")

    def refresh_ollama_models(self):
        import requests
        ollama_url = self.config.get("ollama_url", "http://localhost:11434")
        try:
            r = requests.get(f'{ollama_url}/api/tags', timeout=5)
            r.raise_for_status()
            models = r.json().get('models', [])
            self.model_combo.clear()
            if not models:
                self.model_combo.addItem("No models – use File → Manage Models to download")
                self.status_bar.showMessage("⚠️  No Ollama models downloaded")
                return
            for m in models:
                name = m.get('name', 'unknown')
                size = m.get('size', 0)
                is_cloud = size == 0 or 'cloud' in name.lower()
                label = f"☁️ {name} (Cloud)" if is_cloud else f"💾 {name} ({size/(1024*1024):.0f}MB)"
                self.model_combo.addItem(label, name)
            cloud = sum(1 for m in models if m.get('size', 0) == 0 or 'cloud' in m.get('name', '').lower())
            self.status_bar.showMessage(
                f"Found {len(models)} Ollama model(s) – {cloud} cloud, {len(models)-cloud} local")
        except Exception as e:
            self.model_combo.clear()
            self.model_combo.addItem("Error connecting to Ollama")
            self.status_bar.showMessage(f"❌ Ollama error: {e}")

    def on_model_changed(self, index):
        if index >= 0:
            self.current_model = self.model_combo.itemData(index)
            if self.current_model:
                self.status_bar.showMessage(f"Selected: {self.model_combo.currentText()}")

    # ─── File Upload ──────────────────────────
    def open_file_picker(self):
        """Open file picker to attach files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Files to Attach",
            "",
            "All Files (*.*)"
        )
        if files:
            self.attached_files.extend(files)
            self._update_attachment_display()

    def _update_attachment_display(self):
        """Show attached files above the input box"""
        if not self.attached_files:
            self.message_input.setPlaceholderText("Type your message…")
            return
        
        count = len(self.attached_files)
        names = ", ".join([Path(f).name for f in self.attached_files[:2]])
        if count > 2:
            names += f", +{count-2} more"
        self.message_input.setPlaceholderText(f"📎 {names} — Type your message…")

    def _clear_attachments(self):
        """Clear all attached files"""
        self.attached_files.clear()
        self.message_input.setPlaceholderText("Type your message…")

    # ─── Messaging ────────────────────────────
    def send_message(self):
        message = self.message_input.text().strip()
        
        # Allow sending if there's either a message or attachments
        if not message and not self.attached_files:
            return
        
        if not self.current_model:
            QMessageBox.warning(self, "No Model", "Please select a model first.")
            return
        if not self.backend:
            QMessageBox.warning(self, "No Backend", "Backend not initialized. Check Settings.")
            return

        # Start fresh conversation if needed
        if self.current_conversation is None:
            self.current_conversation = Conversation(model=self.current_model)

        self._current_response = ""
        self.message_input.setEnabled(False)
        self.send_button.setVisible(False)
        self.stop_button.setVisible(True)

        # Display attachments if any
        if self.attached_files:
            attachments_text = "\n".join([f"📎 {Path(f).name}" for f in self.attached_files])
            self.append_message("You", attachments_text, "#007AFF")
        
        # Display message
        display_msg = message if message else "(files attached)"
        self.append_message("You", display_msg, "#007AFF")
        self.message_input.clear()
        
        # Prepare prompt with file CONTENTS
        full_prompt = message if message else "Please analyze the attached files:"
        
        if self.attached_files:
            full_prompt += "\n\n"
            for filepath in self.attached_files:
                filename = Path(filepath).name
                try:
                    # Try to read as text
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # Limit each file to ~10k chars to avoid token limits
                        if len(content) > 10000:
                            content = content[:10000] + "\n\n[... truncated ...]"
                        full_prompt += f"\n--- File: {filename} ---\n{content}\n"
                except Exception as e:
                    full_prompt += f"\n--- File: {filename} ---\n[Error reading file: {e}]\n"
        
        self.append_message("Assistant", "", "#34C759")

        # Build full messages list for Ollama /api/chat so the model has conversation history
        system_prompt_text = self.prompt_manager.active.prompt
        messages = []
        if system_prompt_text:
            messages.append({"role": "system", "content": system_prompt_text})
        if self.current_conversation:
            for msg in self.current_conversation.messages:
                messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": full_prompt})

        self.generation_thread = GenerationThread(
            self.backend, self.current_model, full_prompt,
            self.config.get("max_tokens", 512),
            self.config.get("temperature", 0.7),
            system_prompt_text,
            messages=messages
        )
        self.generation_thread.token_generated.connect(self.on_token_generated)
        self.generation_thread.generation_complete.connect(self.on_generation_complete)
        self.generation_thread.generation_error.connect(self.on_generation_error)
        self.generation_thread.start()
        self.status_bar.showMessage("Generating response…")

        # Save user turn with attachments info
        save_msg = full_prompt if self.attached_files else message
        self.current_conversation.add_message('user', save_msg)
        
        # Clear attachments after sending
        self._clear_attachments()

    def on_token_generated(self, token: str):
        self._current_response += token
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def on_generation_complete(self, stats: dict):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        prompt_tps = stats.get("prompt_tps")
        generation_tps = stats.get("generation_tps")
        if prompt_tps is not None and generation_tps is not None:
            cursor.insertText("\n")
            # Keep default font family/size; only colorize.
            cursor.insertHtml(
                f'<span style="color:#8B5CF6;">'
                f'[ Prompt: {prompt_tps:.1f} t/s | Generation: {generation_tps:.1f} t/s ]'
                f'</span>'
            )
        cursor.insertText("\n")
        self.chat_display.setTextCursor(cursor)

        # Persist assistant response
        if self.current_conversation and self._current_response:
            self.current_conversation.add_message('assistant', self._current_response)
            self.chat_history.save(self.current_conversation)
            self._refresh_history_list()

        self._current_response = ""
        self.message_input.setEnabled(True)
        self.send_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.status_bar.showMessage("Ready")
        self.message_input.setFocus()

    def on_generation_error(self, error: str):
        self.append_message("System", f"Error: {error}", "#FF3B30")
        self.message_input.setEnabled(True)
        self.send_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.status_bar.showMessage("Error occurred")

    def stop_generation(self):
        if self.backend:
            self.backend.stop_generation()
        if self.generation_thread:
            self.generation_thread.quit()
            self.generation_thread.wait()
        self.message_input.setEnabled(True)
        self.send_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.status_bar.showMessage("Generation stopped")

    def append_message(self, sender: str, message: str, color: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        plain = self.chat_display.toPlainText()
        if plain and not plain.endswith('\n'):
            cursor.insertText("\n")
        cursor.insertHtml(f'<b style="color:{color};">{sender}:</b> ')
        if message:
            cursor.insertText(message)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def open_system_prompts(self):
        from ui.system_prompts_dialog import SystemPromptsDialog
        dialog = SystemPromptsDialog(self.prompt_manager, self)
        dialog.prompt_selected.connect(self._on_prompt_selected)
        dialog.exec()

    def _on_prompt_selected(self, prompt_id: str):
        self._update_prompt_indicator()
        sp = self.prompt_manager.active
        self.status_bar.showMessage(f"System prompt: {sp.icon} {sp.name}")

    def _update_prompt_indicator(self):
        sp = self.prompt_manager.active
        self.prompt_indicator.setText(f"{sp.icon} {sp.name}")

    # ─── Misc ─────────────────────────────────
    def clear_chat(self):
        reply = QMessageBox.question(
            self, "Clear Chat",
            "Clear the current chat?\n(Past conversations are preserved in the history pane)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.current_conversation = None
            self._current_response = ""
            self.chat_display.clear()
            self.status_bar.showMessage("Chat cleared")

    def open_settings(self):
        from ui.settings_dialog import SettingsDialog
        dialog = SettingsDialog(self.config, self)
        if dialog.exec():
            # Reload font from config
            font_family = self.config.get("font_family", "SF Pro")
            font_size = self.config.get("font_size", 13)
            self.chat_display.setFont(QFont(font_family, font_size))
            self.load_configuration()

    def open_model_manager(self):
        backend_type = self.config.get("backend_type", "local")
        if backend_type == "ollama":
            from ui.ollama_manager_dialog import OllamaManagerDialog
            dialog = OllamaManagerDialog(
                self.config.get("ollama_url", "http://localhost:11434"), self)
            dialog.exec()
            self.refresh_ollama_models()
        else:
            from ui.model_manager_dialog import ModelManagerDialog
            dialog = ModelManagerDialog(self)
            if dialog.exec():
                self.refresh_models()

    def show_about(self):
        QMessageBox.about(self, "About MeMyselfAI",
            "<h3>MeMyselfAI</h3>"
            "<p>A local AI chat application powered by llama.cpp</p>"
            "<p>Version 1.0.0</p>"
            "<p>Built with Python &amp; PyQt6</p>")
