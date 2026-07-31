"""
logs_dialog.py
A window for displaying categorized application logs in real-time.
"""

import logging
from datetime import datetime
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTabWidget, QPlainTextEdit,
    QPushButton, QHBoxLayout, QFileDialog
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from backend.logger import qt_log_handler


class LogsDialog(QDialog):
    """Dialog to display real-time application logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application Logs")
        self.setMinimumSize(800, 600)

        self.log_widgets = {}
        self.is_reloading = False
        self.init_ui()

        # Connect to the global Qt log handler for future logs
        qt_log_handler.new_log_record.connect(self.on_new_log_record)

    def init_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3A3A3C;
                border-top: none;
            }
            QTabBar::tab {
                background: #2C2C2E;
                color: #EBEBF5;
                padding: 8px 15px;
                border: 1px solid #3A3A3C;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #1C1C1E;
                border-color: #3A3A3C;
            }
            QTabBar::tab:!selected:hover {
                background: #3A3A3C;
            }
        """)

        # Create tabs for different log categories
        log_categories = ["All", "App", "Backend", "Network"]
        for category in log_categories:
            log_edit = QPlainTextEdit()
            log_edit.setReadOnly(True)
            log_edit.setFont(QFont("SF Mono", 11))
            log_edit.setStyleSheet("""
                QPlainTextEdit {
                    background-color: #1C1C1E;
                    color: #E0E0E0;
                    border: none;
                }
            """)
            self.tabs.addTab(log_edit, category)
            self.log_widgets[category.lower()] = log_edit

        layout.addWidget(self.tabs)

        # Bottom button layout
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save Shown Logs")
        save_button.clicked.connect(self.save_shown_logs)
        button_layout.addWidget(save_button)

        clear_button = QPushButton("Clear All Logs")
        clear_button.clicked.connect(self.clear_all_logs)
        button_layout.addWidget(clear_button)

        button_layout.addStretch()

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)

        layout.addLayout(button_layout)

    def showEvent(self, event):
        """When the dialog is shown, reload all logs from cache."""
        super().showEvent(event)
        self.reload_logs()

    def reload_logs(self):
        """Clear and reload all logs from the cache."""
        self.is_reloading = True
        self.clear_all_logs()
        cached_records = qt_log_handler.get_cache()
        for record in cached_records:
            self.on_new_log_record(record, is_realtime=False)
        self.is_reloading = False

    def on_new_log_record(self, record: logging.LogRecord, is_realtime=True):
        """Append a new log record to the appropriate text widgets."""
        if self.is_reloading and is_realtime:
            return

        formatted_message = qt_log_handler.format(record)

        # Append to "All" tab
        self.log_widgets["all"].appendPlainText(formatted_message)

        # Append to specific category tab
        logger_name = record.name.split('.')[0] # e.g., 'app', 'backend'
        if logger_name in self.log_widgets:
            self.log_widgets[logger_name].appendPlainText(formatted_message)
        elif 'urllib3' in record.name and 'network' in self.log_widgets:
            self.log_widgets['network'].appendPlainText(formatted_message)
    
    def clear_all_logs(self):
        """Clear the text in all log widgets."""
        for widget in self.log_widgets.values():
            widget.clear()

    def save_shown_logs(self):
        """Save the content of the currently visible log pane to a file."""
        current_widget = self.tabs.currentWidget()
        if not isinstance(current_widget, QPlainTextEdit):
            return

        log_content = current_widget.toPlainText()
        if not log_content:
            return
        
        # Get the current tab's name
        current_tab_index = self.tabs.currentIndex()
        current_tab_name = self.tabs.tabText(current_tab_index).lower()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        default_filename = f"memyselfai_{current_tab_name}_log_{timestamp}.txt"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Logs",
            default_filename,
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(log_content)
            except Exception as e:
                # In a real app, you might show an error message box here
                print(f"Error saving log file: {e}")

    def closeEvent(self, event):
        """Disconnect the signal on close to prevent errors."""
        try:
            qt_log_handler.new_log_record.disconnect(self.on_new_log_record)
        except TypeError:
            # This can happen if the signal is already disconnected
            pass
        super().closeEvent(event)