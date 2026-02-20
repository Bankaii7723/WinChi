# app.py
import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QSlider, QListWidget, QStackedWidget, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer
from threading import Thread
from queue import Queue, Empty
from chi_backend import ChiBackend

class ChiApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WINCHI - phi-3-mini GUI")
        self.setGeometry(100, 100, 800, 600)

        self.backend = None
        self.tokens = 256
        self.temperature = 0.7
        self.prefix = ""
        self.stop_generation = False  # <-- New

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Sidebar
        self.sidebar = QListWidget()
        self.sidebar.addItem("Chat")
        self.sidebar.addItem("Settings")
        self.sidebar.setFixedWidth(120)
        main_layout.addWidget(self.sidebar)

        # Stacked pages
        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # --- Chat Page ---
        self.chat_page = QWidget()
        chat_layout = QVBoxLayout()
        self.chat_page.setLayout(chat_layout)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        chat_layout.addWidget(self.chat_display)

        self.entry = QLineEdit()
        self.entry.setPlaceholderText("Enter your message...")
        chat_layout.addWidget(self.entry)

        # Quick action buttons
        q_layout = QHBoxLayout()
        self.btn_eli5 = QPushButton("Explain like I'm 5")
        self.btn_deep = QPushButton("Explain thoroughly")
        self.btn_effi = QPushButton("Explain efficiently")
        q_layout.addWidget(self.btn_eli5)
        q_layout.addWidget(self.btn_deep)
        q_layout.addWidget(self.btn_effi)
        chat_layout.addLayout(q_layout)

        self.btn_eli5.clicked.connect(lambda: self.set_prefix("Explain simply like I'm 5: "))
        self.btn_deep.clicked.connect(lambda: self.set_prefix("Explain thoroughly with examples: "))
        self.btn_effi.clicked.connect(lambda: self.set_prefix("Explain efficiently and concisely: "))

        # --- Stop Generation Button ---
        self.btn_stop = QPushButton("Stop Generation")
        self.btn_stop.clicked.connect(self.stop_response)
        chat_layout.addWidget(self.btn_stop)

        self.stack.addWidget(self.chat_page)

        # --- Settings Page ---
        self.settings_page = QWidget()
        settings_layout = QVBoxLayout()
        self.settings_page.setLayout(settings_layout)

        self.token_label = QPushButton(f"Max tokens: {self.tokens}")
        self.token_slider = QSlider(Qt.Orientation.Horizontal)
        self.token_slider.setRange(1, 2048)
        self.token_slider.setValue(self.tokens)
        self.token_slider.valueChanged.connect(self.update_tokens)
        settings_layout.addWidget(self.token_label)
        settings_layout.addWidget(self.token_slider)

        self.temp_label = QPushButton(f"Temperature: {self.temperature}")
        self.temp_slider = QSlider(Qt.Orientation.Horizontal)
        self.temp_slider.setRange(1, 20)
        self.temp_slider.setValue(int(self.temperature * 10))
        self.temp_slider.valueChanged.connect(self.update_temperature)
        settings_layout.addWidget(self.temp_label)
        settings_layout.addWidget(self.temp_slider)

        self.model_btn = QPushButton("Load phi-3-mini GGUF Model")
        self.model_btn.clicked.connect(self.load_model)
        settings_layout.addWidget(self.model_btn)

        self.stack.addWidget(self.settings_page)
        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)

        # --- Streaming ---
        self.token_queue = Queue()
        self.stream_timer = QTimer()
        self.stream_timer.timeout.connect(self.update_chat)
        self.stream_timer.start(20)
        self.thread_active = False

        self.entry.returnPressed.connect(self.handle_entry)

    # --- Methods ---
    def set_prefix(self, text):
        self.prefix = text

    def update_tokens(self):
        self.tokens = self.token_slider.value()
        self.token_label.setText(f"Max tokens: {self.tokens}")

    def update_temperature(self):
        self.temperature = self.temp_slider.value() / 10
        self.temp_label.setText(f"Temperature: {self.temperature}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select GGUF Model", "", "GGUF Files (*.gguf *.bin)")
        if file_path:
            try:
                self.backend = ChiBackend(file_path)
                self.chat_display.clear()
                self.chat_display.append(f"Model loaded: {file_path.split('/')[-1]}")
            except Exception as e:
                self.chat_display.append(f"Error loading model: {e}")

    def handle_entry(self):
        text = self.entry.text().strip()
        if not text:
            return

        prompt = f"{self.prefix}{text}" if self.prefix else text
        self.entry.clear()
        self.prefix = ""
        self.stop_generation = False  # <-- reset stop flag

        if not self.backend:
            self.chat_display.append("Please load a model first in Settings.")
            return

        while not self.token_queue.empty():
            self.token_queue.get()

        self.chat_display.append(f"User: {text}\nAssistant: ")
        self.thread_active = True

        def run_gen():
            gen = self.backend.gentxt(prompt, tokens=self.tokens, temp=self.temperature, experimental_streaming=True)
            for token in gen:
                if self.stop_generation:
                    break
                self.token_queue.put(token)
            self.thread_active = False

        Thread(target=run_gen, daemon=True).start()

    def stop_response(self):
        self.stop_generation = True

    def update_chat(self):
        updated = False
        try:
            while True:
                token = self.token_queue.get_nowait()
                self.chat_display.moveCursor(self.chat_display.textCursor().MoveOperation.End)
                self.chat_display.insertPlainText(token)
                updated = True
        except Empty:
            pass

        if updated:
            # Auto-scroll to bottom
            self.chat_display.moveCursor(self.chat_display.textCursor().MoveOperation.End)
            self.chat_display.ensureCursorVisible()


# --- Run the App ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ChiApp()
    win.show()
    sys.exit(app.exec())

