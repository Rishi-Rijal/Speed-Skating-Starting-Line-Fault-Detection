from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog, QLabel, QPushButton, QSlider, QVBoxLayout, QHBoxLayout, QWidget, QSpinBox, QDoubleSpinBox, QApplication
import sys
import json

class SkatingControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skating Detection Settings")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Line Constants
        self.pre_start_min = QSpinBox()
        self.pre_start_min.setRange(1, 800)
        self.pre_start_min.setValue(200)
        layout.addLayout(self._labeled_input("Pre-Start Line Min (Y):", self.pre_start_min))

        self.pre_start_max = QSpinBox()
        self.pre_start_max.setRange(1, 800)
        self.pre_start_max.setValue(220)
        layout.addLayout(self._labeled_input("Pre-Start Line Max (Y):", self.pre_start_max))

        self.start_line = QSpinBox()
        self.start_line.setRange(1, 800)
        self.start_line.setValue(400)
        layout.addLayout(self._labeled_input("Start Line (Y):", self.start_line))

        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.005)
        self.threshold.setRange(0.01, 0.1)
        self.threshold.setValue(0.02)
        layout.addLayout(self._labeled_input("Movement Threshold:", self.threshold))

        # Sound file inputs
        self.go_sound_btn = QPushButton("Choose Go Sound")
        self.go_sound_btn.clicked.connect(lambda: self._choose_file("go_sound"))
        layout.addWidget(self.go_sound_btn)

        self.ready_sound_btn = QPushButton("Choose Ready Sound")
        self.ready_sound_btn.clicked.connect(lambda: self._choose_file("ready_sound"))
        layout.addWidget(self.ready_sound_btn)

        self.false_start_sound_btn = QPushButton("Choose False Start Sound")
        self.false_start_sound_btn.clicked.connect(lambda: self._choose_file("false_start_sound"))
        layout.addWidget(self.false_start_sound_btn)

        self.gun_sound_btn = QPushButton("Choose Gun Sound")
        self.gun_sound_btn.clicked.connect(lambda: self._choose_file("gun_sound"))
        layout.addWidget(self.gun_sound_btn)

        # Apply Button
        apply_button = QPushButton("Apply Settings")
        apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(apply_button)

        self.setLayout(layout)

    def _labeled_input(self, label_text, input_widget):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        layout.addWidget(input_widget)
        return layout

    def _choose_file(self, name):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Sound File", "", "Audio Files (*.mp3 *.wav)")
        if file_path:
            setattr(self, name, file_path)
            print(f"{name} file set to: {file_path}")

    def apply_settings(self):
        settings = {
            "preStartMin": self.pre_start_min.value(),
            "preStartMax": self.pre_start_max.value(),
            "startLine": self.start_line.value(),
            "threshold": self.threshold.value(),
            "goSound": getattr(self, "go_sound", "Sounds/goSound.mp3"),
            "readySound": getattr(self, "ready_sound", "Sounds/readySound.mp3"),
            "falseStartSound": getattr(self, "false_start_sound", "Sounds/falseStartBuzzer.mp3"),
            "gunSound": getattr(self, "gun_sound", "gunSound.mp3")
        }

        with open("config.json", "w") as f:
            json.dump(settings, f)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SkatingControlPanel()
    window.show()
    sys.exit(app.exec_())
