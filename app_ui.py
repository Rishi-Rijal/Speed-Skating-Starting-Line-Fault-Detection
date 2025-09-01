from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSpinBox, QDoubleSpinBox
import sys
import json

class SkatingControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skating Detection Settings")
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Line Constants for a 1000x500 top-down map
        self.pre_start_min = QSpinBox()
        self.pre_start_min.setRange(1, 500)
        self.pre_start_min.setValue(180)
        layout.addLayout(self._labeled_input("Pre-Start Zone Min (Map Y):", self.pre_start_min))

        self.pre_start_max = QSpinBox()
        self.pre_start_max.setRange(1, 500)
        self.pre_start_max.setValue(220)
        layout.addLayout(self._labeled_input("Pre-Start Zone Max (Map Y):", self.pre_start_max))

        self.start_line = QSpinBox()
        self.start_line.setRange(1, 500)
        self.start_line.setValue(400)
        layout.addLayout(self._labeled_input("Start Line (Map Y):", self.start_line))

        self.threshold = QDoubleSpinBox()
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(0.005)
        self.threshold.setRange(0.001, 0.1)
        self.threshold.setValue(0.02)
        layout.addLayout(self._labeled_input("Movement Threshold:", self.threshold))

        # Sound file inputs omitted for brevity but should remain the same as your original

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

    def apply_settings(self):
        settings = {
            "preStartMin": self.pre_start_min.value(),
            "preStartMax": self.pre_start_max.value(),
            "startLine": self.start_line.value(),
            "threshold": self.threshold.value(),
            # Sound settings would be saved here as well
        }
        with open("config.json", "w") as f:
            json.dump(settings, f, indent=4)
        print("Settings saved to config.json")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SkatingControlPanel()
    window.show()
    sys.exit(app.exec_())