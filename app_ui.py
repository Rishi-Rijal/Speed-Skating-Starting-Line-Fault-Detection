# app_ui.py  (replace file)
from PyQt5.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QSpinBox, QDoubleSpinBox, QCheckBox
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
        self.pre_start_min = QSpinBox(); self.pre_start_min.setRange(1, 500); self.pre_start_min.setValue(180)
        layout.addLayout(self._labeled_input("Pre-Start Zone Min (Map Y):", self.pre_start_min))

        self.pre_start_max = QSpinBox(); self.pre_start_max.setRange(1, 500); self.pre_start_max.setValue(220)
        layout.addLayout(self._labeled_input("Pre-Start Zone Max (Map Y):", self.pre_start_max))

        self.start_line = QSpinBox(); self.start_line.setRange(1, 500); self.start_line.setValue(400)
        layout.addLayout(self._labeled_input("Start Line (Map Y):", self.start_line))

        # Movement thresholds
        self.threshold = QDoubleSpinBox(); self.threshold.setDecimals(3); self.threshold.setSingleStep(0.005)
        self.threshold.setRange(0.001, 0.1); self.threshold.setValue(0.02)
        layout.addLayout(self._labeled_input("Movement Threshold (significant):", self.threshold))

        self.micro_tremor = QDoubleSpinBox(); self.micro_tremor.setDecimals(3); self.micro_tremor.setSingleStep(0.001)
        self.micro_tremor.setRange(0.001, 0.05); self.micro_tremor.setValue(0.008)
        layout.addLayout(self._labeled_input("Micro-tremor Threshold (ignore):", self.micro_tremor))

        # Timings (seconds)
        self.settle_breath = QDoubleSpinBox(); self.settle_breath.setDecimals(1); self.settle_breath.setRange(0.0, 10.0); self.settle_breath.setValue(1.0)
        layout.addLayout(self._labeled_input("Settle/Breath before READY (s):", self.settle_breath))

        self.ready_timeout = QDoubleSpinBox(); self.ready_timeout.setDecimals(1); self.ready_timeout.setRange(1.0, 10.0); self.ready_timeout.setValue(3.0)
        layout.addLayout(self._labeled_input("Time to assume start after READY (s):", self.ready_timeout))

        self.hold_time = QDoubleSpinBox(); self.hold_time.setDecimals(2); self.hold_time.setRange(0.5, 5.0); self.hold_time.setValue(1.10)
        layout.addLayout(self._labeled_input("Hold (READY→gun) (s):", self.hold_time))

        # Lane orientation
        self.inner_left = QCheckBox(); self.inner_left.setChecked(True)
        layout.addLayout(self._labeled_input("Inner Lane is on the Camera's Left", self.inner_left))

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
            "microTremor": self.micro_tremor.value(),
            "settleBreathSeconds": float(self.settle_breath.value()),
            "readyAssumeTimeout": float(self.ready_timeout.value()),
            "holdPauseSeconds": float(self.hold_time.value()),
            "innerOnLeft": bool(self.inner_left.isChecked())
        }
        with open("config.json", "w") as f:
            json.dump(settings, f, indent=4)
        print("Settings saved to config.json")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SkatingControlPanel()
    window.show()
    sys.exit(app.exec_())
