# app_ui.py
import sys, json, os
from PyQt5.QtCore import Qt, QProcess, QTimer, pyqtSignal, QPoint
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QDoubleSpinBox, QCheckBox, QTextEdit, QFileDialog, QMessageBox, QGroupBox, QGridLayout
)
from PyQt5.QtGui import QPixmap, QPainter, QPen

from video_worker import VideoWorker
import numpy as np
import cv2

from PyQt5.QtWidgets import QSizePolicy


APP_PY = "main.py"               # fault detection entry point
HOMOGRAPHY_NPY = "homography_matrix.npy"

# ---------- Clickable video canvas for Homography ----------
class ClickableVideo(QLabel):
    point_added = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self._pix = None
        self.points = []  # list of QPoint in widget coords (we keep logical points after mapping)

    def set_frame(self, qimg):
        self._pix = QPixmap.fromImage(qimg)
        self.update()

    def clear_points(self):
        self.points = []
        self.update()

    def mousePressEvent(self, event):
        if not self._pix: return
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.points.append(QPoint(pos))
            self.update()
            self.point_added.emit(pos.x(), pos.y())

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._pix: return
        p = QPainter(self)
        # scale pixmap to fit
        scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        p.drawPixmap(x, y, scaled)
        # draw picked points
        pen = QPen(Qt.yellow, 3)
        p.setPen(pen)
        for i, pt in enumerate(self.points):
            p.drawEllipse(pt, 6, 6)
            p.drawText(pt + QPoint(8, -8), f"{i+1}")

# ---------- Main Window ----------
class SkatingControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skating — Unified Control Panel")
        self.proc_main = None
        self._live_dir = "live"

        # theme
        self.setStyleSheet("""
            QWidget { font-family: Segoe UI, Roboto, Arial; font-size: 11pt; color: #e5e5e5; }
            QWidget { background: #121417; }
            QGroupBox { border: 1px solid #2a2f36; border-radius: 6px; margin-top: 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9fb3c8; }
            QPushButton { background: #1e2530; border: 1px solid #2a2f36; border-radius: 6px; padding: 8px 14px; }
            QPushButton:hover { background: #243042; }
            QPushButton:pressed { background: #2a3a52; }
            QPushButton:disabled { color: #7b8794; border-color: #243042; }
            QTabBar::tab { background: #1a1f26; padding: 8px 14px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:selected { background: #243042; color: #ffffff; }
            QSpinBox, QDoubleSpinBox, QLineEdit { background: #0f1216; border: 1px solid #2a2f36; border-radius: 6px; padding: 6px; }
            QTextEdit { background: #0f1216; border: 1px solid #2a2f36; border-radius: 6px; }
            QLabel#status_ok { color: #42d392; } 
            QLabel#status_run { color: #f39c12; } 
            QLabel#status_err { color: #ff6b6b; }
        """)

        self._build_ui()
        self._load_existing_config_if_any()

        # embedded visual polling
        self._visual_timer = QTimer(self)
        self._visual_timer.setInterval(100)  # ~10 fps
        self._visual_timer.timeout.connect(self._update_visuals)

        self._left_path = os.path.join("live", "left.jpg")
        self._right_path = os.path.join("live", "right.jpg")
        self._left_mtime = 0.0
        self._right_mtime = 0.0

        QTimer.singleShot(0, self._update_buttons)

    # ---------- UI ----------
    def _build_ui(self):
        root = QVBoxLayout(self)

        # --- Run controls ---
        run_box = QGroupBox("Run Controls")
        run_row = QHBoxLayout()
        self.btn_start = QPushButton("Start")
        self.btn_restart = QPushButton("Restart")
        self.btn_stop = QPushButton("Stop")
        self.btn_start.clicked.connect(self.start_main)
        self.btn_restart.clicked.connect(self.restart_main)
        self.btn_stop.clicked.connect(self.stop_main)
        for b in (self.btn_start, self.btn_restart, self.btn_stop):
            run_row.addWidget(b)
        run_row.addStretch(1)
        run_box.setLayout(run_row)
        root.addWidget(run_box)

        # --- Tabs ---
        tabs = QTabWidget()
        tabs.addTab(self._tab_dashboard(), "Dashboard")
        tabs.addTab(self._tab_homography(), "Homography")
        tabs.addTab(self._tab_settings(), "Settings")
        tabs.addTab(self._tab_visuals(), "Visuals")
        tabs.addTab(self._tab_logs(), "Logs")

        root.addWidget(tabs, 1)


        # --- Status ---
        status_row = QHBoxLayout()
        self.status_lbl = QLabel("Status: idle")
        self.status_lbl.setObjectName("status_ok")
        status_row.addWidget(self.status_lbl)
        status_row.addStretch(1)
        root.addLayout(status_row)

    # ---------- Tabs ----------
    def _tab_dashboard(self):
        w = QWidget(); lay = QVBoxLayout(w)

        # Controls
        row = QHBoxLayout()
        row.addWidget(QLabel("Camera index:"))
        self.cam_index_spin = QSpinBox()
        self.cam_index_spin.setRange(0, 99)
        self.cam_index_spin.setValue(0)
        row.addWidget(self.cam_index_spin)

        self.btn_scan = QPushButton("Scan Cameras")
        self.btn_scan.clicked.connect(self._scan_cameras)
        row.addWidget(self.btn_scan)

        self.btn_preview = QPushButton("Start Preview")
        self.btn_preview.clicked.connect(self._toggle_preview)
        row.addWidget(self.btn_preview)

        row.addStretch(1)
        lay.addLayout(row)

        # Video area
        self.preview_label = QLabel("Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(360)
        self.preview_label.setStyleSheet("border: 1px solid #2a2f36; border-radius: 6px;")
        lay.addWidget(self.preview_label, 1)

        # worker ref
        self.preview_worker = None
        return w

    def _tab_homography(self):
        w = QWidget(); lay = QVBoxLayout(w)

        info = QLabel("Click 4 points (in order) on the track (e.g., corners / reference points). Then Save.")
        lay.addWidget(info)

        # controls
        ctrl = QHBoxLayout()
        self.btn_homo_start = QPushButton("Start Camera")
        self.btn_homo_start.clicked.connect(self._toggle_homo_cam)
        self.btn_homo_clear = QPushButton("Clear Points")
        self.btn_homo_clear.clicked.connect(self._homo_clear_points)
        self.btn_homo_save = QPushButton("Compute & Save Homography")
        self.btn_homo_save.clicked.connect(self._homo_save)
        for b in (self.btn_homo_start, self.btn_homo_clear, self.btn_homo_save):
            ctrl.addWidget(b)
        ctrl.addStretch(1)
        lay.addLayout(ctrl)

        # video canvas (clickable)
        self.homo_canvas = ClickableVideo()
        self.homo_canvas.setMinimumHeight(360)
        self.homo_canvas.setStyleSheet("border: 1px solid #2a2f36; border-radius: 6px;")
        lay.addWidget(self.homo_canvas, 1)

        self.homo_worker = None
        # track logical points in original frame coords
        self._homo_frame_size = None   # (w,h)
        self.homo_canvas.point_added.connect(self._homo_on_click)
        self._homo_points_src = []     # in frame coords
        return w

    def _tab_settings(self):
        w = QWidget()
        g = QGridLayout(w)

        r = 0
        def add_row(lbl, widget):
            nonlocal r
            g.addWidget(QLabel(lbl), r, 0)
            g.addWidget(widget, r, 1)
            r += 1

        # geometry
        self.pre_start_min = QSpinBox(); self.pre_start_min.setRange(1, 700); self.pre_start_min.setValue(180)
        self.pre_start_max = QSpinBox(); self.pre_start_max.setRange(1, 700); self.pre_start_max.setValue(220)
        self.start_line    = QSpinBox(); self.start_line.setRange(1, 700);  self.start_line.setValue(400)

        # thresholds
        self.threshold = QDoubleSpinBox(); self.threshold.setDecimals(3); self.threshold.setSingleStep(0.001); self.threshold.setRange(0.001, 0.1); self.threshold.setValue(0.015)
        self.micro_tremor = QDoubleSpinBox(); self.micro_tremor.setDecimals(3); self.micro_tremor.setSingleStep(0.001); self.micro_tremor.setRange(0.001, 0.05); self.micro_tremor.setValue(0.008)

        # timings
        self.settle_breath = QDoubleSpinBox(); self.settle_breath.setDecimals(1); self.settle_breath.setRange(0.0, 10.0); self.settle_breath.setValue(1.0)
        self.ready_timeout = QDoubleSpinBox(); self.ready_timeout.setDecimals(1); self.ready_timeout.setRange(1.0, 10.0); self.ready_timeout.setValue(3.0)
        self.hold_time     = QDoubleSpinBox(); self.hold_time.setDecimals(2); self.hold_time.setRange(0.5, 5.0); self.hold_time.setValue(1.10)

        # options
        self.inner_left = QCheckBox(); self.inner_left.setChecked(False)

        # camera index also lives here (saved to config)
        self.settings_cam_index = QSpinBox(); self.settings_cam_index.setRange(0, 99); self.settings_cam_index.setValue(0)

        add_row("Pre-Start Zone Min (Map Y):", self.pre_start_min)
        add_row("Pre-Start Zone Max (Map Y):", self.pre_start_max)
        add_row("Start Line (Map Y):",         self.start_line)
        add_row("Movement Threshold (significant):", self.threshold)
        add_row("Micro-tremor Threshold (ignore):",  self.micro_tremor)
        add_row("Settle/Breath before READY (s):",   self.settle_breath)
        add_row("Time to assume start after READY (s):", self.ready_timeout)
        add_row("Hold (READY → gun) (s):",           self.hold_time)
        add_row("Inner Lane is on the Camera's Left", self.inner_left)
        add_row("Default Camera Index", self.settings_cam_index)

        btn_row = QHBoxLayout()
        self.btn_save = QPushButton("Save Settings")
        self.btn_save.clicked.connect(self.save_settings)
        btn_row.addWidget(self.btn_save)
        btn_row.addStretch(1)
        g.addLayout(btn_row, r, 0, 1, 2)
        return w

    def _tab_logs(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lay.addWidget(self.log, 1)
        btn_row = QHBoxLayout()
        export_btn = QPushButton("Save Log to File")
        export_btn.clicked.connect(self._save_log_to_file)
        btn_row.addWidget(export_btn); btn_row.addStretch(1)
        lay.addLayout(btn_row)
        return w
    
    def _tab_visuals(self):
        w = QWidget()
        lay = QVBoxLayout(w)

        grid = QHBoxLayout()
        left_box = QVBoxLayout()
        right_box = QVBoxLayout()

        grid.setStretch(0, 1)
        grid.setStretch(1, 1)

        self.left_img = QLabel("Waiting for frames…")
        self.left_img.setAlignment(Qt.AlignCenter)
        self.left_img.setMinimumHeight(400)
        self.left_img.setStyleSheet("border: 1px solid #2a2f36; border-radius: 6px;")
        self.left_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.left_img.setScaledContents(True)

        self.right_img = QLabel("Waiting for frames…")
        self.right_img.setAlignment(Qt.AlignCenter)
        self.right_img.setMinimumHeight(400)
        self.right_img.setStyleSheet("border: 1px solid #2a2f36; border-radius: 6px;")
        self.right_img.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.right_img.setScaledContents(True)


        left_box.addWidget(self.left_img)
        right_box.addWidget(self.right_img)
        grid.addLayout(left_box)
        grid.addLayout(right_box)

        lay.addLayout(grid)
        return w


    # ---------- Config I/O ----------
    def _load_existing_config_if_any(self):
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    cfg = json.load(f)
                self.pre_start_min.setValue(int(cfg.get("preStartMin", 180)))
                self.pre_start_max.setValue(int(cfg.get("preStartMax", 220)))
                self.start_line.setValue(int(cfg.get("startLine", 400)))
                self.threshold.setValue(float(cfg.get("threshold", 0.015)))
                self.micro_tremor.setValue(float(cfg.get("microTremor", 0.008)))
                self.settle_breath.setValue(float(cfg.get("settleBreathSeconds", 1.0)))
                self.ready_timeout.setValue(float(cfg.get("readyAssumeTimeout", 3.0)))
                self.hold_time.setValue(float(cfg.get("holdPauseSeconds", 1.10)))
                self.inner_left.setChecked(bool(cfg.get("innerOnLeft", False)))
                cam_idx = int(cfg.get("cameraIndex", 0))
                self.settings_cam_index.setValue(cam_idx)
                self.cam_index_spin.setValue(cam_idx)
                self._append_log("Loaded existing config.json")
            except Exception as e:
                self._append_log(f"Failed to load config.json: {e}")

    def _collect_settings(self):
        return {
            "preStartMin": self.pre_start_min.value(),
            "preStartMax": self.pre_start_max.value(),
            "startLine": self.start_line.value(),
            "threshold": float(self.threshold.value()),
            "microTremor": float(self.micro_tremor.value()),
            "settleBreathSeconds": float(self.settle_breath.value()),
            "readyAssumeTimeout": float(self.ready_timeout.value()),
            "holdPauseSeconds": float(self.hold_time.value()),
            "innerOnLeft": bool(self.inner_left.isChecked()),
            "cameraIndex": int(self.settings_cam_index.value()),
        }

    def save_settings(self):
        try:
            with open("config.json", "w") as f:
                json.dump(self._collect_settings(), f, indent=4)
            self._append_log("Settings saved to config.json")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ---------- Process management (main.py with external OpenCV windows) ----------
    def start_main(self):
        if self.proc_main and self.proc_main.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Already Running", "Fault detection is already running.")
            return
        self.save_settings()
        self._append_log("Launching main.py …")
        self.proc_main = QProcess(self)
        python = sys.executable
        self.proc_main.setProgram(python)
        # Pass camera index as an argument so main.py can pick it up.
        cam_idx = str(self.settings_cam_index.value())
        self.proc_main.setArguments([APP_PY, "--camera-index", cam_idx])
        self.proc_main.setProcessChannelMode(QProcess.MergedChannels)
        self.proc_main.readyReadStandardOutput.connect(self._read_main_stdout)
        self.proc_main.finished.connect(self._main_exited)
        self.proc_main.errorOccurred.connect(self._main_error)
        # Ensure no other threads are holding the camera
        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker.wait(500)
            self.preview_worker = None

        if hasattr(self, "homo_worker") and self.homo_worker and self.homo_worker.isRunning():
            self.homo_worker.stop()
            self.homo_worker.wait(500)
            self.homo_worker = None

        self.proc_main.start()
        self._start_visual_timer()
        self._set_status("running")
        self._update_buttons()

    def restart_main(self):
        self._append_log("Restart requested …")
        if self.proc_main and self.proc_main.state() != QProcess.NotRunning:
            self.stop_main(after_cb=self.start_main)
        else:
            self.start_main()

    def stop_main(self, after_cb=None):
        if not self.proc_main or self.proc_main.state() == QProcess.NotRunning:
            self._append_log("No running main.py")
            if after_cb: after_cb()
            return
        self._append_log("Stopping main.py …")
        self.proc_main.terminate()
        QTimer.singleShot(2000, self._kill_if_still_running)
        if after_cb:
            self.proc_main.finished.connect(lambda *_: after_cb())

    def _kill_if_still_running(self):
        if self.proc_main and self.proc_main.state() != QProcess.NotRunning:
            self._append_log("Force killing main.py")
            self.proc_main.kill()

    # ---------- Dashboard: preview ----------
    def _toggle_preview(self):
        if self.preview_worker and self.preview_worker.isRunning():
            self.preview_worker.stop()
            self.preview_worker.wait(500)
            self.preview_worker = None
            self.btn_preview.setText("Start Preview")
            return
        idx = int(self.cam_index_spin.value())
        self.preview_worker = VideoWorker(idx)
        self.preview_worker.frame_qimage.connect(self._on_preview_frame)
        self.preview_worker.opened.connect(lambda w: self._append_log("Preview camera opened" if w else "Preview: failed to open camera"))
        self.preview_worker.closed.connect(lambda: self._append_log("Preview camera closed"))
        self.preview_worker.start()
        self.btn_preview.setText("Stop Preview")

    def _on_preview_frame(self, qimg):
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _scan_cameras(self):
        self._append_log("Scanning camera indices 0..10")
        found = []
        for i in range(0, 11):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            ok = cap.isOpened()
            if ok: 
                found.append(i)
                cap.release()
        msg = "Found cameras: " + (", ".join(map(str, found)) if found else "none")
        self._append_log(msg)
        if found:
            self.cam_index_spin.setValue(found[0])

    # ---------- Homography tab ----------
    def _toggle_homo_cam(self):
        if self.homo_worker and self.homo_worker.isRunning():
            self.homo_worker.stop()
            self.homo_worker.wait(500)
            self.homo_worker = None
            self.btn_homo_start.setText("Start Camera")
            return
        idx = int(self.cam_index_spin.value())
        self.homo_worker = VideoWorker(idx)
        self.homo_worker.frame_qimage.connect(self._on_homo_frame)
        self.homo_worker.opened.connect(lambda w: self._append_log("Homography camera opened" if w else "Homography: failed to open camera"))
        self.homo_worker.start()
        self.btn_homo_start.setText("Stop Camera")
        self._homo_points_src = []
        self.homo_canvas.clear_points()

    def _on_homo_frame(self, qimg):
        # remember source size for point mapping
        self._homo_frame_size = (qimg.width(), qimg.height())
        self.homo_canvas.set_frame(qimg)

    def _homo_clear_points(self):
        self._homo_points_src = []
        self.homo_canvas.clear_points()

    def _homo_on_click(self, xw, yw):
        # map widget coords back to original frame coords
        if self._homo_frame_size is None or not self.homo_canvas._pix:
            return
        frame_w, frame_h = self._homo_frame_size
        # Find how the pixmap was drawn (same logic as in paintEvent)
        label_w, label_h = self.homo_canvas.width(), self.homo_canvas.height()
        pix_w = self.homo_canvas._pix.width()
        pix_h = self.homo_canvas._pix.height()
        # scaled size keeping aspect
        aspect = pix_w / pix_h
        if label_w / label_h > aspect:
            scaled_h = label_h
            scaled_w = int(aspect * scaled_h)
        else:
            scaled_w = label_w
            scaled_h = int(scaled_w / aspect)
        x0 = (label_w - scaled_w) // 2
        y0 = (label_h - scaled_h) // 2
        # click inside scaled rect?
        if xw < x0 or yw < y0 or xw > x0 + scaled_w or yw > y0 + scaled_h:
            return
        # normalize within scaled rect → back to frame coords
        nx = (xw - x0) / max(1, scaled_w)
        ny = (yw - y0) / max(1, scaled_h)
        fx = int(nx * frame_w)
        fy = int(ny * frame_h)
        if len(self._homo_points_src) < 4:
            self._homo_points_src.append((fx, fy))
            self._append_log(f"Homography point {len(self._homo_points_src)}: ({fx}, {fy})")

    def _homo_save(self):
        if len(self._homo_points_src) != 4:
            QMessageBox.warning(self, "Need 4 Points", "Please click 4 points on the image, in order.")
            return
        # Define a canonical destination square (e.g., 1000x500 "map")
        dst_w, dst_h = 1000, 500
        dst_pts = np.array([[0, 0], [dst_w-1, 0], [dst_w-1, dst_h-1], [0, dst_h-1]], dtype=np.float32)
        src_pts = np.array(self._homo_points_src, dtype=np.float32)
        H, mask = cv2.findHomography(src_pts, dst_pts, method=0)
        if H is None:
            QMessageBox.critical(self, "Homography Error", "Failed to compute homography.")
            return
        np.save(HOMOGRAPHY_NPY, H)
        self._append_log(f"Saved {HOMOGRAPHY_NPY} with points: {self._homo_points_src}")

    # ---------- Logs / status ----------
    def _read_main_stdout(self):
        if not self.proc_main: return
        data = bytes(self.proc_main.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_log(data.rstrip())

    def _main_exited(self, code, _status):
        self._append_log(f"main.py exited with code {code}")
        self._set_status("idle")
        self._update_buttons()
        self._stop_visual_timer()


    def _main_error(self, err):
        self._append_log(f"main.py process error: {err}")
        self._set_status("error")
        self._update_buttons()

    def _append_log(self, text):
        if not text: return
        self.log.append(text)
        self.log.moveCursor(self.log.textCursor().End)

    def _save_log_to_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Log", "session_log.txt", "Text Files (*.txt)")
        if not path: return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.log.toPlainText())
        self._append_log(f"Saved log to: {path}")

    def _update_buttons(self):
        running = bool(self.proc_main and self.proc_main.state() != QProcess.NotRunning)
        self.btn_start.setEnabled(not running if False else not running)  # avoid linter fold; keep explicit
        self.btn_restart.setEnabled(True)
        self.btn_stop.setEnabled(running)

    def _set_status(self, state: str):
        if state == "running":
            self.status_lbl.setText("Status: running")
            self.status_lbl.setObjectName("status_run")
        elif state == "error":
            self.status_lbl.setText("Status: error")
            self.status_lbl.setObjectName("status_err")
        else:
            self.status_lbl.setText("Status: idle")
            self.status_lbl.setObjectName("status_ok")
        self.status_lbl.style().unpolish(self.status_lbl); self.status_lbl.style().polish(self.status_lbl)

    def _start_visual_timer(self):
        self._left_mtime = 0.0
        self._right_mtime = 0.0
        self._visual_timer.start()
        self._append_log("Embedded visuals: started")

    def _stop_visual_timer(self):
        self._visual_timer.stop()
        self.left_img.setText("Waiting for frames…")
        self.right_img.setText("Waiting for frames…")
        self._append_log("Embedded visuals: stopped")


    def _load_pingpong(self, name, label, last_mtime_attr):
        """
        name: "left" or "right"
        label: QLabel to paint into
        last_mtime_attr: "_left_mtime" or "_right_mtime"
        """
        # read which slot is safe
        flag_path = os.path.join(self._live_dir, f"{name}.flag")
        try:
            with open(flag_path, "r", encoding="utf-8") as f:
                slot = f.read().strip()            # "a" or "b"
            img_path = os.path.join(self._live_dir, f"{name}_{slot}.jpg")
            if os.path.exists(img_path):
                mt = os.path.getmtime(img_path)
                if mt != getattr(self, last_mtime_attr):
                    setattr(self, last_mtime_attr, mt)
                    pm = QPixmap(img_path)
                    if not pm.isNull():
                        label.setPixmap(pm.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception:
            pass

    def _update_visuals(self):
        self._load_pingpong("left",  self.left_img,  "_left_mtime")
        self._load_pingpong("right", self.right_img, "_right_mtime")



if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SkatingControlPanel()
    win.resize(1100, 720)
    win.show()
    sys.exit(app.exec_())
