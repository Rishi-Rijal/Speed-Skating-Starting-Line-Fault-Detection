# video_worker.py
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from PyQt5.QtGui import QImage

class VideoWorker(QThread):
    frame_qimage = pyqtSignal(QImage)
    opened = pyqtSignal(int)      # emits width if opened, else 0
    closed = pyqtSignal()

    def __init__(self, cam_index=0, parent=None):
        super().__init__(parent)
        self._cam_index = cam_index
        self._running = False
        self._mutex = QMutex()
        self._cap = None

    def set_cam_index(self, idx: int):
        with QMutexLocker(self._mutex):
            self._cam_index = idx

    def run(self):
        with QMutexLocker(self._mutex):
            cam = self._cam_index
            self._running = True

        cap = cv2.VideoCapture(cam, cv2.CAP_DSHOW)
        if not cap or not cap.isOpened():
            self.opened.emit(0)
            self._emit_black_frame("Camera not opened")
            self._running = False
            self.closed.emit()
            return

        self.opened.emit(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

        self._cap = cap
        while True:
            with QMutexLocker(self._mutex):
                if not self._running:
                    break
            ok, frame = cap.read()
            if not ok or frame is None:
                self._emit_black_frame("No frame")
                continue
            # BGR -> RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            self.frame_qimage.emit(qi)
        cap.release()
        self.closed.emit()

    def stop(self):
        with QMutexLocker(self._mutex):
            self._running = False

    def _emit_black_frame(self, text=""):
        # 480p black with tiny text fallback
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        if text:
            cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qi = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        self.frame_qimage.emit(qi)
