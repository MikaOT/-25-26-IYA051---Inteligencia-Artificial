import cv2

class Camara:
    def __init__(self, source=1, width=960, height=540):
        """
        source: índice de cámara (IVCam / móvil suele ser 1)
        """
        self.source = source
        self.cap = None
        self.width = width
        self.height = height

    def start(self):
        # Usamos DirectShow para evitar problemas con backends
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise ValueError(f"No se puede abrir la cámara {self.source}")
        # Bajar resolución para ir fluido
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def get_frame(self):
        if self.cap:
            ok, frame = self.cap.read()
            if ok:
                return frame
        return None

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None
