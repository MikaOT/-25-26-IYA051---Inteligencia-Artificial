import cv2
from detector_cartas import CardDetector, CardClassifier
from camara import Camara   # tu clase de cámara
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "..", "plantillas")


def run_realtime_camera(
        cam_source=1,
        templates_dir="../plantillas",
        debug=False):

    # Crear y arrancar cámara
    camera = Camara(source=cam_source, width=960, height=540)
    try:
        camera.start()
    except ValueError as e:
        print(e)
        return

    detector = CardDetector(debug=debug)
    classifier = CardClassifier(templates_dir=TEMPLATES_DIR, debug=True)


    print("Iniciando reconocimiento en tiempo real... (presiona Q o ESC para salir)")

    while True:
        frame = camera.get_frame()
        if frame is None:
            print("No se pudo leer frame de la cámara")
            break

        original = frame.copy()

        # --------- aquí va tu lógica de detección/clasificación ----------
        cards = detector.detect_cards(frame)

        for card_data in cards:
            warped = card_data["warped"]
            rank, suit = classifier.classify_card(warped)
            label = f"{rank} de {suit}"  # ej: "6 de diamantes"


            quad = card_data["quad"].astype(int)
            cv2.polylines(original, [quad], True, (0, 0, 255), 2)
            x, y = quad[0]
            cv2.putText(original, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)
        # -----------------------------------------------------------------

        cv2.imshow("Reconocimiento de Cartas - Tiempo Real", original)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:  # q o ESC
            break

    # --- SALIDA LIMPIA DEL PROGRAMA ---
    camera.release()
    cv2.destroyAllWindows()
    # “Hack” típico de Windows para asegurarse de que se cierren
    cv2.waitKey(1)
    cv2.waitKey(1)
    cv2.waitKey(1)

    print("Programa cerrado correctamente.")


if __name__ == "__main__":
    # Usa 1 si tu móvil/IVCam es la cámara virtual 1
    run_realtime_camera(
        cam_source=1,               # prueba con 0/1/2 según tu caso
        templates_dir="../plantillas",
        debug=False
    )
