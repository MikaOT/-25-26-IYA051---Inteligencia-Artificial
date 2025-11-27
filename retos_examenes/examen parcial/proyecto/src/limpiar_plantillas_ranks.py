import os
import cv2
import numpy as np
from glob import glob

# === CONFIGURACIÓN ===
# Carpeta donde tienes ahora las plantillas "crudas"
INPUT_DIR = "plantillas/suits"        # cámbialo si hace falta
# Carpeta donde se guardarán las plantillas limpias
OUTPUT_DIR = "plantillas/suits_limpios"

# Tamaño final de las plantillas (ancho, alto)
TEMPLATE_SIZE = (80, 120)  # ajustable, pero no lo hagas muy pequeño


def binarize_and_normalize(gray):
    """
    Binariza y deja el símbolo en BLANCO y el fondo en NEGRO.
    """
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    white = np.sum(bw == 255)
    black = np.sum(bw == 0)

    # Si hay más blanco que negro, asumimos fondo blanco y símbolo oscuro → invertimos
    if white > black:
        bw = cv2.bitwise_not(bw)

    return bw


def keep_biggest_component(bw):
    """
    Se queda solo con el componente conectado más grande (el número/letra).
    bw: imagen binaria (0 fondo, 255 símbolo).
    """
    # Aseguramos tipo uint8
    bw_u8 = bw.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_u8, connectivity=8)

    if num_labels <= 1:
        # Solo fondo o nada
        return bw

    # stats: [label, 5] -> [x, y, w, h, area]
    # Ignoramos el label 0 (fondo), buscamos el de mayor área
    areas = stats[1:, 4]
    max_idx = 1 + np.argmax(areas)

    mask = np.zeros_like(bw_u8)
    mask[labels == max_idx] = 255

    return mask


def crop_to_symbol(bw):
    """
    Recorta la imagen binaria al bounding box del símbolo (píxeles blancos).
    """
    ys, xs = np.where(bw == 255)
    if len(xs) == 0 or len(ys) == 0:
        return bw  # nada que recortar

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cropped = bw[y_min:y_max + 1, x_min:x_max + 1]
    return cropped


def add_margin(img, margin_ratio=0.2):
    """
    Añade un margen negro alrededor de la imagen.
    margin_ratio es porcentaje respecto al tamaño máximo (0.2 = 20%).
    """
    h, w = img.shape[:2]
    m = int(max(w, h) * margin_ratio)

    # Añadimos borde negro: top, bottom, left, right
    bordered = cv2.copyMakeBorder(
        img, m, m, m, m,
        borderType=cv2.BORDER_CONSTANT,
        value=0  # negro
    )
    return bordered


def process_template(path_in, path_out):
    """
    Limpia una plantilla individual y la guarda en path_out.
    """
    img = cv2.imread(path_in, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] No se pudo leer: {path_in}")
        return

    # 1) Binarizar y normalizar (símbolo blanco, fondo negro)
    bw = binarize_and_normalize(img)

    # 2) Quedarse solo con el componente grande (número/letra)
    bw = keep_biggest_component(bw)

    # 3) Recortar al bounding box del símbolo
    bw = crop_to_symbol(bw)

    # 4) Añadir margen
    bw = add_margin(bw, margin_ratio=0.2)

    # 5) Redimensionar a tamaño fijo
    bw_resized = cv2.resize(bw, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)

    # 6) Guardar como PNG
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    cv2.imwrite(path_out, bw_resized)
    print(f"[OK] {os.path.basename(path_in)} -> {path_out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Aceptamos cualquier extensión típica de imagen
    patrones = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    paths = []
    for p in patrones:
        paths.extend(glob(os.path.join(INPUT_DIR, p)))

    if not paths:
        print(f"No se encontraron plantillas en {INPUT_DIR}")
        return

    print(f"Encontradas {len(paths)} plantillas. Procesando...\n")

    for path in paths:
        filename = os.path.basename(path)
        name_no_ext, _ = os.path.splitext(filename)

        # Mantenemos el mismo nombre, pero en la carpeta de salida, en PNG
        out_path = os.path.join(OUTPUT_DIR, name_no_ext + ".png")
        process_template(path, out_path)

    print("\nListo. Ahora puedes usar 'plantillas/ranks_limpios' como carpeta de ranks.")


if __name__ == "__main__":
    main()
