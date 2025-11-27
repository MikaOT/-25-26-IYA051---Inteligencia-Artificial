import cv2
import numpy as np
import os
from glob import glob
from detector_palos import DetectorPalos

detector_palos = DetectorPalos()  # se carga una vez


# ---------------------------
# Utilidades generales
# ---------------------------

def order_points(pts):
    """
    Ordena los 4 puntos de un cuadrilátero en el orden:
    [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def four_point_transform(image, pts, width=250, height=400):
    """
    Aplica una transformación de perspectiva para obtener una carta
    enderezada con tamaño fijo (width x height).
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


# ---------------------------
# Detector de cartas
# ---------------------------

class CardDetector:
    def __init__(self,
                 min_area=10000,
                 aspect_ratio_range=(0.5, 0.8),
                 debug=False):
        self.min_area = min_area
        self.aspect_ratio_range = aspect_ratio_range
        self.debug = debug

    def segment_non_green(self, image_bgr):
        """
        Segmenta el tapete verde en espacio HSV y devuelve una máscara
        de 'no verde' (probables cartas u otros objetos).
        """
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Rango aproximado para verde (ajustar según tapete)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        mask_green = cv2.inRange(hsv, lower_green, upper_green)
        mask_non_green = cv2.bitwise_not(mask_green)

        # Operaciones morfológicas para limpiar
        kernel = np.ones((5, 5), np.uint8)
        mask_non_green = cv2.morphologyEx(mask_non_green,
                                          cv2.MORPH_OPEN, kernel)
        mask_non_green = cv2.morphologyEx(mask_non_green,
                                          cv2.MORPH_CLOSE, kernel)

        if self.debug:
            cv2.imshow("Mask non green", mask_non_green)
            cv2.waitKey(1)

        return mask_non_green

    def detect_cards(self, image_bgr):
        """
        Devuelve una lista de cartas detectadas.
        Cada elemento: dict con:
            - 'contour': contorno original
            - 'quad': np.array con 4 puntos (x, y)
            - 'warped': imagen de la carta enderezada
        """
        mask = self.segment_non_green(image_bgr)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        cards = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            # Aproximar polígono
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) != 4:
                continue

            pts = approx.reshape((4, 2)).astype("float32")

            # Comprobar relación de aspecto aproximada
            rect = order_points(pts)
            (tl, tr, br, bl) = rect
            width = np.linalg.norm(tr - tl)
            height = np.linalg.norm(bl - tl)
            if height == 0:
                continue

            aspect_ratio = width / height
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue

            warped = four_point_transform(image_bgr, pts,
                                          width=250, height=400)

            cards.append({
                "contour": cnt,
                "quad": pts,
                "warped": warped
            })

        return cards


# ---------------------------
# Clasificador de cartas
# ---------------------------

class CardClassifier:
    """
    - Rank: clasificador k-NN sobre símbolos binarizados y normalizados.
    - Palo: DetectorPalos + plantillas clásicas como fallback.
    """
    MIN_SCORE_SUIT = 0.25  # para el fallback de palos

    def __init__(self,
                 templates_dir="plantillas",
                 roi_rank=(15, 15, 80, 110),
                 roi_suit=(15, 110, 80, 190),
                 debug=False):

        self.debug = debug

        # Para k-NN de ranks
        self.rank_vecs = None        # np.ndarray (N, D)
        self.rank_labels = []        # lista de strings (N)
        self.rank_img_size = (32, 48)  # (ancho, alto) símbolo normalizado

        # Para templates de palos
        self.templates_suits = {}    # dict[str, list[np.ndarray]]

        self.roi_rank = roi_rank
        self.roi_suit = roi_suit

        self.load_templates(templates_dir)

    # ------------ COLOR ROJO/NEGRO ------------
    def detect_color(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        # Rojo ocupa dos zonas en H
        lower_red1 = np.array([0, 60, 60])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 60, 60])
        upper_red2 = np.array([179, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask1, mask2)

        red_pixels = cv2.countNonZero(mask_red)
        total_pixels = roi_bgr.shape[0] * roi_bgr.shape[1]

        # si más del 10% de los píxeles son rojos → carta roja
        if total_pixels > 0 and red_pixels / total_pixels > 0.10:
            return "rojo"
        else:
            return "negro"

    # ------------ PREPROCESADO COMÚN (ROI + PLANTILLA) ------------
    def _binarize_and_normalize(self, gray):
        """Binariza y asegura símbolo BLANCO sobre fondo NEGRO."""
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        white = np.sum(bw == 255)
        black = np.sum(bw == 0)

        # Si hay más blanco que negro => el fondo es blanco y el símbolo negro.
        # Invertimos para dejar símbolo BLANCO sobre fondo NEGRO.
        if white > black:
            bw = cv2.bitwise_not(bw)

        return bw

    # ------------ RECORTAR SÍMBOLO PRINCIPAL ------------
    def _crop_symbol(self, bw):
        """
        Recorta el símbolo principal (región blanca más grande) de una imagen binaria.
        bw: imagen binaria, símbolo BLANCO (255) sobre fondo NEGRO (0).
        """
        img = bw.copy().astype(np.uint8)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return bw  # fallback

        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)

        pad = 2
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, bw.shape[1])
        y2 = min(y + h + pad, bw.shape[0])

        cropped = bw[y1:y2, x1:x2]
        return cropped

    # ------------ PLANTILLAS ------------
    def load_templates(self, templates_dir):
        # Ranks (ya recortados/buenos) en 'ranks_limpios'
        ranks_dir = os.path.join(templates_dir, "ranks_limpios")
        suits_dir = os.path.join(templates_dir, "suits_limpios")

        # ---- RANKS: preparamos vectores para k-NN ----
        self.rank_vecs = []
        self.rank_labels = []

        for path in glob(os.path.join(ranks_dir, "*.*")):
            name = os.path.splitext(os.path.basename(path))[0]  # ej. "2_1"
            label = name.split("_")[0].upper()                  # "2"

            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img_bin = self._binarize_and_normalize(img)
            cropped = self._crop_symbol(img_bin)

            w_norm, h_norm = self.rank_img_size
            norm = cv2.resize(cropped, (w_norm, h_norm), interpolation=cv2.INTER_NEAREST)

            vec = (norm.astype(np.float32) / 255.0).flatten()

            self.rank_vecs.append(vec)
            self.rank_labels.append(label)

        if self.rank_vecs:
            self.rank_vecs = np.stack(self.rank_vecs, axis=0)  # (N, D)
        else:
            self.rank_vecs = np.zeros(
                (0, self.rank_img_size[0] * self.rank_img_size[1]),
                dtype=np.float32
            )

        # ---- SUITS: listas de plantillas por palo ----
        self.templates_suits = {}
        for path in glob(os.path.join(suits_dir, "*.*")):
            name = os.path.splitext(os.path.basename(path))[0].lower()
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img_bin = self._binarize_and_normalize(img)

            if name not in self.templates_suits:
                self.templates_suits[name] = []
            self.templates_suits[name].append(img_bin)

        print("RANK templates:", {lab: self.rank_labels.count(lab) for lab in set(self.rank_labels)})
        print("SUIT templates:", {k: len(v) for k, v in self.templates_suits.items()})

    # ------------ PREPROCESADO ROI ------------
    def preprocess_roi(self, roi_bgr):
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        bw = self._binarize_and_normalize(gray)
        return bw

    # ------------ MATCH TEMPLATES (para palos) ------------
    def match_template_set(self, roi_bin, templates, min_score):
        """
        templates: dict[str, list[np.ndarray]]
        Para palos.
        """
        if not templates:
            return "?", 0.0

        h, w = roi_bin.shape[:2]

        best_name = "?"
        best_score = -1.0
        scores_debug = []

        for name, tmpl_list in templates.items():
            for tmpl in tmpl_list:
                if tmpl.ndim == 3:
                    tmpl_gray = cv2.cvtColor(tmpl, cv2.COLOR_BGR2GRAY)
                else:
                    tmpl_gray = tmpl

                tmpl_resized = cv2.resize(tmpl_gray, (w, h))
                res = cv2.matchTemplate(roi_bin, tmpl_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)

                scores_debug.append((name, float(max_val)))

                if max_val > best_score:
                    best_score = max_val
                    best_name = name

        if self.debug:
            scores_debug.sort(key=lambda x: x[1], reverse=True)
            print("TOP scores (suit):", scores_debug[:5])

        if best_score < min_score:
            return "?", best_score
        else:
            return best_name, best_score

    # ------------ CLASIFICADOR k-NN PARA RANK ------------
    def classify_rank_knn(self, roi_bgr, k=3):
        """
        Clasifica el rank usando k-NN sobre imágenes binarias normalizadas.
        Devuelve (label, score_estimada).
        """
        if self.rank_vecs is None or len(self.rank_labels) == 0:
            return "?", 0.0

        bw = self.preprocess_roi(roi_bgr)
        cropped = self._crop_symbol(bw)

        w_norm, h_norm = self.rank_img_size
        norm = cv2.resize(cropped, (w_norm, h_norm), interpolation=cv2.INTER_NEAREST)
        vec = (norm.astype(np.float32) / 255.0).flatten()

        diffs = self.rank_vecs - vec[None, :]    # (N, D)
        dists = np.linalg.norm(diffs, axis=1)    # (N,)

        k = min(k, len(dists))
        idx_sorted = np.argsort(dists)[:k]

        votes = {}
        for idx in idx_sorted:
            lab = self.rank_labels[idx]
            votes[lab] = votes.get(lab, 0) + 1

        best_label = max(votes.items(), key=lambda x: x[1])[0]

        d0 = dists[idx_sorted[0]]
        score = 1.0 / (1.0 + d0)  # 0 dist -> ~1; grande -> pequeño

        if self.debug:
            neigh_info = [(self.rank_labels[i], float(dists[i])) for i in idx_sorted]
            print("RANK k-NN vecinos:", neigh_info)
            print(f"RANK elegido: {best_label} (score ~ {score:.3f})")

        return best_label, score

    # ------------ CLASIFICACIÓN EN UNA SOLA ORIENTACIÓN ------------
    def _classify_card_single(self, warped_card):
        """
        Clasifica una carta warpeada en una sola orientación.
        Devuelve (rank, suit_final, score_rank).
        """
        h, w, _ = warped_card.shape

        # ======== ROIS EN PORCENTAJE (AJUSTABLES) ========
        # Rank (número/letra) esquina sup. izquierda
        rx1 = int(0.02 * w)
        ry1 = int(0.05 * h)
        rx2 = int(0.24 * w)
        ry2 = int(0.22 * h)

        # Palo pequeño (solo lo usamos para debug / fallback)
        sx1 = int(0.02 * w)
        sy1 = int(0.23 * h)
        sx2 = int(0.22 * w)
        sy2 = int(0.36 * h)

        # Región pequeña para estimar color (rojo/negro)
        cx1 = int(0.05 * w)
        cy1 = int(0.05 * h)
        cx2 = int(0.18 * w)
        cy2 = int(0.18 * h)

        roi_rank  = warped_card[ry1:ry2, rx1:rx2]
        roi_suit  = warped_card[sy1:sy2, sx1:sx2]
        roi_color = warped_card[cy1:cy2, cx1:cx2]

        if self.debug:
            cv2.imshow("ROI_rank", roi_rank)
            cv2.imshow("ROI_suit_fallback", roi_suit)
            cv2.imshow("ROI_color", roi_color)
            cv2.waitKey(1)

        # --------- RANK (NÚMERO/LETRA) con k-NN ---------
        rank, score_rank = self.classify_rank_knn(roi_rank, k=3)

        # --------- PALO (USANDO DetectorPalos) ---------
                # --------- PALO (USANDO DetectorPalos) ---------
        palo_en, score_palo = detector_palos.detectar_palo(warped_card)

        # Estimar color SIEMPRE
        color_palo = self.detect_color(roi_color)  # "rojo" o "negro"

        mapa_palos_es = {
            "hearts": "corazones",
            "diamonds": "diamantes",
            "spades": "picas",
            "clubs": "treboles"
        }

        suit_final = mapa_palos_es.get(palo_en, "?")
        score_suit_final = score_palo

        # --- CORRECCIÓN DE INCOHERENCIA ROJO/NEGRO ---
        # Si DetectorPalos dice un palo ROJO pero detect_color dice NEGRO (o al revés),
        # consideramos que hay conflicto y lanzamos el fallback con plantillas.
        es_rojo_detector = palo_en in ["hearts", "diamonds"]
        es_rojo_color    = (color_palo == "rojo")

        hay_conflicto_color = (
            palo_en != "?" and (es_rojo_detector != es_rojo_color)
        )

        # Si el score es bajo O hay conflicto de color -> usamos fallback
        if score_palo < 0.35 or hay_conflicto_color:
            if color_palo == "rojo":
                templates_suits = {
                    k: v for k, v in self.templates_suits.items()
                    if k.startswith("corazones") or k.startswith("diamantes")
                }
            else:
                templates_suits = {
                    k: v for k, v in self.templates_suits.items()
                    if k.startswith("picas") or k.startswith("treboles")
                }

            if not templates_suits:
                templates_suits = self.templates_suits

            suit_bin = self.preprocess_roi(roi_suit)
            suit_tpl, score_suit_tpl = self.match_template_set(
                suit_bin,
                templates_suits,
                self.MIN_SCORE_SUIT
            )

            if score_suit_tpl > score_palo + 0.05 or hay_conflicto_color:
                suit_final = self._clean_suit_name(suit_tpl)
                score_suit_final = score_suit_tpl

        if self.debug:
            print(f"[SINGLE] Rank: {rank} (score≈{score_rank:.3f}), "
                  f"Palo (DetectorPalos): {palo_en} (score={score_palo:.3f}), "
                  f"Palo final: {suit_final} (score={score_suit_final:.3f})")

        return rank, suit_final, score_rank
    
    def _clean_suit_name(self, name):
        """
        Quita sufijos como '_1', '_2', '_3' de nombres de plantillas.
        Ej: 'corazones_2' -> 'corazones'
        """
        if "_" in name:
            base = name.split("_")[0]
            return base
        return name


    # Versión sencilla sin rotaciones (por si la quieres usar en algún test)
    def classify_card(self, warped_card):
        rank, suit, _ = self._classify_card_single(warped_card)
        return rank, suit

    # ------------ CLASIFICACIÓN PROBANDO ROTACIONES ------------
    def classify_card_all_rotations(self, warped_card):
        """
        Prueba las rotaciones 0°, 90°, 180°, 270° y devuelve
        (rank, suit, mejor_angulo, carta_rotada_usada).
        Elige la orientación con mayor score del rank.
        """
        best_rank = "?"
        best_suit = "?"
        best_score = -1.0
        best_rot = 0
        best_card = warped_card

        rotations = {
            0:   warped_card,
            90:  cv2.rotate(warped_card, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(warped_card, cv2.ROTATE_180),
            270: cv2.rotate(warped_card, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }

        for angle, card_rot in rotations.items():
            rank, suit, score_rank = self._classify_card_single(card_rot)

            if score_rank > best_score:
                best_score = score_rank
                best_rank = rank
                best_suit = suit
                best_rot = angle
                best_card = card_rot

        # si el score es muy bajo, preferimos decir "?"
        if best_score < 0.03:
            best_rank = "?"

        if self.debug:
            print(f"[MEJOR ROT] {best_rot}° -> {best_rank} de {best_suit} "
                  f"(score≈{best_score:.3f})")

        return best_rank, best_suit, best_rot, best_card


# ---------------------------
# Función principal para imagen fija
# ---------------------------

def process_image(path_image,
                  templates_dir="plantillas",
                  show=True,
                  save_output=False,
                  output_path="resultado.png",
                  debug=False):
    image = cv2.imread(path_image)
    if image is None:
        print(f"Error al cargar imagen: {path_image}")
        return

    detector = CardDetector(debug=debug)
    classifier = CardClassifier(templates_dir=templates_dir, debug=debug)

    cards = detector.detect_cards(image)
    print(f"Cartas detectadas: {len(cards)}")

    output = image.copy()

    for card_data in cards:
        warped = card_data["warped"]

        # <<< IMPORTANTE: usamos la versión con rotaciones >>>
        rank, suit, rot, used_card = classifier.classify_card_all_rotations(warped)
        label = f"{rank} de {suit}"

        quad = card_data["quad"].astype(int)
        cv2.polylines(output, [quad], True, (0, 0, 255), 2)

        x, y = quad[0]
        cv2.putText(output, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2, cv2.LINE_AA)

    if save_output:
        cv2.imwrite(output_path, output)

    if show:
        cv2.imshow("Resultado", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return output


if __name__ == "__main__":
    imagen_prueba = "../imagenes/prueba1.jpg"
    process_image(
        imagen_prueba,
        templates_dir="../plantillas",
        show=True,
        save_output=True,
        output_path="../resultado_prueba1.png",
        debug=True
    )
