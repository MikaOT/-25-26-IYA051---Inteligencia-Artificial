import cv2
import numpy as np
import os
from typing import Dict, List, Tuple


class DetectorPalos:
    """
    Detector de palos (♠ ♥ ♦ ♣) basado en:
      - ROI dinámico en la esquina superior izquierda de la carta.
      - Detección de color (rojo/negro) para descartar la mitad de los palos.
      - Forma geométrica + template matching como refinamiento.
    """

    def __init__(self, carpeta_plantillas: str = "plantillas/suits", lado_simbolo: int = 70):
        self.lado = lado_simbolo
        self.plantillas: Dict[str, List[np.ndarray]] = {}
        self._cargar_plantillas(carpeta_plantillas)

                # --- ESTABILIDAD TEMPORAL ---
        self.palo_estable = "?"          # último palo realmente "adoptado"
        self.palo_candidato = "?"        # palo que está intentando sustituir al estable
        self.frames_estables = 0         # frames seguidos del candidato
        self.MIN_FRAMES_ESTABLES = 4     # o 3–5, como prefieras

    # ---------------------------------------------------------------------
    # CARGA Y NORMALIZACIÓN DE PLANTILLAS
    # ---------------------------------------------------------------------
    def _cargar_plantillas(self, carpeta: str) -> None:
        """
        Carga todas las imágenes de palos de la carpeta dada.
        Soporta nombres en español o inglés mezclados:
          corazones / hearts, diamantes / diamonds,
          picas / spades, treboles / clubs.
        """
        if not os.path.isdir(carpeta):
            print(f"[DetectorPalos] Carpeta de plantillas no encontrada: {carpeta}")
            return

        for nombre_fichero in os.listdir(carpeta):
            if not nombre_fichero.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            ruta = os.path.join(carpeta, nombre_fichero)
            img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            base = os.path.splitext(nombre_fichero)[0].lower()

            # Normalizar nombre a {hearts, diamonds, spades, clubs}
            # Normalizar nombre a {hearts, diamonds, spades, clubs}
            if base.startswith(("corazon", "corazones", "heart", "hearts")):
                clave = "hearts"
            elif base.startswith(("diamante", "diamantes", "diamond", "diamonds", "oros", "oro")):
                clave = "diamonds"
            elif base.startswith(("pica", "picas", "spade", "spades")):
                clave = "spades"
            elif base.startswith(("trebol", "treboles", "club", "clubs", "basto", "bastos")):
                clave = "clubs"
            else:
                print(f"[DetectorPalos] Nombre de plantilla desconocido: {nombre_fichero}")
                continue
            

            # Redimensionar y binarizar de forma uniforme
            img_res = cv2.resize(img, (self.lado, self.lado), interpolation=cv2.INTER_AREA)
            _, binaria = cv2.threshold(img_res, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Pequeña limpieza
            kernel = np.ones((3, 3), np.uint8)
            binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
            binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=1)

            self.plantillas.setdefault(clave, []).append(binaria)

        print("[DetectorPalos] Plantillas cargadas:")
        for k, v in self.plantillas.items():
            print(f"  - {k}: {len(v)} variantes")

    # ---------------------------------------------------------------------
    # ROI DINÁMICO
    # ---------------------------------------------------------------------
    def _roi_simbolo(self, carta_bgr: np.ndarray) -> np.ndarray:
        """
        Extrae un ROI de la zona donde debería aparecer el palo pequeño.
        Mezcla porcentaje fijo con elección por contornos para adaptarse a
        pequeñas variaciones de posición.
        """
        h, w = carta_bgr.shape[:2]

        # Zona de búsqueda aproximada (esquina superior izquierda)
        zona_h = int(h * 0.55)
        zona_w = int(w * 0.40)
        zona = carta_bgr[0:zona_h, 0:zona_w]

        gris = cv2.cvtColor(zona, cv2.COLOR_BGR2GRAY)
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=1)

        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidatos: List[Tuple[float, Tuple[int, int, int, int]]] = []

        for c in contornos:
            area = cv2.contourArea(c)
            if area < 150 or area > 4000:
                continue

            x, y, cw, ch = cv2.boundingRect(c)
            aspecto = cw / float(ch) if ch > 0 else 0.0

            # Bonus por estar en una zona razonable y tamaño intermedio
            cx = x + cw / 2.0
            cy = y + ch / 2.0
            pos_x = cx / zona_w
            pos_y = cy / zona_h

            puntuacion = 0.0

            # Preferimos x pequeña (más pegado al borde izquierdo)
            if 0.05 < pos_x < 0.45:
                puntuacion += 1.0

            # Preferimos y más o menos en el tercio superior-medio
            if 0.20 < pos_y < 0.65:
                puntuacion += 1.0

            # Tamaño: ni muy pequeño ni enorme
            if 400 < area < 2000:
                puntuacion += 1.0

            # Aspect ratio razonable
            if 0.5 < aspecto < 2.0:
                puntuacion += 1.0

            if puntuacion > 0:
                candidatos.append((puntuacion, (x, y, cw, ch)))

        if not candidatos:
            # Fallback a ROI fijo por porcentaje
            ry1 = int(0.20 * h)
            ry2 = int(0.45 * h)
            rx1 = int(0.05 * w)
            rx2 = int(0.30 * w)
            return carta_bgr[ry1:ry2, rx1:rx2].copy()

        # Elegir el candidato con mayor puntuación
        candidatos.sort(key=lambda x: x[0], reverse=True)
        _, (x, y, cw, ch) = candidatos[0]

        # Ampliar un poco la caja
        margen = int(0.25 * max(cw, ch))
        x0 = max(0, x - margen)
        y0 = max(0, y - margen)
        x1 = min(zona_w, x + cw + margen)
        y1 = min(zona_h, y + ch + margen)

        return zona[y0:y1, x0:x1].copy()

    # ---------------------------------------------------------------------
    # COLOR ROJO / NEGRO
    # ---------------------------------------------------------------------
    def _es_rojo(self, roi_bgr: np.ndarray) -> bool:
        """
        Estima si el símbolo es rojo usando HSV + canales RGB.
        Versión más estricta para evitar falsos positivos en cartas negras.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return False

        # --- máscara de símbolo (lo que no es fondo blanco) ---
        gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, mask_symbol = cv2.threshold(
            gris, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        total_simbolo = cv2.countNonZero(mask_symbol)
        # Si casi no hay símbolo, consideramos que no hay rojo (ruido)
        if total_simbolo < 80:
            return False

        # --- rojo en HSV ---
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        rojo1 = cv2.inRange(hsv, (0, 60, 40), (10, 255, 255))
        rojo2 = cv2.inRange(hsv, (170, 60, 40), (180, 255, 255))
        mask_red_hsv = cv2.bitwise_or(rojo1, rojo2)
        mask_red_hsv = cv2.bitwise_and(mask_red_hsv, mask_symbol)

        pixeles_rojos_hsv = cv2.countNonZero(mask_red_hsv)
        ratio_hsv = pixeles_rojos_hsv / float(total_simbolo)

        # --- rojo en RGB (R claramente mayor que G y B) ---
        b, g, r = cv2.split(roi_bgr)
        mask_red_rgb = np.zeros_like(r, dtype=np.uint8)
        mask_red_rgb[(r > g + 15) & (r > b + 15) & (r > 90)] = 255
        mask_red_rgb = cv2.bitwise_and(mask_red_rgb, mask_symbol)

        pixeles_rojos_rgb = cv2.countNonZero(mask_red_rgb)
        ratio_rgb = pixeles_rojos_rgb / float(total_simbolo)

        # Medias solo sobre el símbolo
        mean_r = cv2.mean(r, mask_symbol)[0]
        mean_g = cv2.mean(g, mask_symbol)[0]
        mean_b = cv2.mean(b, mask_symbol)[0]

        # --- votos: pedimos al menos 2 de 3 para decir "rojo" ---
        votos = 0
        if ratio_hsv > 0.22:      # antes ~0.08 → ahora mucho más exigente
            votos += 1
        if ratio_rgb > 0.25:      # antes ~0.10
            votos += 1
        if (mean_r - max(mean_g, mean_b)) > 25:
            votos += 1

        return votos >= 2

    # ---------------------------------------------------------------------
    # PREPROCESADO DEL SÍMBOLO
    # ---------------------------------------------------------------------
    def _binarizar_simbolo(self, roi_bgr: np.ndarray) -> np.ndarray:
        gris = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=1)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Nos quedamos sólo con el mayor contorno
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contornos:
            mascara = np.zeros_like(binaria)
            c = max(contornos, key=cv2.contourArea)
            cv2.drawContours(mascara, [c], -1, 255, -1)
            binaria = mascara

        # Redimensionar manteniendo proporción dentro de un canvas cuadrado
        h, w = binaria.shape
        if h == 0 or w == 0:
            return np.zeros((self.lado, self.lado), dtype=np.uint8)

        escala = self.lado / max(h, w)
        nh = max(1, int(h * escala))
        nw = max(1, int(w * escala))
        redim = cv2.resize(binaria, (nw, nh), interpolation=cv2.INTER_AREA)

        lienzo = np.zeros((self.lado, self.lado), dtype=np.uint8)
        oy = (self.lado - nh) // 2
        ox = (self.lado - nw) // 2
        lienzo[oy:oy + nh, ox:ox + nw] = redim

        return lienzo

    # ---------------------------------------------------------------------
    # DESCRIPTORES DE FORMA
    # ---------------------------------------------------------------------
    def _medidas_forma(self, binaria: np.ndarray) -> dict:
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            return {"area": 0, "solidez": 0, "aspecto": 0, "defectos": 0}

        cnt = max(contornos, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area == 0:
            return {"area": 0, "solidez": 0, "aspecto": 0, "defectos": 0}

        x, y, w, h = cv2.boundingRect(cnt)
        hull = cv2.convexHull(cnt)
        area_hull = cv2.contourArea(hull) or 1.0
        solidez = area / area_hull
        aspecto = w / float(h) if h > 0 else 0.0

        # Defectos de convexidad (cuantos más, más "trébol")
        hull_idx = cv2.convexHull(cnt, returnPoints=False)
        try:
            defects = cv2.convexityDefects(cnt, hull_idx)
            n_def = 0 if defects is None else defects.shape[0]
        except cv2.error:
            n_def = 0

        return {
            "area": area,
            "solidez": solidez,
            "aspecto": aspecto,
            "defectos": n_def,
        }


    def _actualizar_estabilidad(self, palo_instantaneo: str, score_instantaneo: float) -> Tuple[str, float]:
        """
        Histéresis temporal:
        - Usamos palo_candidato para contar cuántos frames seguidos
          llevamos viendo el mismo palo_instantaneo.
        - Solo cuando supera MIN_FRAMES_ESTABLES lo aceptamos como
          nuevo palo_estable.
        - Mientras tanto:
            * Al principio, antes de tener palo_estable válido → devolvemos "?"
            * Una vez fijado un palo_estable → seguimos devolviéndolo
              hasta que el nuevo candidato lleve N frames seguidos.
        """

        # 1) Actualizar candidato + contador
        if palo_instantaneo == self.palo_candidato:
            self.frames_estables += 1
        else:
            self.palo_candidato = palo_instantaneo
            self.frames_estables = 1

        # 2) ¿El candidato ya es suficientemente estable?
        if self.frames_estables >= self.MIN_FRAMES_ESTABLES:
            self.palo_estable = self.palo_candidato

        # 3) Decidir qué devolvemos
        if self.palo_estable == "?":
            # Fase de arranque: aún no hemos aceptado ningún palo
            return "?", score_instantaneo
        else:
            # Ya tenemos palo estable: lo mantenemos aunque palo_instantaneo
            # fluctúe, hasta que el nuevo candidato aguante N frames.
            return self.palo_estable, score_instantaneo


    def _puntuacion_forma(self, medidas: dict, palo: str) -> float:
        """
        Heurísticas muy simples por palo. No hace falta que sean perfectas;
        el template matching se encarga de afinar.
        """
        s = medidas["solidez"]
        a = medidas["aspecto"]
        d = medidas["defectos"]

        puntuacion = 0.0

        if palo == "diamonds":
            # Diamante: convexo y algo "afilado", pero sin dar un boost enorme.
            # Bajamos el peso de la forma para que no arrase a corazones.
            if 0.85 < s < 1.02:
                puntuacion += 0.4       # antes 0.6
            if 0.65 < a < 1.35:
                puntuacion += 0.2       # antes 0.4

        elif palo == "hearts":
            # Corazón: forma más redondeada y rellena, permitimos más variación.
            if s > 0.75:
                puntuacion += 0.5       # subimos un poco
            if 0.75 < a < 1.6:
                puntuacion += 0.3       # rango de aspecto más amplio
            if d >= 1:
                puntuacion += 0.2       # antes 0.3, menos dependencia de defectos


        elif palo == "spades":
            # Bastante convexo, pocos defectos
            if 0.82 < s < 1.02:
                puntuacion += 0.5
            if d <= 5:
                puntuacion += 0.3
            if 0.6 < a < 1.3:
                puntuacion += 0.2

        elif palo == "clubs":
            # Varios lóbulos: menos solidez y bastantes defectos
            # Lo hacemos más estricto para no confundir picas gruesas con tréboles
            if s < 0.82:          # antes 0.85
                puntuacion += 0.5
            if d >= 6:            # antes 4
                puntuacion += 0.4
            elif d >= 4:
                puntuacion += 0.2
            if 0.7 < a < 1.4:
                puntuacion += 0.1


        # Normalizar a [0, 1] (clipeamos por si acaso)
        return max(0.0, min(1.0, puntuacion))


    def _parece_diamante(self, medidas: dict) -> bool:
        s = medidas["solidez"]
        a = medidas["aspecto"]
        d = medidas["defectos"]

        # Diamante real de tu baraja = convexo pero no perfecto,
        # aspecto ligeramente vertical, muy pocos defectos.
        return (0.80 < s < 1.05) and (0.55 < a < 1.50) and (d <= 2)


    # ---------------------------------------------------------------------
    # TEMPLATE MATCHING SIMPLE
    # ---------------------------------------------------------------------
    def _score_plantilla(self, binaria: np.ndarray, plantilla: np.ndarray) -> float:
        """
        Compara el símbolo con una plantilla (mismo tamaño).
        Usa dos métricas de OpenCV y hace una media.
        """
        if binaria.shape != plantilla.shape:
            binaria = cv2.resize(binaria, (plantilla.shape[1], plantilla.shape[0]), interpolation=cv2.INTER_AREA)

        res1 = cv2.matchTemplate(binaria, plantilla, cv2.TM_CCOEFF_NORMED)
        s1 = float(res1.max())

        res2 = cv2.matchTemplate(binaria, plantilla, cv2.TM_CCORR_NORMED)
        s2 = float(res2.max())

        return (s1 + s2) * 0.5

    def _puntuaciones_por_palo(self, binaria: np.ndarray, candidatos: List[str]) -> dict:
        """
        Combina forma + plantillas para cada palo candidato.

        Ajuste:
        - Palos negros (spades/clubs): plantilla pesa más, pero
          seguimos usando algo de forma.
        - Palos rojos (hearts/diamonds): plantilla manda casi siempre,
          porque visualmente se confunden menos y las heurísticas de forma
          tienden a favorecer diamantes.
        """
        medidas = self._medidas_forma(binaria)
        resultados = {}

        for palo in candidatos:
            score_forma = self._puntuacion_forma(medidas, palo)

            # Mejor plantilla de ese palo
            score_tpl = 0.0
            for tpl in self.plantillas.get(palo, []):
                score_tpl = max(score_tpl, self._score_plantilla(binaria, tpl))

            # Pesos distintos según palo / color
            if palo in ("spades", "clubs"):
                # Palos negros: forma + plantilla
                w_forma = 0.35
                w_tpl   = 0.65
            elif palo in ("hearts", "diamonds"):
                # Palos rojos: que decida SOLO la plantilla
                w_forma = 0.0
                w_tpl   = 1.0


            score_final = w_forma * score_forma + w_tpl * score_tpl
            resultados[palo] = score_final

        return resultados

    # ---------------------------------------------------------------------
    # API PÚBLICA
    # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
    # API PÚBLICA
    # ---------------------------------------------------------------------
    def detectar_palo(self, carta_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Recibe una carta warpeada (BGR) y devuelve:
            (nombre_palo, confianza)
        donde nombre_palo ∈ {'hearts', 'diamonds', 'spades', 'clubs'}
        """
        if carta_bgr is None or carta_bgr.size == 0:
            return "?", 0.0

        roi = self._roi_simbolo(carta_bgr)
        if roi.size == 0:
            return "?", 0.0

        rojo = self._es_rojo(roi)
        binaria = self._binarizar_simbolo(roi)

        # Filtrar palos por color
        if rojo:
            candidatos = ["hearts", "diamonds"]
        else:
            candidatos = ["spades", "clubs"]

        if not candidatos:
            return "?", 0.0

        # (Si quisieras usar medidas de forma extra, ya las tienes aquí)
        medidas = self._medidas_forma(binaria)

        # Probamos varias rotaciones del símbolo para ser robustos a giros
        mejores_scores = {p: 0.0 for p in candidatos}

        for ang in (0, 90, 180, 270):
            if ang == 0:
                rot = binaria
            else:
                M = cv2.getRotationMatrix2D((self.lado / 2, self.lado / 2), ang, 1.0)
                rot = cv2.warpAffine(binaria, M, (self.lado, self.lado))

            scores = self._puntuaciones_por_palo(rot, candidatos)
            for p, s in scores.items():
                mejores_scores[p] = max(mejores_scores[p], s)

        # --------------------------------------------------------------
        # DECISIÓN FINAL (hearts vs diamonds con histéresis temporal)
        # --------------------------------------------------------------
        if rojo and "hearts" in mejores_scores and "diamonds" in mejores_scores:
            s_hearts   = mejores_scores["hearts"]
            s_diamonds = mejores_scores["diamonds"]

            diff     = s_hearts - s_diamonds
            abs_diff = abs(diff)
            print(f"Hearts={s_hearts:.3f}, Diamonds={s_diamonds:.3f}, "
                  f"diff={diff:.3f}, abs_diff={abs_diff:.3f}")

            # Simple: gana el que tenga más score
            palo_instantaneo = "hearts" if s_hearts >= s_diamonds else "diamonds"
        else:
            # Palos negros o solo un candidato
            palo_instantaneo = max(mejores_scores, key=mejores_scores.get)

        score_instantaneo = mejores_scores.get(palo_instantaneo, 0.0)

        palo_final, score_final = self._actualizar_estabilidad(
            palo_instantaneo, score_instantaneo
        )

        return palo_final, score_final


