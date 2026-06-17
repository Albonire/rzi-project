"""
capa 1 - visor de imagen con zoom, pan y seleccion de region.
"""

import numpy as np
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PySide6.QtCore   import Qt, QRectF, Signal
from PySide6.QtGui    import (QPixmap, QImage, QPainter, QColor,
                               QPen, QWheelEvent)


class VisorImagen(QGraphicsView):
    """
    Visor interactivo con zoom (rueda), pan (arrastrar) y
    modo de seleccion rectangular activable.

    la escena tiene el mismo tamano en pixeles que la imagen mostrada,
    por lo que coordenadas de escena = coordenadas de imagen.
    """

    # emite (x, y, w, h) en coords de imagen cuando el usuario suelta el rectangulo
    region_seleccionada = Signal(int, int, int, int)
    mouse_moved = Signal(int, int)  # Emite (x, y) en coordenadas de imagen

    def __init__(self, parent=None):
        super().__init__(parent)

        self._escena      = QGraphicsScene(self)
        self._pixmap_item = None
        self._zoom        = 1.0
        self._modo_sel    = False
        self._sel_inicio  = None
        self._sel_rect    = None   # QGraphicsRectItem visible mientras se dibuja

        self._marcas      = []     # lista de QGraphicsRectItem para region e inset

        self.setMouseTracking(True)
        self.setScene(self._escena)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    # mostrar imagen

    def establecer_imagen(self, imagen: np.ndarray) -> None:
        """recibe ndarray float64 [0,1] o uint8 y lo muestra."""
        if imagen is None:
            return

        img = np.clip(imagen, 0.0, 1.0)
        img_u8 = (img * 255).astype(np.uint8)

        if img_u8.ndim == 2:
            h, w = img_u8.shape
            # QImage necesita bytes contiguos
            qimg = QImage(img_u8.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, _ = img_u8.shape
            qimg = QImage(img_u8.data, w, h, w * 3, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg.copy())
        self._escena.clear()
        self._sel_rect    = None
        self._marcas      = []
        self._pixmap_item = self._escena.addPixmap(pixmap)
        self._escena.setSceneRect(QRectF(pixmap.rect()))
        self.ajustar_a_ventana()

    def limpiar(self) -> None:
        self._escena.clear()
        self._pixmap_item = None
        self._marcas      = []
        self._zoom        = 1.0

    def ajustar_a_ventana(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._zoom = 1.0

    # region visible en coordenadas de imagen

    def obtener_viewport_imagen(self) -> tuple | None:
        """
        devuelve (x, y, w, h) con la region de la imagen actualmente
        visible en el viewport, en coordenadas de pixel de imagen.
        retorna none si no hay imagen cargada.
        """
        if self._pixmap_item is None:
            return None

        vp = self.viewport().rect()
        tl = self.mapToScene(vp.topLeft())
        br = self.mapToScene(vp.bottomRight())

        img_w = int(self._escena.sceneRect().width())
        img_h = int(self._escena.sceneRect().height())

        x  = max(0, int(tl.x()))
        y  = max(0, int(tl.y()))
        x2 = min(img_w, int(br.x()))
        y2 = min(img_h, int(br.y()))

        w = x2 - x
        h = y2 - y

        if w < 4 or h < 4:
            return None

        return (x, y, w, h)

    # sincronizacion de zoom entre visores

    def copiar_estado_zoom(self, otro: 'VisorImagen') -> None:
        """
        copia la transformacion y posicion de scroll de otro visor.
        llamar despues de establecer_imagen para preservar el zoom.
        """
        self.setTransform(otro.transform())
        self.horizontalScrollBar().setValue(otro.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(otro.verticalScrollBar().value())

    def enfocar_region(self, x: int, y: int, w: int, h: int,
                       margen: float = 0.12) -> None:
        """
        encuadra el viewport en la region indicada (coordenadas de imagen)
        con un margen proporcional. llamar despues de establecer_imagen.
        """
        if self._pixmap_item is None:
            return
        mx = w * margen
        my = h * margen
        self.fitInView(
            QRectF(x - mx, y - my, w + 2 * mx, h + 2 * my),
            Qt.KeepAspectRatio,
        )
        self._zoom = self.transform().m11()

    def marcar_region(self, x: int, y: int, w: int, h: int,
                      color: str = '#FF5252') -> None:
        """
        dibuja un recuadro permanente sobre la region indicada
        (coordenadas de imagen). el grosor es cosmetico (constante con el
        zoom). se pueden acumular varios recuadros (region + inset); todos
        se borran automaticamente en la siguiente llamada a establecer_imagen.
        llamar despues de establecer_imagen.
        """
        if self._pixmap_item is None:
            return
        pen = QPen(QColor(color), 2)
        pen.setCosmetic(True)
        item = self._escena.addRect(QRectF(x, y, w, h), pen)
        self._marcas.append(item)

    # zoom con rueda

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self._pixmap_item:
            return

        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        nuevo  = self._zoom * factor

        if 0.05 <= nuevo <= 50.0:
            self.scale(factor, factor)
            self._zoom = nuevo

    # modo seleccion rectangular

    def activar_modo_seleccion(self, activo: bool) -> None:
        self._modo_sel = activo
        self.setDragMode(
            QGraphicsView.NoDrag if activo else QGraphicsView.ScrollHandDrag
        )
        if not activo and self._sel_rect:
            self._escena.removeItem(self._sel_rect)
            self._sel_rect = None

    def mousePressEvent(self, event) -> None:
        if self._modo_sel and event.button() == Qt.LeftButton:
            self._sel_inicio = self.mapToScene(event.pos())
            if self._sel_rect:
                self._escena.removeItem(self._sel_rect)
                self._sel_rect = None
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        # Emitir señal de posicion para modo lupa (solo si hay imagen cargada)
        if self._pixmap_item is not None:
            scene_pos = self.mapToScene(event.pos())
            x_img = int(scene_pos.x())
            y_img = int(scene_pos.y())
            img_w = int(self._escena.sceneRect().width())
            img_h = int(self._escena.sceneRect().height())
            if 0 <= x_img < img_w and 0 <= y_img < img_h:
                self.mouse_moved.emit(x_img, y_img)

        # Logica existente para modo seleccion rectangular
        if self._modo_sel and self._sel_inicio is not None:
            actual = self.mapToScene(event.pos())
            rect   = QRectF(self._sel_inicio, actual).normalized()
            if self._sel_rect:
                self._sel_rect.setRect(rect)
            else:
                pen = QPen(QColor('#2196F3'), 2)
                pen.setCosmetic(True)
                self._sel_rect = self._escena.addRect(rect, pen)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._modo_sel and event.button() == Qt.LeftButton:
            if self._sel_inicio is not None:
                final = self.mapToScene(event.pos())
                rect  = QRectF(self._sel_inicio, final).normalized()
                if rect.width() > 4 and rect.height() > 4:
                    self.region_seleccionada.emit(
                        int(rect.x()), int(rect.y()),
                        int(rect.width()), int(rect.height())
                    )
                self._sel_inicio = None
        else:
            super().mouseReleaseEvent(event)
