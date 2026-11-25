"""
Capa 1 - Visor de imagen con zoom (rueda), pan (arrastrar) y seleccion.

La escena mide lo mismo que la imagen en pixeles: coordenadas de escena =
coordenadas de imagen.

Detalle importante para la demostracion: al acercarse (escala >= 1) el visor
dibuja con vecino mas proximo, de modo que se ven los pixeles reales de cada
imagen. Si se suavizara el zoom, el propio visor "interpolaria" y ocultaria
la diferencia entre los metodos de ampliacion.
"""

import numpy as np
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor, QPen, QWheelEvent


class VisorImagen(QGraphicsView):
    region_seleccionada = Signal(int, int, int, int)   # (x, y, w, h) en la imagen
    mouse_moved = Signal(int, int)                      # (x, y) en la imagen
    vista_cambiada = Signal()                           # zoom o desplazamiento
    doble_clic = Signal()                               # doble clic sobre la imagen

    def __init__(self, parent=None):
        super().__init__(parent)
        self._escena = QGraphicsScene(self)
        self._pixmap_item = None
        self._modo_sel = False
        self._sel_inicio = None
        self._sel_rect = None
        self._marcas = []
        self._sincronizando = False
        self._auto_ajuste = True   # reajustar al cambiar de tamano hasta que el usuario haga zoom

        self.setMouseTracking(True)
        self.setScene(self._escena)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.horizontalScrollBar().valueChanged.connect(self._emitir_cambio)
        self.verticalScrollBar().valueChanged.connect(self._emitir_cambio)

    # utilidades internas

    def _emitir_cambio(self, *_):
        if not self._sincronizando:
            self.vista_cambiada.emit()

    def _escala(self) -> float:
        return self.transform().m11()

    def _actualizar_suavizado(self) -> None:
        """Suaviza solo al alejar (evita aliasing); al acercar, pixeles reales."""
        suave = self._escala() < 1.0
        self.setRenderHint(QPainter.SmoothPixmapTransform, suave)
        if self._pixmap_item is not None:
            self._pixmap_item.setTransformationMode(
                Qt.SmoothTransformation if suave else Qt.FastTransformation)

    # mostrar imagen

    def establecer_imagen(self, imagen: np.ndarray, ajustar: bool = True) -> None:
        """Recibe ndarray float [0,1] (gris o RGB) y lo muestra."""
        if imagen is None:
            return
        img_u8 = np.ascontiguousarray(
            np.round(np.clip(imagen, 0.0, 1.0) * 255).astype(np.uint8))
        if img_u8.ndim == 2:
            h, w = img_u8.shape
            qimg = QImage(img_u8.data, w, h, w, QImage.Format_Grayscale8)
        else:
            img_u8 = np.ascontiguousarray(img_u8[..., :3])
            h, w, _ = img_u8.shape
            qimg = QImage(img_u8.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg.copy())

        mismo_tamano = (self._pixmap_item is not None
                        and self._escena.sceneRect().size() == QRectF(pixmap.rect()).size())
        if self._pixmap_item is not None and mismo_tamano and not ajustar:
            self._pixmap_item.setPixmap(pixmap)   # conserva zoom y posicion
        else:
            self._escena.clear()
            self._sel_rect = None
            self._marcas = []
            self._pixmap_item = self._escena.addPixmap(pixmap)
            self._escena.setSceneRect(QRectF(pixmap.rect()))
            self.ajustar_a_ventana()
        self._actualizar_suavizado()

    def limpiar(self) -> None:
        self._escena.clear()
        self._pixmap_item = None
        self._sel_rect = None
        self._marcas = []
        self.resetTransform()

    def ajustar_a_ventana(self) -> None:
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._auto_ajuste = True
            self._actualizar_suavizado()
            self._emitir_cambio()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._auto_ajuste and self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)
            self._actualizar_suavizado()

    # region visible y marcas

    def obtener_viewport_imagen(self):
        """(x, y, w, h) de la parte visible de la imagen, o None."""
        if self._pixmap_item is None:
            return None
        vp = self.viewport().rect()
        tl = self.mapToScene(vp.topLeft())
        br = self.mapToScene(vp.bottomRight())
        img_w = int(self._escena.sceneRect().width())
        img_h = int(self._escena.sceneRect().height())
        x, y = max(0, int(tl.x())), max(0, int(tl.y()))
        x2, y2 = min(img_w, int(np.ceil(br.x()))), min(img_h, int(np.ceil(br.y())))
        w, h = x2 - x, y2 - y
        if w < 4 or h < 4:
            return None
        return (x, y, w, h)

    def enfocar_region(self, x: int, y: int, w: int, h: int,
                       margen: float = 0.12) -> None:
        """
        Encuadra el viewport en la region indicada (coordenadas de imagen)
        con un margen proporcional. Llamar despues de establecer_imagen.
        """
        if self._pixmap_item is None:
            return
        self._auto_ajuste = False
        mx = w * margen
        my = h * margen
        self.fitInView(
            QRectF(x - mx, y - my, w + 2 * mx, h + 2 * my),
            Qt.KeepAspectRatio,
        )
        self._actualizar_suavizado()
        self._emitir_cambio()

    def marcar_region(self, x: int, y: int, w: int, h: int,
                      color: str = '#FF5252') -> None:
        """
        Dibuja un recuadro permanente sobre la region indicada (coordenadas de
        imagen) con grosor cosmetico constante respecto al zoom. Se borran
        automaticamente en la siguiente llamada a establecer_imagen(ajustar=True).
        """
        if self._pixmap_item is None:
            return
        pen = QPen(QColor(color), 2)
        pen.setCosmetic(True)
        item = self._escena.addRect(QRectF(x, y, w, h), pen)
        self._marcas.append(item)

    def copiar_estado_zoom(self, otro: 'VisorImagen') -> None:
        """Copia la escala y la posicion de otro visor (zoom sincronizado)."""
        self._sincronizando = True
        try:
            self._auto_ajuste = otro._auto_ajuste
            self.setTransform(otro.transform())
            self.horizontalScrollBar().setValue(otro.horizontalScrollBar().value())
            self.verticalScrollBar().setValue(otro.verticalScrollBar().value())
            self._actualizar_suavizado()
        finally:
            self._sincronizando = False

    # eventos

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self._pixmap_item:
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        if 0.02 <= self._escala() * factor <= 64.0:
            self._auto_ajuste = False
            self.scale(factor, factor)
            self._actualizar_suavizado()
            self._emitir_cambio()

    def activar_modo_seleccion(self, activo: bool) -> None:
        self._modo_sel = activo
        self.setDragMode(QGraphicsView.NoDrag if activo else QGraphicsView.ScrollHandDrag)
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
        if self._pixmap_item is not None:
            p = self.mapToScene(event.pos())
            xi, yi = int(p.x()), int(p.y())
            r = self._escena.sceneRect()
            if 0 <= xi < int(r.width()) and 0 <= yi < int(r.height()):
                self.mouse_moved.emit(xi, yi)

        if self._modo_sel and self._sel_inicio is not None:
            rect = QRectF(self._sel_inicio, self.mapToScene(event.pos())).normalized()
            if self._sel_rect:
                self._sel_rect.setRect(rect)
            else:
                pen = QPen(QColor('#2196F3'), 2)
                pen.setCosmetic(True)
                self._sel_rect = self._escena.addRect(rect, pen)
        else:
            super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self._pixmap_item is not None and event.button() == Qt.LeftButton:
            self.doble_clic.emit()
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._modo_sel and event.button() == Qt.LeftButton:
            if self._sel_inicio is not None:
                rect = QRectF(self._sel_inicio, self.mapToScene(event.pos())).normalized()
                if rect.width() > 4 and rect.height() > 4:
                    self.region_seleccionada.emit(int(rect.x()), int(rect.y()),
                                                  int(rect.width()), int(rect.height()))
                self._sel_inicio = None
        else:
            super().mouseReleaseEvent(event)
