"""
Capa 1 - Ventana principal de la aplicacion RZI.

Disposicion:
    [parametros] | [guia de pasos / visor original / visor ampliado] | [resultados]

Flujo principal (guia de 3 pasos en la parte superior):
  1. Abrir una imagen (boton, arrastrar y soltar, o Imagenes de ejemplo).
  2. Hacer zoom con la rueda sobre la zona que se quiere ampliar. La etiqueta
     bajo los visores dice el tamano de la region y de la salida.
  3. Ampliar region (RZI): back-projection + Poisson screened resuelto con
     SOR Red-Black. Muestra zoom nativo vs bicubica vs RZI.

Herramientas de analisis:
  Medir calidad : toma la region visible como verdad de campo, la reduce y la
                  vuelve a ampliar con bicubica, Lanczos y RZI (PSNR, SSIM,
                  GMSD, nitidez).
  Barrido de omega, Benchmark Numba, Lupa y Animacion paso a paso.

Ningun error queda en silencio: los errores de los hilos y de los slots se
muestran en un dialogo y se guardan en rzi_errores.log.
"""

import functools
import inspect
import os
import time
import traceback

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')   # las graficas se dibujan a imagen y se muestran en un visor
import matplotlib.pyplot as plt

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox, QGroupBox, QSplitter,
    QProgressBar, QToolBar, QTabWidget, QScrollArea, QGridLayout, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QSlider, QComboBox,
    QListWidget, QListWidgetItem, QStackedWidget, QPlainTextEdit, QApplication,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QAction

from ui.zoom_viewer import VisorImagen
from core.poisson_builder import construir_problema, imagen_a_grises
from core.sor_solver import (barrido_omega, benchmark_numba_vs_python,
                             NUMBA_DISPONIBLE, _calentamiento, CalculoCancelado)
from core.upscaler import (upscale, generar_frames_pipeline, PARAMS_DEFECTO,
                           MAX_PIXELES_BLOQUE, MAX_PIXELES_SALIDA,
                           SalidaDemasiadoGrande, verificar_tamano)
from core.image_ops import resize_float, resize_rgb_float
from services.metrics import (degradar, interpolacion_bicubica,
                              interpolacion_lanczos, tabla_comparativa,
                              nitidez_gradiente, consistencia_datos)
from services.exporter import (guardar_imagen, exportar_csv, grafica_convergencia,
                               grafica_barrido_omega, grafica_comparacion,
                               grafica_benchmark, elegir_recorte, figura_a_imagen)
from services.diagnostico import ejecutar_diagnostico
from core.denoise import estimar_ruido, UMBRAL_AUTO


RUTA_LOG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'rzi_errores.log')

# tema oscuro: fondo gris oscuro con acento azul
ESTILO = """
QMainWindow, QWidget {
    background-color: #1E1E1E;
    color: #E0E0E0;
    font-family: 'Segoe UI', 'Ubuntu', 'Arial';
    font-size: 10pt;
}
QGroupBox {
    background-color: #252525;
    border: 1px solid #3A3A3A;
    border-radius: 6px;
    margin-top: 12px;
    padding: 10px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 4px 10px;
    color: #2196F3;
    background-color: #1E1E1E;
    border: 1px solid #3A3A3A;
    border-radius: 3px;
}
QPushButton {
    background-color: #2F2F2F;
    color: #E0E0E0;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: 500;
}
QPushButton:hover   { background-color: #3A3A3A; border-color: #505050; }
QPushButton:pressed { background-color: #404040; }
QPushButton:disabled { background-color: #1F1F1F; color: #606060; border-color: #2A2A2A; }
QPushButton:checked { background-color: #1976D2; border-color: #1565C0; }
QPushButton#btn_principal {
    background-color: #2196F3;
    color: white;
    border: 1px solid #1976D2;
    font-weight: bold;
}
QPushButton#btn_principal:hover   { background-color: #42A5F5; }
QPushButton#btn_principal:pressed { background-color: #1976D2; }
QPushButton#btn_principal:disabled { background-color: #1F3A52; color: #8AA4BA; }
QPushButton#btn_cancelar { background-color: #5D1F1F; border-color: #8E2A2A; }
QPushButton#btn_enlace { background: transparent; border: none; color: #64B5F6;
                         text-align: left; padding: 4px 2px; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #2A2A2A;
    color: #E0E0E0;
    border: 1px solid #404040;
    border-radius: 4px;
    padding: 5px 8px;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #505050; }
QComboBox QAbstractItemView {
    background-color: #2A2A2A;
    color: #E0E0E0;
    selection-background-color: #2196F3;
}
QTabWidget::pane {
    border: 1px solid #3A3A3A;
    background-color: #252525;
    border-radius: 4px;
}
QTabBar::tab {
    background-color: #2A2A2A;
    color: #B0B0B0;
    border: 1px solid #3A3A3A;
    padding: 8px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
}
QTabBar::tab:selected { background-color: #252525; color: #2196F3; border-bottom: 2px solid #2196F3; }
QTabBar::tab:hover    { background-color: #353535; color: #E0E0E0; }
QScrollArea { background-color: #1E1E1E; border: none; }
QLabel      { background-color: transparent; }
QLabel#vacio { color: #8A9BA8; font-size: 11pt; padding: 30px; }
QLabel#paso { color: #707070; background-color: #252525; border: 1px solid #333;
              border-radius: 12px; padding: 4px 12px; }
QLabel#paso[estado="activo"] { color: white; background-color: #1976D2; border-color: #42A5F5;
                               font-weight: bold; }
QLabel#paso[estado="hecho"] { color: #A5D6A7; border-color: #2E7D32; }
QStatusBar  { background-color: #1A1A1A; color: #B0B0B0; border-top: 1px solid #3A3A3A; }
QMenuBar { background-color: #252525; color: #E0E0E0; }
QMenuBar::item:selected { background-color: #3A3A3A; }
QMenu { background-color: #2A2A2A; color: #E0E0E0; border: 1px solid #404040; }
QMenu::item:selected { background-color: #1976D2; }
QToolBar    { background-color: #252525; border: none; spacing: 5px; padding: 4px; }
QToolBar QToolButton {
    background-color: transparent;
    color: #E0E0E0;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 6px 12px;
}
QToolBar QToolButton:hover { background-color: #2F2F2F; border-color: #404040; }
QProgressBar {
    background-color: #1A1A1A;
    border: 1px solid #3A3A3A;
    border-radius: 4px;
    text-align: center;
    color: #E0E0E0;
    min-height: 22px;
}
QProgressBar::chunk { background-color: #2196F3; border-radius: 3px; }
QTableWidget {
    background-color: #252525;
    gridline-color: #3A3A3A;
    alternate-background-color: #2A2A2A;
    color: #E0E0E0;
    border: 1px solid #3A3A3A;
}
QHeaderView::section {
    background-color: #2A2A2A;
    color: #2196F3;
    padding: 5px;
    border: 1px solid #3A3A3A;
    font-weight: bold;
}
QCheckBox { color: #E0E0E0; spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #707070;
                       border-radius: 3px; background-color: #2A2A2A; }
QCheckBox::indicator:checked { background-color: #2196F3; border-color: #90CAF9; }
QSlider::groove:horizontal { background: #3A3A3A; height: 8px; border-radius: 4px; }
QSlider::handle:horizontal { background: #2196F3; width: 18px; margin: -5px 0; border-radius: 9px; }
QListWidget {
    background-color: #252525;
    border: 1px solid #3A3A3A;
    border-radius: 4px;
    color: #C0C0C0;
}
QListWidget::item { padding: 5px 8px; }
QListWidget::item:selected {
    background-color: #1565C0;
    color: #FFFFFF;
    font-weight: bold;
}
QPlainTextEdit { background-color: #151515; color: #D0D0D0; font-family: 'Consolas', 'DejaVu Sans Mono', monospace; }
"""

GUIA_USO = """\
COMO USAR RZI

1. Abra una imagen: boton "Abrir imagen", arrastrela a la ventana, o use
   Archivo > Imagenes de ejemplo.
2. Con la rueda del mouse, haga zoom en el visor izquierdo sobre la zona que
   quiere ampliar (arrastre para moverse). La etiqueta bajo los visores dice
   el tamano de la region y de la imagen de salida.
3. Pulse "Ampliar region (RZI)". Se abre una comparacion lado a lado: zoom
   nativo, bicubica y RZI, con zoom sincronizado.

Medir calidad: usa la region visible como "verdad de campo", la achica a
proposito y la vuelve a ampliar con cada metodo. La tabla muestra PSNR y SSIM
(mayor es mejor) y GMSD (menor es mejor).

Barrido de omega y Benchmark Numba: herramientas para la sustentacion del
metodo numerico (se pueden usar en cualquier momento con una imagen abierta).

Calidad: "Mejor calidad" es el ajuste recomendado. "Rapido" sirve para
probar. "Bordes marcados" agrega un realce visual (baja un poco el PSNR).

Si algo falla, aparece un mensaje y el detalle queda en rzi_errores.log.
Ayuda > Diagnostico revisa que cada herramienta funcione en este equipo.
"""


# Errores visibles

def registrar_error(texto: str) -> None:
    """Agrega un error al archivo rzi_errores.log (nunca lanza excepcion)."""
    try:
        with open(RUTA_LOG, 'a', encoding='utf-8') as fh:
            fh.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n{texto}\n")
    except OSError:
        pass


def mostrar_error(parent, titulo: str, resumen: str, detalle: str = '') -> None:
    """Muestra un error con su detalle (traceback) y lo registra en el log."""
    registrar_error(f'{titulo}: {resumen}\n{detalle}')
    caja = QMessageBox(parent)
    caja.setIcon(QMessageBox.Critical)
    caja.setWindowTitle(titulo)
    caja.setText(resumen)
    caja.setInformativeText(f'El detalle quedo guardado en:\n{RUTA_LOG}')
    if detalle:
        caja.setDetailedText(detalle)
    caja.exec()


def reportar_errores(metodo):
    """
    Decorador para slots de Qt: si el slot lanza una excepcion, se muestra un
    dialogo en lugar de imprimirse solo en la consola (que en Windows suele
    no verse y deja la pestana vacia sin ninguna explicacion).
    """
    @functools.wraps(metodo)
    def envoltura(self, *args, **kwargs):
        try:
            return metodo(self, *args, **kwargs)
        except Exception as exc:
            mostrar_error(self, 'Error al mostrar el resultado',
                          f'{type(exc).__name__}: {exc}', traceback.format_exc())
    return envoltura


class HiloCalculo(QThread):
    """
    Ejecuta una funcion en segundo plano para no congelar la interfaz.

    Si la funcion acepta los argumentos 'progreso' y/o 'cancelado', se le
    pasan: progreso(fraccion, texto) actualiza la barra y cancelado()
    indica si el usuario pulso Cancelar.
    """
    terminado = Signal(object)
    error = Signal(str, str)          # resumen, traceback
    cancelado_sig = Signal()
    progreso = Signal(float, str)

    def __init__(self, funcion):
        super().__init__()
        self._fn = funcion
        self._cancelar = False
        self._ultimo = 0.0
        params = inspect.signature(funcion).parameters
        self._acepta = {n for n in ('progreso', 'cancelado') if n in params}

    def cancelar(self):
        self._cancelar = True

    def _emitir_progreso(self, fraccion: float, texto: str) -> None:
        ahora = time.perf_counter()
        if ahora - self._ultimo > 0.05 or fraccion >= 1.0:
            self._ultimo = ahora
            self.progreso.emit(float(fraccion), str(texto))

    def run(self):
        kw = {}
        if 'progreso' in self._acepta:
            kw['progreso'] = self._emitir_progreso
        if 'cancelado' in self._acepta:
            kw['cancelado'] = lambda: self._cancelar
        try:
            self.terminado.emit(self._fn(**kw))
        except CalculoCancelado:
            self.cancelado_sig.emit()
        except Exception as exc:   # se muestra al usuario, no se silencia
            self.error.emit(f'{type(exc).__name__}: {exc}', traceback.format_exc())


# Panel de parametros

PRESETS = {
    'Mejor calidad': dict(PARAMS_DEFECTO, realce=0.0),
    'Rapido': dict(PARAMS_DEFECTO, ciclos=4, sor_iters=60, bp_final=3, realce=0.0),
    'Bordes marcados': dict(PARAMS_DEFECTO, realce=0.6),
}
PERSONALIZADO = 'Personalizado'

ENTRADAS = (
    ('Automatica (recomendada)', 'auto'),
    ('Con ruido: limpiar', 'ruido'),
    ('JPEG muy comprimido', 'jpeg'),
    ('Sin limpieza', 'no'),
)
TEXTO_LIMPIEZA = {'no': 'sin limpieza', 'ruido': 'limpieza de ruido',
                  'jpeg': 'suavizado de bloques JPEG'}


class PanelParametros(QWidget):
    reconstruir_solicitado = Signal()
    comparar_solicitado = Signal()
    barrido_omega_solicitado = Signal()
    benchmark_solicitado = Signal()
    lupa_solicitado = Signal(bool)
    animar_solicitado = Signal()
    parametros_cambiados = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(250)
        self.setMaximumWidth(310)
        self._aplicando_preset = False
        self._construir()
        self.aplicar_preset(next(iter(PRESETS)))

    @staticmethod
    def _spin(minimo, maximo, valor, paso, decimales=None, tooltip=''):
        s = QDoubleSpinBox() if decimales is not None else QSpinBox()
        if decimales is not None:
            s.setDecimals(decimales)
        s.setRange(minimo, maximo)
        s.setValue(valor)
        s.setSingleStep(paso)
        s.setToolTip(tooltip)
        return s

    def _construir(self):
        d = PARAMS_DEFECTO
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # Ampliacion (lo esencial)
        g_amp = QGroupBox('Ampliacion')
        gl = QGridLayout(g_amp)
        gl.addWidget(QLabel('Factor:'), 0, 0)
        self.spin_factor = self._spin(2, 12, 4, 1, tooltip=(
            'Cuantas veces se agranda cada lado. Factor 4: cada pixel '
            'original se convierte en un bloque de 4x4.'))
        self.spin_factor.setSuffix('x')
        gl.addWidget(self.spin_factor, 0, 1)
        gl.addWidget(QLabel('Calidad:'), 1, 0)
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(PRESETS) + [PERSONALIZADO])
        self.combo_preset.setToolTip(
            'Mejor calidad: ajuste recomendado (mejores metricas).\n'
            'Rapido: menos ciclos, para probar.\n'
            'Bordes marcados: agrega realce visual (baja un poco el PSNR).')
        gl.addWidget(self.combo_preset, 1, 1)
        gl.addWidget(QLabel('Entrada:'), 2, 0)
        self.combo_entrada = QComboBox()
        for texto, modo in ENTRADAS:
            self.combo_entrada.addItem(texto, modo)
        self.combo_entrada.setToolTip(
            'Automatica: mide el ruido de la zona y la limpia solo si es claro.\n'
            'Con ruido: limpia siempre con variacion total (fotos de celular de noche,\n'
            'escaneos con grano).\n'
            'JPEG muy comprimido: suaviza levemente los bloques de 8x8.\n'
            'Sin limpieza: usa la imagen tal cual.')
        gl.addWidget(self.combo_entrada, 2, 1)
        lay.addWidget(g_amp)

        self.btn_reconstruir = QPushButton('Ampliar region (RZI)')
        self.btn_reconstruir.setObjectName('btn_principal')
        self.btn_reconstruir.setMinimumHeight(40)
        self.btn_reconstruir.setToolTip('Amplia la zona visible en el visor izquierdo.')
        self.btn_reconstruir.clicked.connect(self.reconstruir_solicitado)
        lay.addWidget(self.btn_reconstruir)

        self.btn_animar = QPushButton('Ver el proceso paso a paso')
        self.btn_animar.setToolTip('Animacion de cada etapa del metodo sobre la zona visible.')
        self.btn_animar.clicked.connect(self.animar_solicitado)
        self.btn_lupa = QPushButton('Lupa en tiempo real')
        self.btn_lupa.setCheckable(True)
        self.btn_lupa.setToolTip('Amplia x3 la zona bajo el cursor con el modelo RZI.')
        self.btn_lupa.toggled.connect(self.lupa_solicitado)
        for b in (self.btn_animar, self.btn_lupa):
            b.setMinimumHeight(30)
            lay.addWidget(b)

        # Herramientas de analisis
        g_an = QGroupBox('Analisis y sustentacion')
        al = QVBoxLayout(g_an)
        self.btn_comparar = QPushButton('Medir calidad (comparar)')
        self.btn_comparar.setToolTip(
            'Achica la zona visible, la vuelve a ampliar con bicubica, Lanczos y RZI '
            'y mide cuanto se parece cada una al original.')
        self.btn_comparar.clicked.connect(self.comparar_solicitado)
        self.btn_barrido = QPushButton('Barrido de omega')
        self.btn_barrido.setToolTip('Curva de iteraciones de SOR frente a omega (forma de V).')
        self.btn_barrido.clicked.connect(self.barrido_omega_solicitado)
        self.btn_benchmark = QPushButton('Benchmark Numba')
        self.btn_benchmark.setToolTip('Compara la velocidad de Numba contra Python puro.')
        self.btn_benchmark.clicked.connect(self.benchmark_solicitado)
        for b in (self.btn_comparar, self.btn_barrido, self.btn_benchmark):
            b.setMinimumHeight(30)
            al.addWidget(b)
        lay.addWidget(g_an)

        # Opciones avanzadas (plegables)
        self.btn_avanzadas = QPushButton('Mostrar opciones avanzadas')
        self.btn_avanzadas.setObjectName('btn_enlace')
        self.btn_avanzadas.setCheckable(True)
        lay.addWidget(self.btn_avanzadas)

        self.cont_avanzadas = QWidget()
        av = QVBoxLayout(self.cont_avanzadas)
        av.setContentsMargins(0, 0, 0, 0)

        g_mod = QGroupBox('Modelo RZI')
        ml = QGridLayout(g_mod)
        filas = [
            ('Realce perceptual:', 'spin_realce', self._spin(0.0, 1.0, 0.0, 0.1, 2,
             '0 = salida mas fiel. Mayor = bordes mas marcados a la vista (baja el PSNR).')),
            ('Lambda (fidelidad):', 'spin_lam', self._spin(0.05, 5.0, d['lam'], 0.05, 2,
             'Peso de lambda*(u - t). Pequeno: manda el campo de gradientes.')),
            ('Beta (bordes):', 'spin_beta', self._spin(1.0, 3.0, d['beta'], 0.1, 2,
             'Refuerzo de bordes del primer ciclo. 1 = sin refuerzo.')),
            ('Rho (decaimiento):', 'spin_rho', self._spin(0.5, 1.0, d['rho'], 0.01, 2,
             'El refuerzo del ciclo k es 1 + (beta-1)*rho^k. Menor que 1 garantiza '
             'estabilidad; 1 = refuerzo constante (inestable con muchos ciclos).')),
            ('Tau (umbral):', 'spin_tau', self._spin(0.005, 0.3, d['tau'], 0.005, 3,
             'Gradientes menores que tau casi no se refuerzan (evita realzar ruido).')),
            ('Sigma (suavizado):', 'spin_sigma', self._spin(0.0, 2.0, d['sigma'], 0.1, 2,
             'Suavizado Gaussiano antes de medir gradientes.')),
            ('Ciclos:', 'spin_ciclos', self._spin(1, 20, d['ciclos'], 1,
             tooltip='Veces que se alterna back-projection + SOR (10 recomendado).')),
            ('Back-proj. final:', 'spin_bp', self._spin(0, 20, d['bp_final'], 1,
             tooltip='Pasadas finales de back-projection.')),
        ]
        for i, (txt, attr, w) in enumerate(filas):
            setattr(self, attr, w)
            ml.addWidget(QLabel(txt), i, 0)
            ml.addWidget(w, i, 1)
        ml.addWidget(QLabel('Anti-ringing:'), len(filas), 0)
        self.chk_anti_ringing = QCheckBox()
        self.chk_anti_ringing.setToolTip('Recorta al rango local de la bicubica (quita halos).')
        ml.addWidget(self.chk_anti_ringing, len(filas), 1)
        av.addWidget(g_mod)

        g_sol = QGroupBox('Solver SOR Red-Black')
        sl = QGridLayout(g_sol)
        sl.addWidget(QLabel('Tolerancia:'), 0, 0)
        self.spin_tol = self._spin(1e-9, 1e-2, d['tol'], 1e-6, 9,
                                   tooltip='Residuo relativo ||r||/||b|| para parar.')
        sl.addWidget(self.spin_tol, 0, 1)
        sl.addWidget(QLabel('Iter. por ciclo:'), 1, 0)
        self.spin_iters = self._spin(10, 5000, d['sor_iters'], 10,
                                     tooltip='Maximo de iteraciones SOR en cada ciclo.')
        sl.addWidget(self.spin_iters, 1, 1)
        sl.addWidget(QLabel('Omega auto:'), 2, 0)
        self.chk_omega_auto = QCheckBox()
        self.chk_omega_auto.setChecked(True)
        sl.addWidget(self.chk_omega_auto, 2, 1)
        sl.addWidget(QLabel('Omega manual:'), 3, 0)
        self.spin_omega = self._spin(1.0, 1.99, 1.4, 0.05, 3)
        self.spin_omega.setEnabled(False)
        self.chk_omega_auto.toggled.connect(lambda a: self.spin_omega.setEnabled(not a))
        sl.addWidget(self.spin_omega, 3, 1)
        av.addWidget(g_sol)

        btn_restaurar = QPushButton('Restaurar valores recomendados')
        btn_restaurar.clicked.connect(lambda: self.combo_preset.setCurrentIndex(0))
        av.addWidget(btn_restaurar)
        self.cont_avanzadas.setVisible(False)
        lay.addWidget(self.cont_avanzadas)

        def plegar(abierto):
            self.cont_avanzadas.setVisible(abierto)
            self.btn_avanzadas.setText('Ocultar opciones avanzadas' if abierto
                                       else 'Mostrar opciones avanzadas')
        self.btn_avanzadas.toggled.connect(plegar)

        color = '#4CAF50' if NUMBA_DISPONIBLE else '#FF9800'
        texto = ('Numba: activo (calculo rapido)' if NUMBA_DISPONIBLE else
                 'Numba NO disponible: todo sera muy lento. Instale numba.')
        self.lbl_numba = QLabel(texto)
        self.lbl_numba.setWordWrap(True)
        self.lbl_numba.setStyleSheet(f'color: {color}; font-size: 9pt; padding: 4px;')
        lay.addWidget(self.lbl_numba)
        lay.addStretch()

        # sincronizacion preset <-> controles
        self.combo_preset.currentTextChanged.connect(self.aplicar_preset)
        for w in (self.spin_realce, self.spin_lam, self.spin_beta, self.spin_rho,
                  self.spin_tau, self.spin_sigma, self.spin_ciclos, self.spin_bp,
                  self.spin_tol, self.spin_iters, self.spin_omega):
            w.valueChanged.connect(self._marcar_personalizado)
        for w in (self.chk_anti_ringing, self.chk_omega_auto):
            w.toggled.connect(self._marcar_personalizado)
        self.spin_factor.valueChanged.connect(self.parametros_cambiados)

    def aplicar_preset(self, nombre: str) -> None:
        if nombre not in PRESETS:
            return
        p = PRESETS[nombre]
        self._aplicando_preset = True
        try:
            self.spin_realce.setValue(p['realce'])
            self.spin_lam.setValue(p['lam'])
            self.spin_beta.setValue(p['beta'])
            self.spin_rho.setValue(p['rho'])
            self.spin_tau.setValue(p['tau'])
            self.spin_sigma.setValue(p['sigma'])
            self.spin_ciclos.setValue(p['ciclos'])
            self.spin_bp.setValue(p['bp_final'])
            self.spin_tol.setValue(p['tol'])
            self.spin_iters.setValue(p['sor_iters'])
            self.chk_anti_ringing.setChecked(False)
            self.chk_omega_auto.setChecked(True)
            if self.combo_preset.currentText() != nombre:
                self.combo_preset.setCurrentText(nombre)
        finally:
            self._aplicando_preset = False
        self.parametros_cambiados.emit()

    def _marcar_personalizado(self, *_):
        if not self._aplicando_preset and self.combo_preset.currentText() != PERSONALIZADO:
            self.combo_preset.blockSignals(True)
            self.combo_preset.setCurrentText(PERSONALIZADO)
            self.combo_preset.blockSignals(False)

    def obtener_params(self) -> dict:
        return {
            'factor':       self.spin_factor.value(),
            'realce':       self.spin_realce.value(),
            'lam':          self.spin_lam.value(),
            'beta':         self.spin_beta.value(),
            'rho':          self.spin_rho.value(),
            'tau':          self.spin_tau.value(),
            'sigma':        self.spin_sigma.value(),
            'ciclos':       self.spin_ciclos.value(),
            'bp_final':     self.spin_bp.value(),
            'anti_ringing': self.chk_anti_ringing.isChecked(),
            'tol':          self.spin_tol.value(),
            'sor_iters':    self.spin_iters.value(),
            'omega':        None if self.chk_omega_auto.isChecked() else self.spin_omega.value(),
            'limpieza':     self.combo_entrada.currentData(),
        }

    def deshabilitar(self, flag: bool):
        for btn in (self.btn_reconstruir, self.btn_comparar, self.btn_barrido,
                    self.btn_benchmark, self.btn_animar):
            btn.setEnabled(not flag)


def kwargs_rzi(p: dict, **cambios) -> dict:
    """Traduce los parametros del panel a los argumentos de core.upscaler.upscale."""
    kw = dict(lam=p['lam'], beta=p['beta'], rho=p['rho'], tau=p['tau'], sigma=p['sigma'],
              ciclos=p['ciclos'], bp_final=p['bp_final'], sor_iters=p['sor_iters'],
              tol=p['tol'], omega=p['omega'], realce=p['realce'],
              usar_anti_ringing=p['anti_ringing'], limpieza=p.get('limpieza', 'auto'),
              sigma_ruido=p.get('sigma_ruido'))
    kw.update(cambios)
    return kw


# Panel de resultados

TEXTOS_VACIOS = {
    0: ('Aqui aparece la comparacion de calidad.\n\n'
        'Haga zoom sobre una zona con detalle y pulse "Medir calidad (comparar)".\n'
        'La zona se achica a proposito y se vuelve a ampliar con bicubica, Lanczos '
        'y RZI; la tabla dice cual se parece mas al original.'),
    1: ('Aqui aparece la convergencia del solver.\n\n'
        'Pulse "Ampliar region (RZI)": se grafica el residuo del SOR y el error '
        'de consistencia en cada ciclo.'),
    2: ('Aqui aparece la curva de iteraciones frente a omega.\n\n'
        'Pulse "Barrido de omega" (con una imagen abierta). El minimo de la curva '
        'debe coincidir con el omega teorico estimado.'),
    3: ('Aqui aparece la comparacion de velocidad.\n\n'
        'Pulse "Benchmark Numba" (con una imagen de al menos 64x64 px). Se mide el '
        'mismo trabajo en Python puro y con Numba.'),
}


class PanelResultados(QWidget):
    """Pestanas con las graficas (como imagen con zoom) y la tabla de metricas."""

    TAB_COMP, TAB_CONV, TAB_OMEGA, TAB_BENCH = range(4)
    figura_ampliada = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(440)
        self.figuras = {}
        self.visores = {}
        self.pilas = {}
        self._construir()

    def _contenedor_grafica(self, idx: int) -> QStackedWidget:
        pila = QStackedWidget()
        vacio = QLabel(TEXTOS_VACIOS[idx])
        vacio.setObjectName('vacio')
        vacio.setWordWrap(True)
        vacio.setAlignment(Qt.AlignCenter)
        visor = VisorImagen()
        visor.setToolTip('Rueda: zoom. Doble clic: ver en grande.')
        visor.doble_clic.connect(lambda i=idx: self.figura_ampliada.emit(i))
        pila.addWidget(vacio)
        pila.addWidget(visor)
        self.visores[idx] = visor
        self.pilas[idx] = pila
        return pila

    def _construir(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        tab_comp = QWidget()
        lc = QVBoxLayout(tab_comp)
        lc.addWidget(self._contenedor_grafica(self.TAB_COMP), 3)
        nota = QLabel('PSNR, SSIM y nitidez: mayor es mejor. GMSD: menor es mejor. '
                      'Doble clic en la grafica para verla en grande.')
        nota.setWordWrap(True)
        nota.setStyleSheet('color: #B0B0B0; font-size: 9pt;')
        lc.addWidget(nota)
        self.tabla = QTableWidget(0, 5)
        self.tabla.setHorizontalHeaderLabels(['Modelo', 'PSNR (dB)', 'SSIM', 'GMSD', 'Nitidez'])
        self.tabla.verticalHeader().setVisible(False)
        self.tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla.setAlternatingRowColors(True)
        self.tabla.setMaximumHeight(170)
        lc.addWidget(self.tabla, 1)
        self.tabs.addTab(tab_comp, 'Comparacion')

        for idx, nombre in ((self.TAB_CONV, 'Convergencia'),
                            (self.TAB_OMEGA, 'Barrido Omega'),
                            (self.TAB_BENCH, 'Benchmark')):
            w = QWidget()
            lw = QVBoxLayout(w)
            lw.addWidget(self._contenedor_grafica(idx))
            self.tabs.addTab(w, nombre)

        lay_exp = QHBoxLayout()
        self.btn_guardar_img = QPushButton('Guardar imagen')
        self.btn_guardar_graf = QPushButton('Guardar grafica')
        self.btn_exportar_csv = QPushButton('Exportar CSV')
        for btn in (self.btn_guardar_img, self.btn_guardar_graf, self.btn_exportar_csv):
            btn.setMinimumHeight(28)
            lay_exp.addWidget(btn)
        lay.addLayout(lay_exp)

    def mostrar_figura(self, idx: int, fig) -> None:
        imagen = figura_a_imagen(fig)   # si falla, se informa (decorador del llamador)
        anterior = self.figuras.get(idx)
        if anterior is not None and anterior is not fig:
            plt.close(anterior)
        self.figuras[idx] = fig
        self.tabs.setCurrentIndex(idx)
        self.pilas[idx].setCurrentIndex(1)   # primero visible, para ajustar al tamano real
        self.visores[idx].establecer_imagen(imagen)

    def figura_actual(self):
        return self.figuras.get(self.tabs.currentIndex())

    def mostrar_tabla(self, tabla: list) -> None:
        self.tabla.setRowCount(0)
        mejor_psnr = max(f['psnr'] for f in tabla) if tabla else None
        for fila in tabla:
            r = self.tabla.rowCount()
            self.tabla.insertRow(r)
            valores = [fila['modelo'], f"{fila['psnr']:.2f}", f"{fila['ssim']:.4f}",
                       f"{fila['gmsd']:.4f}", f"{fila['nitidez']:.4f}"]
            for c, v in enumerate(valores):
                item = QTableWidgetItem(v)
                if fila['psnr'] == mejor_psnr:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self.tabla.setItem(r, c, item)


# Ventana principal

class VentanaPrincipal(QMainWindow):
    INSET_FRAC = 0.40
    MIN_LADO_EVAL = 32
    LADO_EVAL_MAX = 512

    def __init__(self):
        super().__init__()
        self._imagen_orig = None    # float64 [0,1] en grises (luminancia)
        self._imagen_color = None   # float64 [0,1] RGB, o None si es gris
        self._nombre_imagen = ''
        self._hilo = None
        self._t_inicio = 0.0
        self._resultado = None      # ultimo resultado (para exportar y vista con inset)
        self._modo_lupa_activo = False
        self._hilo_lupa = None
        self._ultima_lupa = 0.0
        self._ya_amplio = False

        self.setAcceptDrops(True)
        self._construir_ui()
        self._construir_menus()
        self._conectar()
        self._actualizar_guia()

        # compilar los kernels Numba en segundo plano (la primera ampliacion
        # no tendra la pausa de compilacion JIT)
        if NUMBA_DISPONIBLE:
            self._hilo_init = HiloCalculo(_calentamiento)
            self._hilo_init.terminado.connect(
                lambda _: self.statusBar().showMessage('Listo. Abra una imagen para comenzar.'))
            self._hilo_init.start()
            self.statusBar().showMessage('Preparando el solver (compilacion Numba en segundo plano)...')

    # construccion de la interfaz

    def _construir_ui(self):
        self.setWindowTitle('RZI - Reconstructor de Zoom Iterativo (Poisson + SOR Red-Black)')
        self.setMinimumSize(1280, 760)
        self.resize(1560, 920)

        central = QWidget()
        self.setCentralWidget(central)
        lay = QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)

        scroll_p = QScrollArea()
        scroll_p.setWidgetResizable(True)
        scroll_p.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_params = PanelParametros()
        scroll_p.setWidget(self.panel_params)

        cen_w = QWidget()
        cen_l = QVBoxLayout(cen_w)
        cen_l.setContentsMargins(6, 6, 6, 6)
        cen_l.setSpacing(6)

        # guia de 3 pasos
        guia = QHBoxLayout()
        guia.addStretch()
        self.lbl_pasos = []
        for texto in ('1  Abrir imagen', '2  Zoom sobre la zona', '3  Ampliar region'):
            lbl = QLabel(texto)
            lbl.setObjectName('paso')
            self.lbl_pasos.append(lbl)
            guia.addWidget(lbl)
        guia.addStretch()
        cen_l.addLayout(guia)

        visores_lay = QHBoxLayout()
        col_orig = QVBoxLayout()
        lbl_orig = QLabel('Original (zoom nativo): rueda = zoom, arrastrar = mover')
        lbl_orig.setStyleSheet('color: #2196F3; font-weight: bold; padding: 2px;')
        lbl_orig.setAlignment(Qt.AlignCenter)
        self.pila_orig = QStackedWidget()
        self.lbl_soltar = QLabel('Arrastre una imagen aqui\n\n'
                                 'o use "Abrir imagen" / Archivo > Imagenes de ejemplo')
        self.lbl_soltar.setObjectName('vacio')
        self.lbl_soltar.setAlignment(Qt.AlignCenter)
        self.lbl_soltar.setStyleSheet('border: 2px dashed #3A3A3A; border-radius: 8px;')
        self.visor_orig = VisorImagen()
        self.pila_orig.addWidget(self.lbl_soltar)
        self.pila_orig.addWidget(self.visor_orig)
        col_orig.addWidget(lbl_orig)
        col_orig.addWidget(self.pila_orig)
        visores_lay.addLayout(col_orig, 1)

        col_recon = QVBoxLayout()
        hdr_recon = QHBoxLayout()
        lbl_recon = QLabel('Ampliada con RZI')
        lbl_recon.setStyleSheet('color: #4CAF50; font-weight: bold; padding: 2px;')
        self.chk_vista_inset = QCheckBox('Vista con inset')
        self.chk_vista_inset.setToolTip(
            'Muestra la imagen original completa con la zona de zoom en rojo '
            'y un inset del resultado RZI en verde en la esquina mas lejana.')
        hdr_recon.addStretch()
        hdr_recon.addWidget(lbl_recon)
        hdr_recon.addSpacing(8)
        hdr_recon.addWidget(self.chk_vista_inset)
        hdr_recon.addStretch()
        self.pila_recon = QStackedWidget()
        vacio_recon = QLabel('Aqui aparece la zona ampliada.\n\n'
                             'Haga zoom en el visor izquierdo y pulse\n"Ampliar region (RZI)".')
        vacio_recon.setObjectName('vacio')
        vacio_recon.setAlignment(Qt.AlignCenter)
        self.visor_recon = VisorImagen()
        self.pila_recon.addWidget(vacio_recon)
        self.pila_recon.addWidget(self.visor_recon)
        col_recon.addLayout(hdr_recon)
        col_recon.addWidget(self.pila_recon)
        visores_lay.addLayout(col_recon, 1)
        cen_l.addLayout(visores_lay, 1)

        self.lbl_region = QLabel('')
        self.lbl_region.setAlignment(Qt.AlignCenter)
        self.lbl_region.setStyleSheet('color: #B0B0B0; padding: 2px;')
        cen_l.addWidget(self.lbl_region)

        fila_prog = QHBoxLayout()
        self.barra_progreso = QProgressBar()
        self.barra_progreso.setRange(0, 0)
        self.btn_cancelar = QPushButton('Cancelar')
        self.btn_cancelar.setObjectName('btn_cancelar')
        self.btn_cancelar.clicked.connect(self._cancelar)
        fila_prog.addWidget(self.barra_progreso, 1)
        fila_prog.addWidget(self.btn_cancelar)
        self.cont_progreso = QWidget()
        self.cont_progreso.setLayout(fila_prog)
        self.cont_progreso.setVisible(False)
        cen_l.addWidget(self.cont_progreso)

        self.panel_res = PanelResultados()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_p)
        splitter.addWidget(cen_w)
        splitter.addWidget(self.panel_res)
        splitter.setSizes([280, 780, 500])
        lay.addWidget(splitter)
        self.statusBar().showMessage('Listo. Abra una imagen para comenzar.')

    def _construir_menus(self):
        self.act_abrir = QAction('Abrir imagen...', self)
        self.act_abrir.setShortcut('Ctrl+O')
        self.act_ajustar = QAction('Ajustar vistas', self)
        self.act_ajustar.setShortcut('Ctrl+0')
        self.act_guardar_img = QAction('Guardar imagen ampliada...', self)
        self.act_guardar_img.setShortcut('Ctrl+S')
        self.act_guardar_graf = QAction('Guardar grafica activa...', self)
        self.act_export_csv = QAction('Exportar CSV...', self)
        self.act_salir = QAction('Salir', self)
        self.act_salir.setShortcut('Ctrl+Q')
        self.act_ayuda = QAction('Como usar', self)
        self.act_ayuda.setShortcut('F1')
        self.act_diag = QAction('Diagnostico...', self)
        self.act_log = QAction('Abrir registro de errores', self)

        menu = self.menuBar()
        m_arch = menu.addMenu('Archivo')
        m_arch.addAction(self.act_abrir)
        m_ej = m_arch.addMenu('Imagenes de ejemplo')
        for nombre, clave in (('Fotografo (grises)', 'camera'), ('Astronauta (color)', 'astronaut'),
                              ('Cafe (color)', 'coffee'), ('Gato (color)', 'chelsea')):
            act = QAction(nombre, self)
            act.triggered.connect(lambda _=False, c=clave: self._cargar_ejemplo(c))
            m_ej.addAction(act)
        m_arch.addSeparator()
        for act in (self.act_guardar_img, self.act_guardar_graf, self.act_export_csv):
            m_arch.addAction(act)
        m_arch.addSeparator()
        m_arch.addAction(self.act_salir)
        m_ver = menu.addMenu('Ver')
        m_ver.addAction(self.act_ajustar)
        m_ayuda = menu.addMenu('Ayuda')
        m_ayuda.addAction(self.act_ayuda)
        m_ayuda.addAction(self.act_diag)
        m_ayuda.addAction(self.act_log)

        tb = QToolBar('Herramientas')
        tb.setMovable(False)
        self.addToolBar(tb)
        tb.addAction(self.act_abrir)
        tb.addAction(self.act_ajustar)
        tb.addSeparator()
        for act in (self.act_guardar_img, self.act_guardar_graf, self.act_export_csv):
            tb.addAction(act)
        tb.addSeparator()
        tb.addAction(self.act_ayuda)

    def _conectar(self):
        self.act_abrir.triggered.connect(self._abrir_imagen)
        self.act_ajustar.triggered.connect(self._ajustar_vistas)
        self.act_salir.triggered.connect(self.close)
        self.act_guardar_img.triggered.connect(self._guardar_imagen)
        self.act_guardar_graf.triggered.connect(self._guardar_grafica)
        self.act_export_csv.triggered.connect(self._exportar_csv)
        self.act_ayuda.triggered.connect(self._mostrar_ayuda)
        self.act_diag.triggered.connect(self._diagnostico)
        self.act_log.triggered.connect(self._mostrar_log)
        self.chk_vista_inset.toggled.connect(self._actualizar_visor_recon)
        self.visor_orig.vista_cambiada.connect(self._actualizar_info_region)

        pp = self.panel_params
        pp.reconstruir_solicitado.connect(self._reconstruir)
        pp.comparar_solicitado.connect(self._comparar)
        pp.barrido_omega_solicitado.connect(self._barrido_omega)
        pp.benchmark_solicitado.connect(self._benchmark)
        pp.lupa_solicitado.connect(self._toggle_modo_lupa)
        pp.animar_solicitado.connect(self._animar_convergencia)
        pp.parametros_cambiados.connect(self._actualizar_info_region)

        pr = self.panel_res
        pr.btn_guardar_img.clicked.connect(self._guardar_imagen)
        pr.btn_guardar_graf.clicked.connect(self._guardar_grafica)
        pr.btn_exportar_csv.clicked.connect(self._exportar_csv)
        pr.figura_ampliada.connect(self._ver_figura_grande)

    # guia y etiqueta de region

    def _actualizar_guia(self):
        if self._imagen_orig is None:
            estados = ('activo', '', '')
        elif not self._ya_amplio:
            region = self.visor_orig.obtener_viewport_imagen()
            H, W = self._imagen_orig.shape
            con_zoom = region is not None and region[2] * region[3] < 0.9 * W * H
            estados = ('hecho', 'hecho' if con_zoom else 'activo',
                       'activo' if con_zoom else '')
        else:
            estados = ('hecho', 'hecho', 'hecho')
        for lbl, estado in zip(self.lbl_pasos, estados):
            lbl.setProperty('estado', estado)
            lbl.style().unpolish(lbl)
            lbl.style().polish(lbl)

    def _actualizar_info_region(self, *_):
        self._actualizar_guia()
        if self._imagen_orig is None:
            self.lbl_region.setText('')
            return
        region = self.visor_orig.obtener_viewport_imagen()
        H, W = self._imagen_orig.shape
        x, y, w, h = region if region is not None else (0, 0, W, H)
        k = self.panel_params.spin_factor.value()
        mp = w * h * k * k / 1e6
        texto = f'Region visible: {w}x{h} px   ->   salida x{k}: {w * k}x{h * k} px ({mp:.1f} MP)'
        color = '#B0B0B0'
        if w * h * k * k > MAX_PIXELES_SALIDA:
            texto += '   Demasiado grande: haga zoom o baje el factor'
            color = '#EF5350'
        elif w * h * k * k > MAX_PIXELES_BLOQUE:
            texto += '   (se procesara por bloques, puede tardar)'
            color = '#FFB74D'
        self.lbl_region.setText(texto)
        self.lbl_region.setStyleSheet(f'color: {color}; padding: 2px;')

    # carga de imagen

    def _mostrar_imagen_cargada(self, arr: np.ndarray, nombre: str) -> None:
        arr = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        es_color = not (np.allclose(arr[..., 0], arr[..., 1])
                        and np.allclose(arr[..., 1], arr[..., 2]))
        self._imagen_orig = imagen_a_grises(arr)
        # el ruido es una propiedad de toda la foto: se mide una vez sobre una
        # zona central grande (en un recorte pequeno se confunde con textura)
        H0, W0 = self._imagen_orig.shape
        c = self._imagen_orig[max(0, (H0 - 1024) // 2):][:1024, max(0, (W0 - 1024) // 2):][:, :1024]
        self._sigma_imagen = estimar_ruido(c)
        self._imagen_color = arr if es_color else None
        self._nombre_imagen = nombre
        self.pila_orig.setCurrentIndex(1)
        self.visor_orig.establecer_imagen(arr if es_color else self._imagen_orig)
        self._resultado = None
        self._ya_amplio = False
        self.visor_recon.limpiar()
        self.pila_recon.setCurrentIndex(0)
        h, w = self._imagen_orig.shape
        self._actualizar_info_region()
        self.statusBar().showMessage(
            f"Imagen cargada ({'color' if es_color else 'grises'}): {nombre} ({w}x{h} px). "
            f"Ruido estimado {self._sigma_imagen:.4f} "
            f"({'se limpiara automaticamente' if self._sigma_imagen > UMBRAL_AUTO else 'no hace falta limpiar'}). "
            f"Paso 2: haga zoom con la rueda sobre la zona que quiere ampliar.")

    def cargar_imagen(self, ruta: str) -> None:
        """Carga una imagen desde disco (tambien usado por pruebas)."""
        arr = np.asarray(Image.open(ruta).convert('RGB'), dtype=np.float64) / 255.0
        self._mostrar_imagen_cargada(arr, os.path.basename(ruta))

    def _cargar_ejemplo(self, clave: str) -> None:
        try:
            from skimage import data
            arr = getattr(data, clave)().astype(np.float64) / 255.0
            self._mostrar_imagen_cargada(arr, f'ejemplo: {clave}')
        except Exception as exc:
            mostrar_error(self, 'No se pudo cargar el ejemplo',
                          f'{type(exc).__name__}: {exc}', traceback.format_exc())

    def _abrir_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, 'Abrir imagen', '',
            'Imagenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;Todos (*)')
        if ruta:
            self._abrir_ruta(ruta)

    def _abrir_ruta(self, ruta: str) -> None:
        try:
            self.cargar_imagen(ruta)
        except Exception as exc:
            mostrar_error(self, 'Error al cargar imagen', f'{type(exc).__name__}: {exc}',
                          traceback.format_exc())

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self._abrir_ruta(urls[0].toLocalFile())

    def _ajustar_vistas(self):
        self.visor_orig.ajustar_a_ventana()
        self.visor_recon.ajustar_a_ventana()

    def _tile_desde_viewport(self, avisar: bool = True):
        """
        Region visible del visor izquierdo: (tile, x, y, w, h). Si la imagen es
        a color el tile es RGB. Retorna None si no hay imagen o es muy pequena.
        """
        if self._imagen_orig is None:
            if avisar:
                QMessageBox.information(self, 'Primero abra una imagen',
                                        'Paso 1: abra una imagen (boton "Abrir imagen", '
                                        'arrastrela a la ventana o use Archivo > '
                                        'Imagenes de ejemplo).')
            return None
        region = self.visor_orig.obtener_viewport_imagen()
        if region is None:
            region = (0, 0, self._imagen_orig.shape[1], self._imagen_orig.shape[0])
        x, y, w, h = region
        fuente = self._imagen_color if self._imagen_color is not None else self._imagen_orig
        tile = fuente[y:y + h, x:x + w, ...]
        if min(tile.shape[0], tile.shape[1]) < 8:
            if avisar:
                QMessageBox.information(self, 'Region muy pequena',
                                        'La zona visible tiene menos de 8 px por lado. '
                                        'Aleje un poco el zoom.')
            return None
        return tile, x, y, w, h

    # hilos

    def _iniciar_hilo(self, funcion, al_terminar, texto: str) -> bool:
        if self._hilo is not None and self._hilo.isRunning():
            QMessageBox.information(self, 'Calculo en curso',
                                    'Ya hay un calculo en progreso. Espere o pulse Cancelar.')
            return False
        self._t_inicio = time.perf_counter()
        self._texto_tarea = texto
        self.barra_progreso.setRange(0, 0)
        self.barra_progreso.setFormat(texto)
        self.cont_progreso.setVisible(True)
        self.btn_cancelar.setEnabled(True)
        self.panel_params.deshabilitar(True)
        self.statusBar().showMessage(texto + '...')
        self._hilo = HiloCalculo(funcion)
        # conectar ANTES de start(): si no, un calculo muy rapido podria
        # emitir 'terminado' antes de que exista la conexion
        self._hilo.terminado.connect(al_terminar)
        self._hilo.error.connect(self._en_error)
        self._hilo.cancelado_sig.connect(self._en_cancelado)
        self._hilo.progreso.connect(self._en_progreso)
        self._hilo.finished.connect(self._en_fin_hilo)
        self._hilo.start()
        return True

    def _en_progreso(self, fraccion: float, texto: str):
        self.barra_progreso.setRange(0, 100)
        self.barra_progreso.setValue(int(round(100 * fraccion)))
        seg = time.perf_counter() - self._t_inicio
        self.barra_progreso.setFormat(f'{self._texto_tarea}: {texto}  (%p%, {seg:.0f} s)')

    def _cancelar(self):
        if self._hilo is not None and self._hilo.isRunning():
            self._hilo.cancelar()
            self.btn_cancelar.setEnabled(False)
            self.barra_progreso.setFormat('Cancelando...')

    def _en_cancelado(self):
        self.statusBar().showMessage('Calculo cancelado.')

    def _en_error(self, resumen: str, detalle: str):
        mostrar_error(self, 'Error en el calculo', resumen, detalle)
        self.statusBar().showMessage(f'Error: {resumen}')

    def _en_fin_hilo(self):
        self.cont_progreso.setVisible(False)
        self.panel_params.deshabilitar(False)

    # ampliacion compartida (Reconstruir)

    @staticmethod
    def _ampliar(tile: np.ndarray, factor: int, params: dict, progreso=None,
                 cancelado=None, **cambios) -> dict:
        """Amplia el tile con RZI y prepara las referencias (vecino y bicubica)."""
        res = upscale(tile, factor, progreso=progreso, cancelado=cancelado,
                      **kwargs_rzi(params, **cambios))
        H, W = res['imagen_recon'].shape[:2]
        if tile.ndim == 3:
            bic = resize_rgb_float(tile, W, H, Image.BICUBIC)
            nn = resize_rgb_float(tile, W, H, Image.NEAREST)
        else:
            bic = np.clip(resize_float(tile, W, H, Image.BICUBIC), 0, 1)
            nn = np.clip(resize_float(tile, W, H, Image.NEAREST), 0, 1)
        return {'res': res, 'display': res['imagen_recon'], 'bicubica': bic,
                'nn': nn, 'tile': tile}

    def _params_ampliar(self) -> dict:
        """Parametros del panel + el ruido medido sobre la imagen completa."""
        params = self.panel_params.obtener_params()
        params['sigma_ruido'] = getattr(self, '_sigma_imagen', None)
        return params

    def _verificar_tamano(self, tile: np.ndarray, factor: int) -> bool:
        """Avisa con un mensaje claro si la salida seria demasiado grande."""
        try:
            verificar_tamano(tile.shape[0], tile.shape[1], factor)
            return True
        except SalidaDemasiadoGrande as exc:
            QMessageBox.information(self, 'Region demasiado grande', str(exc))
            return False

    # composicion con inset opcional

    def _componer_con_inset(self, patch: np.ndarray, region: tuple) -> tuple:
        """
        Construye el contexto del visor derecho: imagen original completa con
        el resultado SR pegado como inset en la esquina mas lejana a 'region'.
        """
        tx, ty, tw, th = region
        orig = self._imagen_color if self._imagen_color is not None else self._imagen_orig
        H, W = orig.shape[:2]

        tx = max(0, min(tx, W - 1))
        ty = max(0, min(ty, H - 1))
        tw = max(1, min(tw, W - tx))
        th = max(1, min(th, H - ty))

        ph, pw = patch.shape[:2]
        aspect = pw / max(ph, 1)
        ih = max(1, round(self.INSET_FRAC * H))
        iw = max(1, round(ih * aspect))
        if iw > self.INSET_FRAC * W:
            iw = max(1, round(self.INSET_FRAC * W))
            ih = max(1, round(iw / aspect))

        cx = tx + tw / 2
        cy = ty + th / 2
        m = max(2, round(0.02 * min(W, H)))
        corners = (
            (m, m),
            (W - iw - m, m),
            (m, H - ih - m),
            (W - iw - m, H - ih - m),
        )
        ix, iy = max(corners, key=lambda c: (c[0] - cx) ** 2 + (c[1] - cy) ** 2)
        ix = int(max(0, min(ix, W - iw)))
        iy = int(max(0, min(iy, H - ih)))

        base = orig.copy()
        if base.ndim == 3 and patch.ndim == 2:
            patch_src = np.stack([patch] * 3, axis=-1)
        elif base.ndim == 2 and patch.ndim == 3:
            patch_src = imagen_a_grises(patch)
        else:
            patch_src = patch

        if patch_src.ndim == 3:
            patch_inset = resize_rgb_float(patch_src, iw, ih, Image.LANCZOS)
        else:
            patch_inset = resize_float(patch_src, iw, ih, Image.LANCZOS)
        base[iy:iy + ih, ix:ix + iw] = np.clip(patch_inset, 0.0, 1.0)

        return base, (tx, ty, tw, th), (ix, iy, iw, ih)

    @reportar_errores
    def _actualizar_visor_recon(self, *_):
        if not self._resultado or 'imagen_upscale' not in self._resultado:
            return
        self.pila_recon.setCurrentIndex(1)
        display = self._resultado['imagen_upscale']
        region = self._resultado.get('region')
        if self.chk_vista_inset.isChecked() and region is not None and self._imagen_orig is not None:
            contexto, reg, inset = self._componer_con_inset(display, region)
            self.visor_recon.establecer_imagen(contexto)
            self.visor_recon.marcar_region(*reg, color='#FF5252')
            self.visor_recon.marcar_region(*inset, color='#00E676')
        else:
            self.visor_recon.establecer_imagen(display)

    # Ampliar region (Reconstruir)

    def _reconstruir(self):
        info = self._tile_desde_viewport()
        if info is None:
            return
        tile, tx, ty, tw, th = info
        params = self._params_ampliar()
        factor = params['factor']
        if not self._verificar_tamano(tile, factor):
            return

        def tarea(progreso, cancelado):
            datos = self._ampliar(tile, factor, params, progreso=progreso, cancelado=cancelado)
            gris = lambda a: a if a.ndim == 2 else imagen_a_grises(a)
            peq = gris(datos['tile'])
            datos['metricas'] = {
                nombre: (nitidez_gradiente(gris(img)), consistencia_datos(gris(img), peq))
                for nombre, img in (('nn', datos['nn']), ('bicubica', datos['bicubica']),
                                    ('rzi', datos['display']))}
            datos.update(region=(tx, ty, tw, th), factor=factor)
            return datos

        self._iniciar_hilo(tarea, self._mostrar_reconstruccion,
                           f'Ampliando {tw}x{th} px x{factor}')

    @reportar_errores
    def _mostrar_reconstruccion(self, datos: dict):
        self._resultado = {
            'imagen_upscale': datos['display'],
            'res':            datos['res'],
            'region':         datos['region'],
        }
        self._ya_amplio = True
        self._actualizar_guia()
        res = datos['res']
        self._actualizar_visor_recon()
        fig = grafica_convergencia(res['historial'], res['omega'], res['error_datos'],
                                   compacta=True)
        self.panel_res.mostrar_figura(PanelResultados.TAB_CONV, fig)

        tx, ty, tw, th = datos['region']
        H, W = datos['display'].shape[:2]
        bloques = f"  |  {res.get('bloques', 1)} bloques" if res.get('por_bloques') else ''
        seg = time.perf_counter() - self._t_inicio
        self.statusBar().showMessage(
            f"RZI x{datos['factor']}: region {tw}x{th} -> {W}x{H} px en {seg:.1f} s  |  "
            f"iter SOR={res['iteraciones']}  omega={res['omega']:.3f}  "
            f"conv={'Si' if res['convergio'] else 'No'}{bloques}  |  "
            f"ruido estimado {res.get('sigma_estimado', 0):.4f}: "
            f"{TEXTO_LIMPIEZA.get(res.get('limpieza_aplicada', 'no'))}")
        self._mostrar_dialogo_detalle(datos)

    def _mostrar_dialogo_detalle(self, datos: dict):
        dialogo = QDialog(self)
        dialogo.setWindowTitle('Comparacion en detalle: zoom nativo vs bicubica vs RZI')
        dialogo.setStyleSheet(ESTILO)
        dialogo.resize(1150, 580)
        layout = QVBoxLayout(dialogo)

        lbl = QLabel(
            'Las tres imagenes tienen el mismo tamano y el zoom esta sincronizado: '
            'use la rueda sobre cualquiera. Nitidez: magnitud media del gradiente '
            '(mayor = bordes mas marcados). Consistencia: PSNR entre la ampliacion '
            'reducida de nuevo y la original (mayor = respeta mejor los datos reales).')
        lbl.setWordWrap(True)
        lbl.setStyleSheet('color: #B0B0B0; padding: 4px;')
        layout.addWidget(lbl)
        res = datos['res']
        aplicada = res.get('limpieza_aplicada', 'no')
        lbl_ruido = QLabel(
            f"Ruido estimado de la entrada: {res.get('sigma_estimado', 0):.4f}. "
            + ('No hizo falta limpiarla.' if aplicada == 'no' else
               f"Se aplico {TEXTO_LIMPIEZA[aplicada]} (variacion total) antes de ampliar; "
               'las columnas de vecino y bicubica muestran la entrada original. '
               'Por eso la nitidez y la consistencia de RZI salen mas bajas: el ruido '
               'tambien cuenta como "borde" y ya no se copia (es lo buscado).'))
        lbl_ruido.setWordWrap(True)
        lbl_ruido.setStyleSheet('color: #FFB74D; padding: 2px 4px;' if aplicada != 'no'
                                else 'color: #81C784; padding: 2px 4px;')
        layout.addWidget(lbl_ruido)

        fila = QHBoxLayout()
        visores = []
        for clave, titulo, color, img in (
                ('nn', 'Zoom nativo (vecino mas proximo)', '#F44336', datos['nn']),
                ('bicubica', 'Bicubica', '#FF9800', datos['bicubica']),
                ('rzi', f"RZI x{datos['factor']}", '#4CAF50', datos['display'])):
            nit, cons = datos['metricas'][clave]
            col = QVBoxLayout()
            t = QLabel(f'{titulo}\nNitidez {nit:.4f}  |  Consistencia {cons:.1f} dB')
            t.setAlignment(Qt.AlignCenter)
            t.setStyleSheet(f'color: {color}; font-weight: bold; padding: 2px;')
            v = VisorImagen()
            v.setMinimumSize(320, 320)
            v.establecer_imagen(img)
            visores.append(v)
            col.addWidget(t)
            col.addWidget(v)
            fila.addLayout(col)
        layout.addLayout(fila)

        def sincronizar(origen):
            for v in visores:
                if v is not origen:
                    v.copiar_estado_zoom(origen)
        for v in visores:
            v.vista_cambiada.connect(lambda o=v: sincronizar(o))

        botones = QHBoxLayout()
        btn_guardar = QPushButton('Guardar imagen RZI...')
        btn_guardar.clicked.connect(self._guardar_imagen)
        btn = QPushButton('Cerrar')
        btn.clicked.connect(dialogo.accept)
        botones.addWidget(btn_guardar)
        botones.addStretch()
        botones.addWidget(btn)
        layout.addLayout(botones)
        dialogo.exec()

    # Medir calidad (comparar con verdad de campo)

    def _comparar(self):
        info = self._tile_desde_viewport()
        if info is None:
            return
        tile, tx, ty, tw, th = info
        params = self.panel_params.obtener_params()
        factor = params['factor']
        lado_corto = min(th, tw)
        if lado_corto < self.MIN_LADO_EVAL:
            QMessageBox.information(
                self, 'Region pequena',
                f'Para medir la calidad la zona visible debe tener al menos '
                f'{self.MIN_LADO_EVAL} px por lado. Aleje un poco el zoom.')
            return

        def tarea(progreso, cancelado):
            gris = imagen_a_grises(tile) if tile.ndim == 3 else tile
            # en regiones grandes se evalua el recorte con mas detalle (como
            # maximo LADO_EVAL_MAX px por lado): rapido y con formas exactas
            recortada = False
            if max(gris.shape) > self.LADO_EVAL_MAX:
                y0, x0, lado = elegir_recorte(gris, min(self.LADO_EVAL_MAX, min(gris.shape)))
                gris = gris[y0:y0 + lado, x0:x0 + lado]
                recortada = True
            corto = min(gris.shape)
            # si la zona es menor que 16*factor, se adapta el factor de evaluacion
            factor_eval = factor if corto >= 16 * factor else max(2, min(factor, corto // 16))

            hr, lr = degradar(gris, factor_eval)
            bic = interpolacion_bicubica(lr, factor_eval)
            lan = interpolacion_lanczos(lr, factor_eval)
            p1 = (lambda f, t: progreso(0.5 * f, f'RZI fiel: {t}')) if progreso else None
            res = upscale(lr, factor_eval, progreso=p1, cancelado=cancelado,
                          **kwargs_rzi(params, realce=0.0))
            fuerza = params['realce'] if params['realce'] > 0 else 0.6
            p2 = (lambda f, t: progreso(0.5 + 0.5 * f, f'RZI con realce: {t}')) if progreso else None
            res_r = upscale(lr, factor_eval, progreso=p2, cancelado=cancelado,
                            **kwargs_rzi(params, realce=fuerza))
            nombre_r = f'RZI + realce {fuerza:.1f}'
            candidatas = {'Bicubica': bic, 'Lanczos': lan,
                          'RZI (fiel)': res['imagen_recon'], nombre_r: res_r['imagen_recon']}
            tabla = tabla_comparativa(hr, candidatas, modelos_rzi=('RZI (fiel)', nombre_r))
            return {
                'hr':          hr,
                'candidatas':  candidatas,
                'tabla':       tabla,
                'res':         res,
                'recorte':     elegir_recorte(hr, max(24, min(hr.shape) // 3)),
                'factor_eval': factor_eval,
                'factor_ped':  factor,
                'recortada':   recortada,
            }

        self._iniciar_hilo(tarea, self._mostrar_comparacion,
                           f'Midiendo calidad (zona {tw}x{th}, x{factor})')

    @reportar_errores
    def _mostrar_comparacion(self, datos: dict):
        self._resultado = {
            'tabla':          datos['tabla'],
            'res':            datos['res'],
            'imagen_upscale': datos['candidatas']['RZI (fiel)'],
        }
        fe = datos['factor_eval']
        fig = grafica_comparacion(
            datos['hr'], datos['candidatas'], datos['tabla'],
            recorte=datos['recorte'],
            factor_eval=fe if fe != datos['factor_ped'] else None, compacta=True)
        self.panel_res.mostrar_figura(PanelResultados.TAB_COMP, fig)
        self.panel_res.mostrar_tabla(datos['tabla'])
        filas = {f['modelo']: f for f in datos['tabla']}
        ganancia = filas['RZI (fiel)']['psnr'] - filas['Bicubica']['psnr']
        extra = f' (factor de evaluacion x{fe})' if fe != datos['factor_ped'] else ''
        if datos['recortada']:
            h, w = datos['hr'].shape
            extra += f' (evaluado sobre el recorte con mas detalle, {w}x{h} px)'
        self.statusBar().showMessage(
            f'Calidad medida{extra}: RZI (fiel) {ganancia:+.2f} dB sobre la bicubica.')

    # Barrido de omega

    def _barrido_omega(self):
        if self._imagen_orig is None:
            self._tile_desde_viewport()
            return
        info = self._tile_desde_viewport(avisar=False)
        tile = info[0] if info else self._imagen_orig
        if tile.ndim == 3:
            tile = imagen_a_grises(tile)
        # recorte central de hasta 256x256 para que el barrido sea rapido
        H, W = tile.shape
        lado_h, lado_w = min(H, 256), min(W, 256)
        y0, x0 = (H - lado_h) // 2, (W - lado_w) // 2
        tile = tile[y0:y0 + lado_h, x0:x0 + lado_w]
        factor = self.panel_params.obtener_params()['factor']

        def tarea(progreso, cancelado):
            problema = construir_problema(tile, factor)
            return barrido_omega(problema, tol=1e-3, max_iter=500,
                                 progreso=progreso, cancelado=cancelado)

        self._iniciar_hilo(tarea, self._en_barrido_terminado,
                           f'Barrido de omega ({lado_w}x{lado_h} px, factor {factor})')

    @reportar_errores
    def _en_barrido_terminado(self, r: dict):
        fig = grafica_barrido_omega(r['omegas'], r['iteraciones'], r['omega_teorico'],
                                    r['omega_malla_completa'], compacta=True)
        self.panel_res.mostrar_figura(PanelResultados.TAB_OMEGA, fig)
        self.statusBar().showMessage(
            f"Barrido omega | minimo medido={r['omega_empirico']:.3f} | "
            f"teorico con mascara={r['omega_teorico']:.3f} | "
            f"formula malla completa={r['omega_malla_completa']:.3f}")

    # Benchmark

    def _benchmark(self):
        if self._imagen_orig is None:
            self._tile_desde_viewport()
            return
        img = self._imagen_orig
        if min(img.shape) < 64:
            QMessageBox.information(self, 'Imagen pequena',
                                    'El benchmark necesita una imagen de al menos 64x64 px.')
            return

        def tarea(progreso):
            resultados = []
            lados = [lado for lado in (64, 128, 256) if min(img.shape) >= lado]
            for k, lado in enumerate(lados):
                progreso(k / len(lados), f'malla {lado}x{lado}')
                cy, cx = img.shape[0] // 2, img.shape[1] // 2
                t = img[cy - lado // 2:cy + lado // 2, cx - lado // 2:cx + lado // 2]
                resultados.append(benchmark_numba_vs_python(construir_problema(t, 4), n_iter=20))
            return resultados

        self._iniciar_hilo(tarea, self._en_benchmark_terminado,
                           'Benchmark Numba vs Python (Python puro tarda unos segundos)')

    @reportar_errores
    def _en_benchmark_terminado(self, resultados: list):
        self.panel_res.mostrar_figura(PanelResultados.TAB_BENCH, grafica_benchmark(resultados, compacta=True))
        r = resultados[-1]
        self.statusBar().showMessage(
            f"Benchmark {r['forma'][0]}x{r['forma'][1]}: Numba {r['aceleracion']:.0f} "
            f"veces mas rapido (mismas {r['iteraciones']} iteraciones).")

    # Grafica en grande

    @reportar_errores
    def _ver_figura_grande(self, idx: int):
        fig = self.panel_res.figuras.get(idx)
        if fig is None:
            return
        dialogo = QDialog(self)
        dialogo.setWindowTitle(self.panel_res.tabs.tabText(idx))
        dialogo.setStyleSheet(ESTILO)
        dialogo.resize(1200, 800)
        lay = QVBoxLayout(dialogo)
        visor = VisorImagen()
        visor.establecer_imagen(figura_a_imagen(fig, dpi=220))
        lay.addWidget(visor)
        btn = QPushButton('Cerrar')
        btn.clicked.connect(dialogo.accept)
        lay.addWidget(btn)
        dialogo.exec()

    # Lupa

    def _toggle_modo_lupa(self, activado: bool):
        if activado:
            if self._imagen_orig is None:
                self.panel_params.btn_lupa.setChecked(False)
                self._tile_desde_viewport()
                return
            self._modo_lupa_activo = True
            self.visor_orig.setCursor(Qt.CrossCursor)
            self.visor_orig.mouse_moved.connect(self._actualizar_lupa)
            self.statusBar().showMessage(
                'Lupa activa: mueva el mouse sobre la imagen izquierda. La derecha muestra x3.')
        else:
            self._modo_lupa_activo = False
            self.visor_orig.unsetCursor()
            try:
                self.visor_orig.mouse_moved.disconnect(self._actualizar_lupa)
            except (RuntimeError, TypeError):
                pass
            self.statusBar().showMessage('Lupa desactivada.')

    def _actualizar_lupa(self, x_img: int, y_img: int):
        if not self._modo_lupa_activo or self._imagen_orig is None:
            return
        ahora = time.time()
        if ahora - self._ultima_lupa < 0.12:
            return
        self._ultima_lupa = ahora
        if self._hilo_lupa is not None and self._hilo_lupa.isRunning():
            return

        vp = self.visor_orig.obtener_viewport_imagen()
        radio = 60 if vp is None else max(20, min(int(0.18 * min(vp[2], vp[3])), 80))
        H, W = self._imagen_orig.shape
        x0, y0 = max(0, x_img - radio), max(0, y_img - radio)
        x1, y1 = min(W, x_img + radio), min(H, y_img + radio)
        fuente = self._imagen_color if self._imagen_color is not None else self._imagen_orig
        tile = fuente[y0:y1, x0:x1, ...]
        if min(tile.shape[0], tile.shape[1]) < 8:
            return
        params = self._params_ampliar()

        def tarea():
            return upscale(tile, 3, **kwargs_rzi(params, ciclos=4, sor_iters=60,
                                                 bp_final=3))['imagen_recon']

        self._hilo_lupa = HiloCalculo(tarea)
        self._hilo_lupa.terminado.connect(self._mostrar_lupa)
        self._hilo_lupa.error.connect(
            lambda resumen, detalle: (registrar_error(detalle),
                                      self.statusBar().showMessage(f'Error en lupa: {resumen}')))
        self._hilo_lupa.start()

    @reportar_errores
    def _mostrar_lupa(self, imagen):
        if not self._modo_lupa_activo:
            return
        self.pila_recon.setCurrentIndex(1)
        self.visor_recon.establecer_imagen(imagen)
        self.statusBar().showMessage(f'Lupa RZI x3: {imagen.shape[1]}x{imagen.shape[0]} px')

    # Animacion didactica del proceso completo de reconstruccion

    def _animar_convergencia(self):
        info = self._tile_desde_viewport()
        if info is None:
            return
        tile = info[0]
        params = self._params_ampliar()
        factor = params['factor']
        if tile.shape[0] * tile.shape[1] * factor * factor > MAX_PIXELES_BLOQUE:
            QMessageBox.information(
                self, 'Haga zoom primero',
                'La animacion paso a paso necesita una zona mas pequena (salida de hasta '
                f'{MAX_PIXELES_BLOQUE / 1e6:.0f} MP). Haga zoom sobre la zona que le '
                'interesa y vuelva a intentar.')
            return

        def tarea(progreso, cancelado):
            return generar_frames_pipeline(tile, factor, progreso=progreso,
                                           cancelado=cancelado, **kwargs_rzi(params))

        self._iniciar_hilo(tarea, self._mostrar_dialogo_animacion,
                           'Generando la animacion del proceso')

    @reportar_errores
    def _mostrar_dialogo_animacion(self, gen: dict):
        frames = gen.get('frames', [])
        if not frames:
            QMessageBox.warning(self, 'Error', 'No se generaron frames.')
            return

        pasos = gen['pasos']
        pasos_de_frame = gen['pasos_de_frame']
        subtitulos = gen['subtitulos']
        n_frames = len(frames)
        n_pasos = len(pasos)

        dialogo = QDialog(self)
        dialogo.setWindowTitle('Proceso completo de reconstruccion RZI')
        dialogo.resize(1220, 700)
        dialogo.setStyleSheet(ESTILO)

        root = QVBoxLayout(dialogo)
        root.setSpacing(6)

        lbl_info = QLabel(
            f"Frames: {n_frames}  |  Pasos: {n_pasos}  |  "
            f"Ciclos: {gen['ciclos']}  |  "
            f"Omega: {gen['omega']:.3f}  |  "
            f"Iters SOR: {gen['iteraciones']}  |  "
            f"Tiempo SOR: {gen['tiempo']:.2f}s"
        )
        lbl_info.setStyleSheet('color: #2196F3; font-weight: bold; padding: 4px;')
        root.addWidget(lbl_info)

        cuerpo = QHBoxLayout()
        cuerpo.setSpacing(8)

        lista_pasos = QListWidget()
        lista_pasos.setFixedWidth(240)
        for i, nombre in enumerate(pasos):
            lista_pasos.addItem(QListWidgetItem(f'{i + 1}. {nombre}'))

        col_lista = QVBoxLayout()
        lbl_lista = QLabel('Etapas del pipeline (clic para saltar)')
        lbl_lista.setAlignment(Qt.AlignCenter)
        lbl_lista.setStyleSheet('color: #90A4AE; font-size: 11px; padding-bottom: 2px;')
        col_lista.addWidget(lbl_lista)
        col_lista.addWidget(lista_pasos)
        cuerpo.addLayout(col_lista)

        visor_anim = VisorImagen()
        visor_bic = VisorImagen()
        visor_bic.establecer_imagen(gen['bicubica'])
        visor_anim.vista_cambiada.connect(lambda: visor_bic.copiar_estado_zoom(visor_anim))
        visor_bic.vista_cambiada.connect(lambda: visor_anim.copiar_estado_zoom(visor_bic))

        for titulo, color, visor in (
                ('RZI (paso a paso)', '#4CAF50', visor_anim),
                ('Bicubica (referencia fija)', '#FF9800', visor_bic)):
            col = QVBoxLayout()
            lbl_v = QLabel(titulo)
            lbl_v.setAlignment(Qt.AlignCenter)
            lbl_v.setStyleSheet(f'color: {color}; font-weight: bold; padding-bottom: 2px;')
            col.addWidget(lbl_v)
            col.addWidget(visor, stretch=1)
            cuerpo.addLayout(col, stretch=1)

        root.addLayout(cuerpo, stretch=1)

        lbl_leyenda = QLabel('')
        lbl_leyenda.setWordWrap(True)
        lbl_leyenda.setAlignment(Qt.AlignCenter)
        lbl_leyenda.setStyleSheet(
            'color: #E0E0E0; background: #2A2A2A; border-radius: 4px; '
            'padding: 6px; font-size: 11px;')
        lbl_leyenda.setMinimumHeight(44)
        root.addWidget(lbl_leyenda)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, n_frames - 1)
        slider.setValue(0)
        root.addWidget(slider)

        lbl_frame = QLabel(f'Frame 1 / {n_frames}')
        lbl_frame.setAlignment(Qt.AlignCenter)
        lbl_frame.setStyleSheet('color: #808080; font-size: 10px;')
        root.addWidget(lbl_frame)

        ctrl = QHBoxLayout()
        lbl_vel = QLabel('Velocidad:')
        lbl_vel.setStyleSheet('color: #B0B0B0;')
        combo_vel = QComboBox()
        VELOCIDADES = {'Muy lento': 1200, 'Lento': 800, 'Normal': 450, 'Rapido': 220}
        combo_vel.addItems(list(VELOCIDADES))
        combo_vel.setCurrentText('Lento')

        btn_play = QPushButton('Reproducir')
        btn_pause = QPushButton('Pausar')
        btn_close = QPushButton('Cerrar')
        ctrl.addWidget(lbl_vel)
        ctrl.addWidget(combo_vel)
        ctrl.addStretch()
        for b in (btn_play, btn_pause, btn_close):
            b.setMinimumHeight(32)
            ctrl.addWidget(b)
        root.addLayout(ctrl)

        primera = [True]
        HOLD_PASO_MULT = 2

        def _saltar_a_paso(row):
            for fi, pi in enumerate(pasos_de_frame):
                if pi == row:
                    slider.setValue(fi)
                    break

        lista_pasos.currentRowChanged.connect(_saltar_a_paso)

        def actualizar_frame(idx):
            visor_anim.establecer_imagen(frames[idx], ajustar=primera[0])
            primera[0] = False
            paso_idx = pasos_de_frame[idx]
            paso_name = pasos[paso_idx]
            sub = subtitulos[idx]
            sufijo = '  [Identica a Ampliar region]' if idx == n_frames - 1 else ''
            lbl_leyenda.setText(f'Paso {paso_idx + 1}/{n_pasos}: {paso_name}  |  {sub}{sufijo}')
            lbl_frame.setText(f'Frame {idx + 1} / {n_frames}')
            lista_pasos.blockSignals(True)
            lista_pasos.setCurrentRow(paso_idx)
            lista_pasos.blockSignals(False)

        slider.valueChanged.connect(actualizar_frame)

        timer = QTimer(dialogo)

        def intervalo_para(idx: int) -> int:
            base = VELOCIDADES[combo_vel.currentText()]
            es_nuevo = idx == 0 or pasos_de_frame[idx] != pasos_de_frame[idx - 1]
            return base * HOLD_PASO_MULT if es_nuevo else base

        def avanzar():
            idx = slider.value()
            if idx >= n_frames - 1:
                timer.stop()
                return
            next_idx = idx + 1
            slider.setValue(next_idx)
            timer.setInterval(intervalo_para(next_idx))

        def iniciar():
            if slider.value() >= n_frames - 1:
                slider.setValue(0)
            timer.start(VELOCIDADES[combo_vel.currentText()])

        def cambiar_velocidad():
            if timer.isActive():
                timer.setInterval(intervalo_para(slider.value()))

        timer.timeout.connect(avanzar)
        combo_vel.currentTextChanged.connect(cambiar_velocidad)
        btn_play.clicked.connect(iniciar)
        btn_pause.clicked.connect(timer.stop)
        btn_close.clicked.connect(dialogo.accept)
        dialogo.finished.connect(timer.stop)

        actualizar_frame(0)
        iniciar()
        dialogo.exec()

    # Ayuda y diagnostico

    def _mostrar_ayuda(self):
        self._dialogo_texto('Como usar RZI', GUIA_USO)

    def _dialogo_texto(self, titulo: str, texto: str, copiar: bool = False):
        dialogo = QDialog(self)
        dialogo.setWindowTitle(titulo)
        dialogo.setStyleSheet(ESTILO)
        dialogo.resize(760, 560)
        lay = QVBoxLayout(dialogo)
        caja = QPlainTextEdit(texto)
        caja.setReadOnly(True)
        lay.addWidget(caja)
        botones = QHBoxLayout()
        if copiar:
            b_copiar = QPushButton('Copiar al portapapeles')
            b_copiar.clicked.connect(lambda: QApplication.clipboard().setText(texto))
            botones.addWidget(b_copiar)
        botones.addStretch()
        b = QPushButton('Cerrar')
        b.clicked.connect(dialogo.accept)
        botones.addWidget(b)
        lay.addLayout(botones)
        dialogo.exec()

    def _diagnostico(self):
        def tarea(progreso):
            return ejecutar_diagnostico(progreso=progreso)
        self._iniciar_hilo(tarea, self._mostrar_diagnostico, 'Diagnostico')

    @reportar_errores
    def _mostrar_diagnostico(self, resultado):
        texto, ok = resultado
        self.statusBar().showMessage('Diagnostico: todo funciona.' if ok else
                                     'Diagnostico: hay errores (ver informe).')
        self._dialogo_texto('Diagnostico de RZI', texto, copiar=True)

    def _mostrar_log(self):
        try:
            with open(RUTA_LOG, encoding='utf-8') as fh:
                texto = fh.read()[-20000:] or '(vacio)'
        except OSError:
            texto = '(todavia no hay errores registrados)'
        self._dialogo_texto('Registro de errores', f'{RUTA_LOG}\n\n{texto}', copiar=True)

    # Exportacion

    def _guardar_imagen(self):
        img = (self._resultado or {}).get('imagen_upscale')
        if img is None:
            QMessageBox.information(self, 'Sin resultado',
                                    'Primero amplie una region o mida la calidad.')
            return
        base = os.path.splitext(self._nombre_imagen.replace('ejemplo: ', ''))[0] or 'imagen'
        ruta, _ = QFileDialog.getSaveFileName(self, 'Guardar imagen ampliada',
                                              f'{base}_rzi.png', 'PNG (*.png);;JPEG (*.jpg)')
        if ruta:
            try:
                guardar_imagen(img, ruta)
                self.statusBar().showMessage(f'Imagen guardada en {ruta}')
            except Exception as exc:
                mostrar_error(self, 'No se pudo guardar', f'{type(exc).__name__}: {exc}',
                              traceback.format_exc())

    def _guardar_grafica(self):
        fig = self.panel_res.figura_actual()
        if fig is None:
            QMessageBox.information(self, 'Sin grafica',
                                    'La pestana activa todavia no tiene grafica.')
            return
        ruta, _ = QFileDialog.getSaveFileName(self, 'Guardar grafica', 'grafica.png',
                                              'PNG (*.png);;PDF (*.pdf)')
        if ruta:
            fig.savefig(ruta, dpi=150, bbox_inches='tight')
            self.statusBar().showMessage(f'Grafica guardada en {ruta}')

    def _exportar_csv(self):
        if not self._resultado or 'tabla' not in self._resultado:
            QMessageBox.information(self, 'Sin datos', 'Pulse "Medir calidad (comparar)" primero.')
            return
        ruta, _ = QFileDialog.getSaveFileName(self, 'Exportar CSV', 'resultados.csv',
                                              'CSV (*.csv)')
        if ruta:
            exportar_csv(self._resultado['tabla'], self._resultado.get('res', {}), ruta)
            self.statusBar().showMessage(f'Datos exportados a {ruta}')
