"""
Capa 1 - Ventana principal de la aplicacion RZI.

Disposicion:
    [parametros] | [visor original] [visor ampliado] | [resultados]

Modos:
  Reconstruir  : amplia la region visible del visor izquierdo con el modelo
                 RZI (back-projection + Poisson screened resuelto con SOR
                 Red-Black). Muestra zoom nativo vs bicubica vs RZI, con
                 opcion de vista contextual con inset en el visor derecho.
  Comparar     : toma la region visible como verdad de campo, la reduce a
                 proposito y la vuelve a ampliar con bicubica, Lanczos y RZI,
                 midiendo PSNR, SSIM, GMSD y nitidez.
  Barrido omega, Benchmark, Lupa y Animacion: herramientas de analisis y
                 animacion didactica paso a paso del pipeline RZI.
"""

import os
import time

import numpy as np
from PIL import Image

import matplotlib
matplotlib.use('Agg')   # las graficas se dibujan a imagen y se muestran en un visor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox, QGroupBox, QSplitter,
    QProgressBar, QToolBar, QTabWidget, QScrollArea, QGridLayout, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QDialog, QSlider, QComboBox,
    QListWidget, QListWidgetItem,
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QAction

from ui.zoom_viewer import VisorImagen
from core.poisson_builder import construir_problema, imagen_a_grises
from core.sor_solver import (barrido_omega, benchmark_numba_vs_python,
                             NUMBA_DISPONIBLE, _calentamiento)
from core.upscaler import upscale, generar_frames_pipeline, PARAMS_DEFECTO
from core.image_ops import resize_float, resize_rgb_float
from services.metrics import (degradar, interpolacion_bicubica,
                              interpolacion_lanczos, tabla_comparativa,
                              nitidez_gradiente, consistencia_datos)
from services.exporter import (guardar_imagen, exportar_csv, grafica_convergencia,
                               grafica_barrido_omega, grafica_comparacion,
                               grafica_benchmark, elegir_recorte)


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
QStatusBar  { background-color: #1A1A1A; color: #B0B0B0; border-top: 1px solid #3A3A3A; }
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
"""


def figura_a_imagen(fig) -> np.ndarray:
    """Dibuja una figura de matplotlib a un array RGB float [0,1]."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].astype(np.float64) / 255.0


class HiloCalculo(QThread):
    """Ejecuta una funcion en segundo plano para no congelar la interfaz."""
    terminado = Signal(object)
    error = Signal(str)

    def __init__(self, funcion, *args, **kwargs):
        super().__init__()
        self._fn = funcion
        self._args = args
        self._kw = kwargs

    def run(self):
        try:
            self.terminado.emit(self._fn(*self._args, **self._kw))
        except Exception as exc:   # se muestra al usuario, no se silencia
            self.error.emit(f'{type(exc).__name__}: {exc}')


class PanelParametros(QWidget):
    reconstruir_solicitado = Signal()
    comparar_solicitado = Signal()
    barrido_omega_solicitado = Signal()
    benchmark_solicitado = Signal()
    lupa_solicitado = Signal(bool)
    animar_solicitado = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(240)
        self.setMaximumWidth(300)
        self._construir()

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
        lay.setSpacing(10)

        tit = QLabel('Parametros')
        tit.setFont(QFont('Segoe UI', 13, QFont.Bold))
        tit.setStyleSheet('color: #2196F3; padding: 4px;')
        lay.addWidget(tit)

        # Ampliacion
        g_amp = QGroupBox('Ampliacion')
        gl = QGridLayout(g_amp)
        gl.addWidget(QLabel('Factor:'), 0, 0)
        self.spin_factor = self._spin(2, 12, 4, 1, tooltip=(
            'Cuantas veces se agranda cada lado (2..12). Factor 4: cada pixel '
            'original se convierte en un bloque de 4x4.'))
        gl.addWidget(self.spin_factor, 0, 1)
        gl.addWidget(QLabel('Realce perceptual:'), 1, 0)
        self.spin_realce = self._spin(0.0, 1.0, 0.0, 0.1, 2, tooltip=(
            '0 = salida mas fiel (mejores metricas). Valores mayores marcan '
            'mas los bordes a la vista, a costa de bajar un poco el PSNR.'))
        gl.addWidget(self.spin_realce, 1, 1)
        lay.addWidget(g_amp)

        # Modelo RZI
        g_mod = QGroupBox('Modelo RZI (Poisson screened)')
        ml = QGridLayout(g_mod)
        ml.addWidget(QLabel('Lambda (fidelidad):'), 0, 0)
        self.spin_lam = self._spin(0.05, 5.0, d['lam'], 0.05, 2, tooltip=(
            'Peso del termino lambda*(u - t). Pequeno: manda el campo de '
            'gradientes (bordes firmes). Grande: manda la estimacion t.'))
        ml.addWidget(self.spin_lam, 0, 1)
        ml.addWidget(QLabel('Beta (bordes):'), 1, 0)
        self.spin_beta = self._spin(1.0, 3.0, d['beta'], 0.1, 2, tooltip=(
            'Ganancia maxima de los gradientes en bordes reales. '
            '1 = sin amplificacion.'))
        ml.addWidget(self.spin_beta, 1, 1)
        ml.addWidget(QLabel('Tau (umbral):'), 2, 0)
        self.spin_tau = self._spin(0.005, 0.3, d['tau'], 0.005, 3, tooltip=(
            'Gradientes menores que tau casi no se amplifican (evita '
            'realzar ruido).'))
        ml.addWidget(self.spin_tau, 2, 1)
        ml.addWidget(QLabel('Sigma (suavizado):'), 3, 0)
        self.spin_sigma = self._spin(0.0, 2.0, d['sigma'], 0.1, 2, tooltip=(
            'Suavizado Gaussiano antes de medir gradientes.'))
        ml.addWidget(self.spin_sigma, 3, 1)
        ml.addWidget(QLabel('Ciclos:'), 4, 0)
        self.spin_ciclos = self._spin(1, 30, d['ciclos'], 1, tooltip=(
            'Veces que se alterna back-projection + solver SOR.'))
        ml.addWidget(self.spin_ciclos, 4, 1)
        ml.addWidget(QLabel('Back-proj. final:'), 5, 0)
        self.spin_bp = self._spin(0, 20, d['bp_final'], 1, tooltip=(
            'Pasadas finales de back-projection (consistencia con la '
            'imagen original).'))
        ml.addWidget(self.spin_bp, 5, 1)
        ml.addWidget(QLabel('Anti-ringing:'), 6, 0)
        self.chk_anti_ringing = QCheckBox()
        self.chk_anti_ringing.setToolTip(
            'Recorta al rango local de la bicubica. Quita halos pero suaviza '
            'un poco (baja ~0.1 dB).')
        ml.addWidget(self.chk_anti_ringing, 6, 1)
        lay.addWidget(g_mod)

        # Solver
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
        lay.addWidget(g_sol)

        # Acciones
        self.btn_reconstruir = QPushButton('Reconstruir (RZI)')
        self.btn_reconstruir.setObjectName('btn_principal')
        self.btn_reconstruir.setMinimumHeight(36)
        self.btn_reconstruir.clicked.connect(self.reconstruir_solicitado)
        lay.addWidget(self.btn_reconstruir)

        self.btn_comparar = QPushButton('Comparar modelos (metricas)')
        self.btn_comparar.clicked.connect(self.comparar_solicitado)
        self.btn_barrido = QPushButton('Barrido Omega')
        self.btn_barrido.clicked.connect(self.barrido_omega_solicitado)
        self.btn_benchmark = QPushButton('Benchmark Numba vs Python')
        self.btn_benchmark.clicked.connect(self.benchmark_solicitado)
        self.btn_lupa = QPushButton('Modo Lupa (tiempo real)')
        self.btn_lupa.setCheckable(True)
        self.btn_lupa.setToolTip('Amplia x3 la zona bajo el cursor con el modelo RZI')
        self.btn_lupa.toggled.connect(self.lupa_solicitado)
        self.btn_animar = QPushButton('Animar convergencia')
        self.btn_animar.clicked.connect(self.animar_solicitado)
        for b in (self.btn_comparar, self.btn_barrido, self.btn_benchmark,
                  self.btn_lupa, self.btn_animar):
            b.setMinimumHeight(30)
            lay.addWidget(b)

        color = '#4CAF50' if NUMBA_DISPONIBLE else '#FF9800'
        texto = 'Numba: activo' if NUMBA_DISPONIBLE else 'Numba: no disponible (lento)'
        lbl_nb = QLabel(texto)
        lbl_nb.setStyleSheet(f'color: {color}; font-size: 9pt; padding: 4px;')
        lay.addWidget(lbl_nb)
        lay.addStretch()

    def obtener_params(self) -> dict:
        return {
            'factor':       self.spin_factor.value(),
            'realce':       self.spin_realce.value(),
            'lam':          self.spin_lam.value(),
            'beta':         self.spin_beta.value(),
            'tau':          self.spin_tau.value(),
            'sigma':        self.spin_sigma.value(),
            'ciclos':       self.spin_ciclos.value(),
            'bp_final':     self.spin_bp.value(),
            'anti_ringing': self.chk_anti_ringing.isChecked(),
            'tol':          self.spin_tol.value(),
            'sor_iters':    self.spin_iters.value(),
            'omega':        None if self.chk_omega_auto.isChecked() else self.spin_omega.value(),
        }

    def deshabilitar(self, flag: bool):
        for btn in (self.btn_reconstruir, self.btn_comparar, self.btn_barrido,
                    self.btn_benchmark, self.btn_animar):
            btn.setEnabled(not flag)


def kwargs_rzi(p: dict, **cambios) -> dict:
    """Traduce los parametros del panel a los argumentos de core.upscaler.upscale."""
    kw = dict(lam=p['lam'], beta=p['beta'], tau=p['tau'], sigma=p['sigma'],
              ciclos=p['ciclos'], bp_final=p['bp_final'], sor_iters=p['sor_iters'],
              tol=p['tol'], omega=p['omega'], realce=p['realce'],
              usar_anti_ringing=p['anti_ringing'])
    kw.update(cambios)
    return kw


class PanelResultados(QWidget):
    """Pestanas con las graficas (como imagen con zoom) y la tabla de metricas."""

    TAB_COMP, TAB_CONV, TAB_OMEGA, TAB_BENCH = range(4)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(460)
        self.figuras = {}
        self._construir()

    def _construir(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        self.visores = {}
        tab_comp = QWidget()
        lc = QVBoxLayout(tab_comp)
        self.visores[self.TAB_COMP] = VisorImagen()
        lc.addWidget(self.visores[self.TAB_COMP], 3)
        nota = QLabel(
            'Metricas contra la verdad de campo. PSNR, SSIM y nitidez: mayor es mejor. '
            'GMSD: menor es mejor. Con Realce perceptual > 0 los bordes se ven mas '
            'marcados a costa de bajar ligeramente el PSNR.')
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
            self.visores[idx] = VisorImagen()
            lw.addWidget(self.visores[idx])
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
        anterior = self.figuras.get(idx)
        if anterior is not None and anterior is not fig:
            plt.close(anterior)
        self.figuras[idx] = fig
        self.tabs.setCurrentIndex(idx)   # primero visible, para ajustar al tamano real
        self.visores[idx].establecer_imagen(figura_a_imagen(fig))

    def figura_actual(self):
        return self.figuras.get(self.tabs.currentIndex())

    def mostrar_tabla(self, tabla: list) -> None:
        self.tabla.setRowCount(0)
        for fila in tabla:
            r = self.tabla.rowCount()
            self.tabla.insertRow(r)
            valores = [fila['modelo'], f"{fila['psnr']:.2f}", f"{fila['ssim']:.4f}",
                       f"{fila['gmsd']:.4f}", f"{fila['nitidez']:.4f}"]
            for c, v in enumerate(valores):
                self.tabla.setItem(r, c, QTableWidgetItem(v))


class VentanaPrincipal(QMainWindow):
    INSET_FRAC = 0.40
    MIN_LADO_EVAL = 32

    def __init__(self):
        super().__init__()
        self._imagen_orig = None    # float64 [0,1] en grises (luminancia)
        self._imagen_color = None   # float64 [0,1] RGB, o None si es gris
        self._hilo = None
        self._resultado = None      # ultimo resultado (para exportar y vista con inset)
        self._modo_lupa_activo = False
        self._hilo_lupa = None
        self._ultima_lupa = 0.0

        self._construir_ui()
        self._construir_toolbar()
        self._conectar()

        # compilar los kernels Numba en segundo plano (la primera ampliacion
        # no tendra la pausa de compilacion JIT)
        if NUMBA_DISPONIBLE:
            self._hilo_init = HiloCalculo(_calentamiento)
            self._hilo_init.terminado.connect(
                lambda _: self.statusBar().showMessage('Listo. Cargue una imagen para comenzar.'))
            self._hilo_init.start()
            self.statusBar().showMessage('Iniciando: compilando kernels Numba en segundo plano...')

    # construccion de la interfaz

    def _construir_ui(self):
        self.setWindowTitle('RZI - Reconstructor de Zoom Iterativo (Poisson + SOR Red-Black)')
        self.setMinimumSize(1300, 780)
        self.resize(1550, 900)

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
        visores_lay = QHBoxLayout()

        # visor izquierdo: imagen original
        col_orig = QVBoxLayout()
        lbl_orig = QLabel('Original (zoom nativo)')
        lbl_orig.setStyleSheet('color: #2196F3; font-weight: bold; padding: 2px;')
        lbl_orig.setAlignment(Qt.AlignCenter)
        self.visor_orig = VisorImagen()
        col_orig.addWidget(lbl_orig)
        col_orig.addWidget(self.visor_orig)
        visores_lay.addLayout(col_orig, 1)

        # visor derecho: imagen ampliada + toggle de vista contextual con inset
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
        self.visor_recon = VisorImagen()
        col_recon.addLayout(hdr_recon)
        col_recon.addWidget(self.visor_recon)
        visores_lay.addLayout(col_recon, 1)

        cen_l.addLayout(visores_lay)

        self.barra_progreso = QProgressBar()
        self.barra_progreso.setRange(0, 0)
        self.barra_progreso.setVisible(False)
        cen_l.addWidget(self.barra_progreso)

        self.panel_res = PanelResultados()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_p)
        splitter.addWidget(cen_w)
        splitter.addWidget(self.panel_res)
        splitter.setSizes([270, 780, 500])
        lay.addWidget(splitter)
        self.statusBar().showMessage('Listo. Cargue una imagen para comenzar.')

    def _construir_toolbar(self):
        tb = QToolBar('Herramientas')
        tb.setMovable(False)
        self.addToolBar(tb)
        self.act_abrir = QAction('Abrir imagen', self)
        self.act_abrir.setShortcut('Ctrl+O')
        self.act_ajustar = QAction('Ajustar vistas', self)
        self.act_guardar_img = QAction('Guardar imagen ampliada', self)
        self.act_guardar_graf = QAction('Guardar grafica activa', self)
        self.act_export_csv = QAction('Exportar CSV', self)
        self.act_salir = QAction('Salir', self)
        self.act_salir.setShortcut('Ctrl+Q')
        tb.addAction(self.act_abrir)
        tb.addAction(self.act_ajustar)
        tb.addSeparator()
        for act in (self.act_guardar_img, self.act_guardar_graf, self.act_export_csv):
            tb.addAction(act)
        tb.addSeparator()
        tb.addAction(self.act_salir)

    def _conectar(self):
        self.act_abrir.triggered.connect(self._abrir_imagen)
        self.act_ajustar.triggered.connect(self._ajustar_vistas)
        self.act_salir.triggered.connect(self.close)
        self.act_guardar_img.triggered.connect(self._guardar_imagen)
        self.act_guardar_graf.triggered.connect(self._guardar_grafica)
        self.act_export_csv.triggered.connect(self._exportar_csv)
        self.chk_vista_inset.toggled.connect(self._actualizar_visor_recon)

        pp = self.panel_params
        pp.reconstruir_solicitado.connect(self._reconstruir)
        pp.comparar_solicitado.connect(self._comparar)
        pp.barrido_omega_solicitado.connect(self._barrido_omega)
        pp.benchmark_solicitado.connect(self._benchmark)
        pp.lupa_solicitado.connect(self._toggle_modo_lupa)
        pp.animar_solicitado.connect(self._animar_convergencia)

        pr = self.panel_res
        pr.btn_guardar_img.clicked.connect(self._guardar_imagen)
        pr.btn_guardar_graf.clicked.connect(self._guardar_grafica)
        pr.btn_exportar_csv.clicked.connect(self._exportar_csv)

    # carga de imagen

    def cargar_imagen(self, ruta: str) -> None:
        """Carga una imagen desde disco (tambien usado por pruebas)."""
        arr = np.asarray(Image.open(ruta).convert('RGB'), dtype=np.float64) / 255.0
        # una imagen es "a color" si sus canales difieren en algun pixel
        es_color = not (np.allclose(arr[..., 0], arr[..., 1])
                        and np.allclose(arr[..., 1], arr[..., 2]))
        self._imagen_orig = imagen_a_grises(arr)
        self._imagen_color = arr if es_color else None
        self.visor_orig.establecer_imagen(arr if es_color else self._imagen_orig)
        self._resultado = None
        self.visor_recon.limpiar()
        h, w = self._imagen_orig.shape
        self.statusBar().showMessage(
            f"Imagen cargada ({'color' if es_color else 'grises'}): "
            f"{os.path.basename(ruta)} ({w}x{h} px). Haga zoom sobre una region "
            f"y pulse Reconstruir.")

    def _abrir_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, 'Abrir imagen', '',
            'Imagenes (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp);;Todos (*)')
        if not ruta:
            return
        try:
            self.cargar_imagen(ruta)
        except Exception as exc:
            QMessageBox.critical(self, 'Error al cargar imagen', str(exc))

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
                QMessageBox.warning(self, 'Sin imagen', 'Cargue una imagen primero.')
            return None
        region = self.visor_orig.obtener_viewport_imagen()
        if region is None:
            region = (0, 0, self._imagen_orig.shape[1], self._imagen_orig.shape[0])
        x, y, w, h = region
        fuente = self._imagen_color if self._imagen_color is not None else self._imagen_orig
        tile = fuente[y:y + h, x:x + w, ...]
        if min(tile.shape[0], tile.shape[1]) < 8:
            if avisar:
                QMessageBox.warning(self, 'Region muy pequena',
                                    'Haga zoom hacia afuera o cargue una imagen mas grande.')
            return None
        return tile, x, y, w, h

    # hilos

    def _iniciar_hilo(self, funcion, al_terminar) -> bool:
        if self._hilo is not None and self._hilo.isRunning():
            QMessageBox.warning(self, 'Ocupado', 'Ya hay un calculo en progreso.')
            return False
        self.barra_progreso.setVisible(True)
        self.panel_params.deshabilitar(True)
        self._hilo = HiloCalculo(funcion)
        # conectar ANTES de start(): si no, un calculo muy rapido podria
        # emitir 'terminado' antes de que exista la conexion
        self._hilo.terminado.connect(al_terminar)
        self._hilo.error.connect(self._en_error)
        self._hilo.finished.connect(self._en_fin_hilo)
        self._hilo.start()
        return True

    def _en_error(self, msg: str):
        QMessageBox.critical(self, 'Error en calculo', msg)

    def _en_fin_hilo(self):
        self.barra_progreso.setVisible(False)
        self.panel_params.deshabilitar(False)

    # ampliacion compartida (Reconstruir)

    @staticmethod
    def _ampliar(tile: np.ndarray, factor: int, params: dict, **cambios) -> dict:
        """Amplia el tile con RZI y prepara las referencias (vecino y bicubica)."""
        res = upscale(tile, factor, **kwargs_rzi(params, **cambios))
        H, W = res['imagen_recon'].shape[:2]
        th, tw = tile.shape[:2]
        if res['tile_reducido']:
            # el tope de memoria redujo la entrada: las referencias se hacen
            # sobre el mismo tile reducido para que la comparacion sea justa
            th, tw = H // factor, W // factor
            tile = (resize_rgb_float(tile, tw, th, Image.LANCZOS) if tile.ndim == 3
                    else np.clip(resize_float(tile, tw, th, Image.LANCZOS), 0, 1))
        if tile.ndim == 3:
            bic = resize_rgb_float(tile, W, H, Image.BICUBIC)
            nn = resize_rgb_float(tile, W, H, Image.NEAREST)
        else:
            bic = np.clip(resize_float(tile, W, H, Image.BICUBIC), 0, 1)
            nn = np.clip(resize_float(tile, W, H, Image.NEAREST), 0, 1)
        return {'res': res, 'display': res['imagen_recon'], 'bicubica': bic,
                'nn': nn, 'tile': tile}

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

    def _actualizar_visor_recon(self, *_):
        if not self._resultado or 'imagen_upscale' not in self._resultado:
            return
        display = self._resultado['imagen_upscale']
        region = self._resultado.get('region')
        if self.chk_vista_inset.isChecked() and region is not None and self._imagen_orig is not None:
            contexto, reg, inset = self._componer_con_inset(display, region)
            self.visor_recon.establecer_imagen(contexto)
            self.visor_recon.marcar_region(*reg, color='#FF5252')
            self.visor_recon.marcar_region(*inset, color='#00E676')
        else:
            self.visor_recon.establecer_imagen(display)

    # Reconstruir

    def _reconstruir(self):
        info = self._tile_desde_viewport()
        if info is None:
            return
        tile, tx, ty, tw, th = info
        params = self.panel_params.obtener_params()
        factor = params['factor']

        def tarea():
            datos = self._ampliar(tile, factor, params)
            gris = lambda a: a if a.ndim == 2 else imagen_a_grises(a)
            peq = gris(datos['tile'])
            datos['metricas'] = {
                nombre: (nitidez_gradiente(gris(img)), consistencia_datos(gris(img), peq))
                for nombre, img in (('nn', datos['nn']), ('bicubica', datos['bicubica']),
                                    ('rzi', datos['display']))}
            datos.update(region=(tx, ty, tw, th), factor=factor)
            return datos

        if self._iniciar_hilo(tarea, self._mostrar_reconstruccion):
            self.statusBar().showMessage(
                f'Ampliando region ({tx},{ty}) {tw}x{th} px x{factor} con RZI...')

    def _mostrar_reconstruccion(self, datos: dict):
        self._resultado = {
            'imagen_upscale': datos['display'],
            'res':            datos['res'],
            'region':         datos['region'],
        }
        res = datos['res']
        self._actualizar_visor_recon()
        fig = grafica_convergencia(res['historial'], res['omega'], res['error_datos'])
        self.panel_res.mostrar_figura(PanelResultados.TAB_CONV, fig)

        tx, ty, tw, th = datos['region']
        H, W = datos['display'].shape[:2]
        aviso = '  |  (entrada reducida por tope de memoria)' if res['tile_reducido'] else ''
        self.statusBar().showMessage(
            f"RZI x{datos['factor']}: region {tw}x{th} -> {W}x{H} px  |  "
            f"ciclos={len(res['error_datos']) - 2}  iter SOR={res['iteraciones']}  "
            f"t_SOR={res['tiempo']:.2f}s  omega={res['omega']:.3f}  "
            f"conv={'Si' if res['convergio'] else 'No'}{aviso}")
        self._mostrar_dialogo_detalle(datos)

    def _mostrar_dialogo_detalle(self, datos: dict):
        dialogo = QDialog(self)
        dialogo.setWindowTitle('Comparacion en detalle: zoom nativo vs bicubica vs RZI')
        dialogo.setStyleSheet(ESTILO)
        dialogo.resize(1150, 560)
        layout = QVBoxLayout(dialogo)

        lbl = QLabel(
            'Las tres columnas tienen el mismo tamano. Use la rueda para hacer zoom. '
            'Nitidez: magnitud media del gradiente (mayor = bordes mas marcados). '
            'Consistencia: PSNR entre la imagen ampliada reducida de nuevo y la '
            'original (mayor = respeta mejor los datos reales).')
        lbl.setWordWrap(True)
        lbl.setStyleSheet('color: #B0B0B0; padding: 4px;')
        layout.addWidget(lbl)

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

        # zoom sincronizado: mover uno mueve los tres
        def sincronizar(origen):
            for v in visores:
                if v is not origen:
                    v.copiar_estado_zoom(origen)
        for v in visores:
            v.vista_cambiada.connect(lambda o=v: sincronizar(o))

        btn = QPushButton('Cerrar')
        btn.clicked.connect(dialogo.accept)
        layout.addWidget(btn)
        dialogo.exec()

    # Comparar (con verdad de campo)

    def _comparar(self):
        info = self._tile_desde_viewport()
        if info is None:
            return
        tile, tx, ty, tw, th = info
        params = self.panel_params.obtener_params()
        factor = params['factor']
        lado_corto = min(th, tw)
        if lado_corto < self.MIN_LADO_EVAL:
            QMessageBox.warning(
                self, 'Region pequena',
                f'Para comparar modelos la region visible debe tener al menos '
                f'{self.MIN_LADO_EVAL} px por lado. Haga zoom hacia afuera.')
            return

        # si la region visible es menor que 16*factor, adaptar factor_eval
        # para evitar un low-res degenerado sin bloquear el flujo
        factor_eval = (factor if lado_corto >= 16 * factor
                       else max(2, min(factor, lado_corto // 16)))

        def tarea():
            gris = imagen_a_grises(tile) if tile.ndim == 3 else tile
            hr, lr = degradar(gris, factor_eval)
            bic = interpolacion_bicubica(lr, factor_eval)
            lan = interpolacion_lanczos(lr, factor_eval)
            res = upscale(lr, factor_eval, **kwargs_rzi(params, realce=0.0))
            fuerza = params['realce'] if params['realce'] > 0 else 0.6
            res_r = upscale(lr, factor_eval, **kwargs_rzi(params, realce=fuerza))
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
            }

        if self._iniciar_hilo(tarea, self._mostrar_comparacion):
            aviso = '' if factor_eval == factor else f' (evaluacion ajustada a x{factor_eval})'
            self.statusBar().showMessage(
                f'Comparando: region {tw}x{th} reducida x{factor_eval} y ampliada de nuevo{aviso}...')

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
            factor_eval=fe if fe != datos['factor_ped'] else None)
        self.panel_res.mostrar_figura(PanelResultados.TAB_COMP, fig)
        self.panel_res.mostrar_tabla(datos['tabla'])
        filas = {f['modelo']: f for f in datos['tabla']}
        ganancia = filas['RZI (fiel)']['psnr'] - filas['Bicubica']['psnr']
        extra = f' (factor eval x{fe})' if fe != datos['factor_ped'] else ''
        self.statusBar().showMessage(
            f'Comparacion lista{extra}: RZI (fiel) {ganancia:+.2f} dB sobre la bicubica.')

    # Barrido de omega

    def _barrido_omega(self):
        if self._imagen_orig is None:
            QMessageBox.warning(self, 'Sin imagen', 'Cargue una imagen primero.')
            return
        info = self._tile_desde_viewport(avisar=False)
        tile = imagen_a_grises(info[0]) if info and info[0].ndim == 3 else (
            info[0] if info else self._imagen_orig)
        # recorte central de hasta 256x256 para que el barrido sea rapido
        H, W = tile.shape
        lado_h, lado_w = min(H, 256), min(W, 256)
        y0, x0 = (H - lado_h) // 2, (W - lado_w) // 2
        tile = tile[y0:y0 + lado_h, x0:x0 + lado_w]
        factor = self.panel_params.obtener_params()['factor']

        def tarea():
            problema = construir_problema(tile, factor)
            return barrido_omega(problema, tol=1e-3, max_iter=500)

        if self._iniciar_hilo(tarea, self._en_barrido_terminado):
            self.statusBar().showMessage('Ejecutando barrido de omega...')

    def _en_barrido_terminado(self, r: dict):
        fig = grafica_barrido_omega(r['omegas'], r['iteraciones'], r['omega_teorico'],
                                    r['omega_malla_completa'])
        self.panel_res.mostrar_figura(PanelResultados.TAB_OMEGA, fig)
        self.statusBar().showMessage(
            f"Barrido omega | empirico={r['omega_empirico']:.3f} | "
            f"teorico con mascara={r['omega_teorico']:.3f} | "
            f"formula malla completa={r['omega_malla_completa']:.3f}")

    # Benchmark

    def _benchmark(self):
        if self._imagen_orig is None:
            QMessageBox.warning(self, 'Sin imagen', 'Cargue una imagen primero.')
            return
        img = self._imagen_orig

        def tarea():
            resultados = []
            for lado in (64, 128, 256):
                H, W = img.shape
                if H < lado or W < lado:
                    continue
                cy, cx = H // 2, W // 2
                t = img[cy - lado // 2:cy + lado // 2, cx - lado // 2:cx + lado // 2]
                resultados.append(benchmark_numba_vs_python(construir_problema(t, 4), n_iter=20))
            return resultados

        if self._iniciar_hilo(tarea, self._en_benchmark_terminado):
            self.statusBar().showMessage(
                'Ejecutando benchmark Numba vs Python (puede tardar unos segundos)...')

    def _en_benchmark_terminado(self, resultados: list):
        if not resultados:
            QMessageBox.warning(self, 'Imagen pequena', 'Se necesita al menos 64x64 px.')
            return
        self.panel_res.mostrar_figura(PanelResultados.TAB_BENCH, grafica_benchmark(resultados))
        r = resultados[-1]
        self.statusBar().showMessage(
            f"Benchmark {r['forma'][0]}x{r['forma'][1]}: Numba {r['aceleracion']:.0f} "
            f"veces mas rapido (mismas {r['iteraciones']} iteraciones).")

    # Lupa

    def _toggle_modo_lupa(self, activado: bool):
        if activado:
            if self._imagen_orig is None:
                QMessageBox.warning(self, 'Sin imagen', 'Cargue una imagen primero.')
                self.panel_params.btn_lupa.setChecked(False)
                return
            self._modo_lupa_activo = True
            self.visor_orig.setCursor(Qt.CrossCursor)
            self.visor_orig.mouse_moved.connect(self._actualizar_lupa)
            self.statusBar().showMessage('Modo Lupa activo: mueva el mouse sobre la imagen.')
        else:
            self._modo_lupa_activo = False
            self.visor_orig.unsetCursor()
            try:
                self.visor_orig.mouse_moved.disconnect(self._actualizar_lupa)
            except (RuntimeError, TypeError):
                pass
            self.statusBar().showMessage('Modo Lupa desactivado.')

    def _actualizar_lupa(self, x_img: int, y_img: int):
        if not self._modo_lupa_activo or self._imagen_orig is None:
            return
        ahora = time.time()
        if ahora - self._ultima_lupa < 0.12:
            return
        self._ultima_lupa = ahora
        if self._hilo_lupa is not None and self._hilo_lupa.isRunning():
            return

        # radio proporcional a la region visible (con zoom, lupa mas chica)
        vp = self.visor_orig.obtener_viewport_imagen()
        radio = 60 if vp is None else max(20, min(int(0.18 * min(vp[2], vp[3])), 80))
        H, W = self._imagen_orig.shape
        x0, y0 = max(0, x_img - radio), max(0, y_img - radio)
        x1, y1 = min(W, x_img + radio), min(H, y_img + radio)
        fuente = self._imagen_color if self._imagen_color is not None else self._imagen_orig
        tile = fuente[y0:y1, x0:x1, ...]
        if min(tile.shape[0], tile.shape[1]) < 8:
            return
        params = self.panel_params.obtener_params()

        def tarea():
            # version rapida: menos ciclos e iteraciones
            return upscale(tile, 3, **kwargs_rzi(params, ciclos=4, sor_iters=60,
                                                 bp_final=3))['imagen_recon']

        self._hilo_lupa = HiloCalculo(tarea)
        self._hilo_lupa.terminado.connect(self._mostrar_lupa)
        self._hilo_lupa.error.connect(
            lambda msg: self.statusBar().showMessage(f'Error en lupa: {msg}'))
        self._hilo_lupa.start()

    def _mostrar_lupa(self, imagen):
        if not self._modo_lupa_activo:
            return
        self.visor_recon.establecer_imagen(imagen)
        self.statusBar().showMessage(
            f'Lupa RZI x3: {imagen.shape[1]}x{imagen.shape[0]} px')

    # Animacion didactica del proceso completo de reconstruccion

    def _animar_convergencia(self):
        info = self._tile_desde_viewport()
        if info is None:
            return
        tile = info[0]
        params = self.panel_params.obtener_params()

        def tarea():
            return generar_frames_pipeline(tile, params['factor'], **kwargs_rzi(params))

        if self._iniciar_hilo(tarea, self._mostrar_dialogo_animacion):
            self.statusBar().showMessage(
                'Generando animacion del proceso completo de reconstruccion...')

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

        # cuerpo: lista lateral de pasos + visores sincronizados (RZI vs Bicubica)
        cuerpo = QHBoxLayout()
        cuerpo.setSpacing(8)

        lista_pasos = QListWidget()
        lista_pasos.setFixedWidth(240)
        for i, nombre in enumerate(pasos):
            lista_pasos.addItem(QListWidgetItem(f'{i + 1}. {nombre}'))

        col_lista = QVBoxLayout()
        lbl_lista = QLabel('Etapas del pipeline')
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
            sufijo = '  [Identica a Reconstruir]' if idx == n_frames - 1 else ''
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

    # Exportacion

    def _guardar_imagen(self):
        img = (self._resultado or {}).get('imagen_upscale')
        if img is None:
            QMessageBox.warning(self, 'Sin resultado', 'Ejecute Reconstruir o Comparar primero.')
            return
        ruta, _ = QFileDialog.getSaveFileName(self, 'Guardar imagen ampliada',
                                              'rzi_ampliada.png', 'PNG (*.png);;JPEG (*.jpg)')
        if ruta:
            guardar_imagen(img, ruta)
            QMessageBox.information(self, 'Guardado', f'Imagen guardada en:\n{ruta}')

    def _guardar_grafica(self):
        fig = self.panel_res.figura_actual()
        if fig is None:
            QMessageBox.warning(self, 'Sin grafica', 'La pestana activa no tiene grafica.')
            return
        ruta, _ = QFileDialog.getSaveFileName(self, 'Guardar grafica', 'grafica.png',
                                              'PNG (*.png);;PDF (*.pdf)')
        if ruta:
            fig.savefig(ruta, dpi=150, bbox_inches='tight')
            QMessageBox.information(self, 'Guardado', f'Grafica guardada en:\n{ruta}')

    def _exportar_csv(self):
        if not self._resultado or 'tabla' not in self._resultado:
            QMessageBox.warning(self, 'Sin datos', 'Ejecute "Comparar modelos" primero.')
            return
        ruta, _ = QFileDialog.getSaveFileName(self, 'Exportar CSV', 'resultados.csv',
                                              'CSV (*.csv)')
        if ruta:
            exportar_csv(self._resultado['tabla'], self._resultado.get('res', {}), ruta)
            QMessageBox.information(self, 'Exportado', f'Datos exportados a:\n{ruta}')
