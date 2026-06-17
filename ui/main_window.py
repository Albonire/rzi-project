"""
capa 1 - ventana principal de la aplicacion rzi.

layout:
    [parametros] | [visor original] [visor reconstruido] | [resultados]

flujo de "reconstruir" (upscaling armonico + embed):
    1. el usuario hace zoom sobre una region de interes en el visor izquierdo
    2. al pulsar "reconstruir", la region visible se agranda factor veces con laplace sor
    3. el resultado se reduce de vuelta al tamano original (LANCZOS) y se embebe
    4. el visor derecho muestra la imagen COMPLETA con la region refinada, al mismo zoom

flujo de "comparar 3 modelos" (enfoque b, para metricas psnr/ssim):
    se degrada la imagen a proposito y se mide la calidad de la reconstruccion
"""

import os
import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDoubleSpinBox, QSpinBox, QFileDialog, QMessageBox,
    QGroupBox, QSplitter, QProgressBar, QToolBar, QTabWidget,
    QScrollArea, QGridLayout, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from PySide6.QtCore  import Qt, Signal, QThread
from PySide6.QtGui   import QFont, QAction

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ui.zoom_viewer import VisorImagen
from core.poisson_builder import (construir_problema, construir_problema_screened,
                                   imagen_a_grises, calcular_omega_optimo)
from core.sor_solver import (resolver, resolver_screened, barrido_omega as _barrido,
                              benchmark_numba_vs_python, NUMBA_DISPONIBLE,
                              _calentamiento)
from core.upscaler import (upscale_poisson_screened, finalizar_display,
                           finalizar_display_color, generar_frames_pipeline)
from core.image_ops import rgb_a_ycbcr, ycbcr_a_rgb, resize_float, resize_rgb_float
from services.metrics import (psnr, ssim, interpolacion_bicubica, tabla_comparativa)
from services.exporter import (guardar_imagen, exportar_csv,
                                grafica_convergencia, grafica_barrido_omega,
                                grafica_comparacion_4modelos, grafica_benchmark)


# tema oscuro clasico: fondo gris oscuro con acento azul
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
    padding: 8px 18px;
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
"""

class HiloCalculo(QThread):
    terminado = Signal(object)
    error     = Signal(str)

    def __init__(self, funcion, *args, **kwargs):
        super().__init__()
        self._fn = funcion
        self._args = args
        self._kw = kwargs

    def run(self):
        try:
            resultado = self._fn(*self._args, **self._kw)
            self.terminado.emit(resultado)
        except Exception as exc:
            self.error.emit(str(exc))


class PanelParametros(QWidget):
    reconstruir_solicitado   = Signal()
    comparar_solicitado      = Signal()
    barrido_omega_solicitado = Signal()
    benchmark_solicitado     = Signal()
    lupa_solicitado     = Signal(bool)  # True = activar, False = desactivar
    animar_solicitado   = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(220)
        self.setMaximumWidth(280)
        self._construir()

    def _construir(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(10)

        tit = QLabel('Parametros')
        tit.setFont(QFont('Segoe UI', 13, QFont.Bold))
        tit.setStyleSheet('color: #2196F3; padding: 4px;')
        lay.addWidget(tit)

        
        g_entrada = QGroupBox('Submuestreo')
        gl = QGridLayout(g_entrada)
        gl.setSpacing(6)

        gl.addWidget(QLabel('Factor:'), 0, 0)
        self.spin_factor = QSpinBox()
        self.spin_factor.setRange(2, 12)
        self.spin_factor.setValue(4)
        self.spin_factor.setToolTip(
            'factor de upscaling: 2..12 (con factor=4 se conoce 1 de cada 16 '
            'pixeles). Factores grandes activan mas etapas en la cascada y '
            'tardan mas; la descomposicion es automatica (ej: x12 -> x2,x2,x3).')
        gl.addWidget(self.spin_factor, 0, 1)

        lay.addWidget(g_entrada)

        
        g_solver = QGroupBox('Solver SOR Red-Black')
        sl = QGridLayout(g_solver)
        sl.setSpacing(6)

        sl.addWidget(QLabel('Tolerancia:'), 0, 0)
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setDecimals(8)
        self.spin_tol.setRange(1e-7, 1e-2)
        self.spin_tol.setValue(1e-5)
        self.spin_tol.setSingleStep(1e-6)
        sl.addWidget(self.spin_tol, 0, 1)

        sl.addWidget(QLabel('Max iter:'), 1, 0)
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(50, 10000)
        self.spin_max_iter.setValue(2000)
        self.spin_max_iter.setSingleStep(100)
        sl.addWidget(self.spin_max_iter, 1, 1)

        sl.addWidget(QLabel('Omega auto:'), 2, 0)
        self.chk_omega_auto = QCheckBox()
        self.chk_omega_auto.setChecked(True)
        self.chk_omega_auto.toggled.connect(
            lambda auto: self.spin_omega.setEnabled(not auto))
        sl.addWidget(self.chk_omega_auto, 2, 1)

        sl.addWidget(QLabel('Omega manual:'), 3, 0)
        self.spin_omega = QDoubleSpinBox()
        self.spin_omega.setDecimals(4)
        self.spin_omega.setRange(1.0001, 1.9999)
        self.spin_omega.setValue(1.5)
        self.spin_omega.setSingleStep(0.05)
        self.spin_omega.setEnabled(False)
        sl.addWidget(self.spin_omega, 3, 1)

        lay.addWidget(g_solver)


        g_sharp = QGroupBox('Nitidez (USM)')
        nl = QGridLayout(g_sharp)
        nl.setSpacing(6)
        nl.addWidget(QLabel('Fuerza (%):'), 0, 0)
        self.spin_usm = QSpinBox()
        self.spin_usm.setRange(0, 300)
        self.spin_usm.setValue(60)
        self.spin_usm.setSingleStep(25)
        self.spin_usm.setToolTip(
            '0 = sin realce; 60 = toque suave (recomendado, el modelo '
            'screened ya realza con beta); 150+ = fuerte, puede generar halos')
        nl.addWidget(self.spin_usm, 0, 1)
        lay.addWidget(g_sharp)


        g_screened = QGroupBox('Nitidez Adaptativa (Screened Poisson)')
        scl = QGridLayout(g_screened)
        scl.setSpacing(6)

        scl.addWidget(QLabel('Lambda (fidelidad):'), 0, 0)
        self.spin_lam = QDoubleSpinBox()
        self.spin_lam.setDecimals(2)
        self.spin_lam.setRange(0.5, 50.0)
        self.spin_lam.setValue(8.0)
        self.spin_lam.setSingleStep(0.5)
        self.spin_lam.setToolTip(
            'peso del anclaje suave a las muestras conocidas; mayor lambda '
            '= mas fiel a la guia bicubica, menor = mas libre la PDE')
        scl.addWidget(self.spin_lam, 0, 1)

        scl.addWidget(QLabel('Beta (amplificacion):'), 1, 0)
        self.spin_beta = QDoubleSpinBox()
        self.spin_beta.setDecimals(2)
        self.spin_beta.setRange(1.0, 3.0)
        self.spin_beta.setValue(1.7)
        self.spin_beta.setSingleStep(0.1)
        self.spin_beta.setToolTip(
            'ganancia maxima sobre los gradientes de bordes reales (1 = sin '
            'amplificacion)')
        scl.addWidget(self.spin_beta, 1, 1)

        scl.addWidget(QLabel('Tau (umbral):'), 2, 0)
        self.spin_tau = QDoubleSpinBox()
        self.spin_tau.setDecimals(3)
        self.spin_tau.setRange(0.001, 0.2)
        self.spin_tau.setValue(0.02)
        self.spin_tau.setSingleStep(0.005)
        self.spin_tau.setToolTip(
            'magnitud de gradiente bajo la cual no se amplifica (evita '
            'realzar ruido)')
        scl.addWidget(self.spin_tau, 2, 1)

        scl.addWidget(QLabel('Multiescala:'), 3, 0)
        self.chk_multiescala = QCheckBox()
        self.chk_multiescala.setChecked(True)
        self.chk_multiescala.setToolTip(
            'resuelve en cascada coarse-to-fine (factores potencia de 2)')
        scl.addWidget(self.chk_multiescala, 3, 1)

        scl.addWidget(QLabel('Back-projection:'), 4, 0)
        self.chk_back_projection = QCheckBox()
        self.chk_back_projection.setChecked(True)
        self.chk_back_projection.setToolTip(
            'fuerza que la imagen reducida coincida con la original')
        scl.addWidget(self.chk_back_projection, 4, 1)

        scl.addWidget(QLabel('Anti-ringing:'), 5, 0)
        self.chk_anti_ringing = QCheckBox()
        self.chk_anti_ringing.setChecked(True)
        self.chk_anti_ringing.setToolTip(
            'recorta el resultado a los limites locales de la guia bicubica')
        scl.addWidget(self.chk_anti_ringing, 5, 1)

        scl.addWidget(QLabel('Filtro shock:'), 6, 0)
        self.chk_shock = QCheckBox()
        self.chk_shock.setChecked(True)
        self.chk_shock.setToolTip(
            'filtro de shock (Osher-Rudin): convierte rampas borrosas en '
            'bordes nitidos, sin afectar zonas planas')
        scl.addWidget(self.chk_shock, 6, 1)

        scl.addWidget(QLabel('Difusion (denoise):'), 7, 0)
        self.chk_difusion = QCheckBox()
        self.chk_difusion.setChecked(True)
        self.chk_difusion.setToolTip(
            'difusion anisotropica (Perona-Malik): limpia ruido de grano en '
            'zonas planas preservando los bordes')
        scl.addWidget(self.chk_difusion, 7, 1)

        scl.addWidget(QLabel('Intensidad back-projection:'), 8, 0)
        self.spin_back_gain = QDoubleSpinBox()
        self.spin_back_gain.setDecimals(2)
        self.spin_back_gain.setRange(0.0, 1.0)
        self.spin_back_gain.setValue(0.6)
        self.spin_back_gain.setSingleStep(0.1)
        self.spin_back_gain.setToolTip(
            'gain de back-projection (0.0..1.0): mas alto = mayor fidelidad '
            'con la imagen original pero mas grano; mas bajo = imagen mas limpia. '
            'Default 0.6 equilibra fidelidad y ruido.')
        self.chk_back_projection.toggled.connect(self.spin_back_gain.setEnabled)
        scl.addWidget(self.spin_back_gain, 8, 1)

        lay.addWidget(g_screened)


        self.btn_reconstruir = QPushButton('Reconstruir (Poisson SOR)')
        self.btn_reconstruir.setObjectName('btn_principal')
        self.btn_reconstruir.setMinimumHeight(36)
        self.btn_reconstruir.clicked.connect(self.reconstruir_solicitado)
        lay.addWidget(self.btn_reconstruir)

        self.btn_comparar = QPushButton('Comparar 3 modelos')
        self.btn_comparar.setMinimumHeight(30)
        self.btn_comparar.clicked.connect(self.comparar_solicitado)
        lay.addWidget(self.btn_comparar)

        self.btn_barrido = QPushButton('Barrido Omega')
        self.btn_barrido.setMinimumHeight(30)
        self.btn_barrido.clicked.connect(self.barrido_omega_solicitado)
        lay.addWidget(self.btn_barrido)

        self.btn_benchmark = QPushButton('Benchmark Numba vs Python')
        self.btn_benchmark.setMinimumHeight(30)
        self.btn_benchmark.clicked.connect(self.benchmark_solicitado)
        lay.addWidget(self.btn_benchmark)

        self.btn_lupa = QPushButton('Modo Lupa (tiempo real)')
        self.btn_lupa.setMinimumHeight(30)
        self.btn_lupa.setCheckable(True)
        self.btn_lupa.setToolTip('Activa lupa que sigue el cursor con upscaling SOR')
        self.btn_lupa.toggled.connect(self.lupa_solicitado)
        lay.addWidget(self.btn_lupa)

        self.btn_animar = QPushButton('Animar Convergencia')
        self.btn_animar.setMinimumHeight(30)
        self.btn_animar.setToolTip('Muestra animacion frame a frame de como converge SOR')
        self.btn_animar.clicked.connect(self.animar_solicitado)
        lay.addWidget(self.btn_animar)

        # estado de numba visible para el usuario
        color = '#4CAF50' if NUMBA_DISPONIBLE else '#FF9800'
        texto = 'Numba: activo' if NUMBA_DISPONIBLE else 'Numba: no disponible (lento)'
        lbl_nb = QLabel(texto)
        lbl_nb.setStyleSheet(f'color: {color}; font-size: 9pt; padding: 4px;')
        lay.addWidget(lbl_nb)

        lay.addStretch()

    def obtener_params(self) -> dict:
        return {
            'factor':   self.spin_factor.value(),
            'tol':      self.spin_tol.value(),
            'max_iter': self.spin_max_iter.value(),
            # none = usar omega optimo calculado automaticamente
            'omega':    None if self.chk_omega_auto.isChecked() else self.spin_omega.value(),
            'usm_pct':  self.spin_usm.value(),
            'lam':              self.spin_lam.value(),
            'beta':             self.spin_beta.value(),
            'tau':              self.spin_tau.value(),
            'multiescala':      self.chk_multiescala.isChecked(),
            'back_projection':  self.chk_back_projection.isChecked(),
            'anti_ringing':     self.chk_anti_ringing.isChecked(),
            'usar_shock':       self.chk_shock.isChecked(),
            'usar_difusion':    self.chk_difusion.isChecked(),
            'back_gain':        self.spin_back_gain.value(),
        }

    def deshabilitar(self, flag: bool):
        for btn in (self.btn_reconstruir, self.btn_comparar,
                    self.btn_barrido, self.btn_benchmark,
                    self.btn_lupa, self.btn_animar):
            btn.setEnabled(not flag)

class PanelResultados(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(460)
        self._fig_actual = None
        self._construir()

    def _construir(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        # tab comparacion: muestra las 4 imagenes con metricas
        self._tab_comp = QWidget()
        lay_comp = QVBoxLayout(self._tab_comp)

        self.canvas_comp = FigureCanvas(Figure(figsize=(10, 3), facecolor='#252525'))
        lay_comp.addWidget(self.canvas_comp)

        lay_comp.addWidget(QLabel('Metricas de Calidad:'))
        self.tabla_metricas_comp = QTableWidget(0, 5)
        self.tabla_metricas_comp.setHorizontalHeaderLabels(
            ['Modelo', 'PSNR (dB)', 'SSIM', 'GMSD', 'Info Solver'])
        self.tabla_metricas_comp.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_metricas_comp.setAlternatingRowColors(True)
        self.tabla_metricas_comp.setMaximumHeight(120)
        lay_comp.addWidget(self.tabla_metricas_comp)

        self.tabs.addTab(self._tab_comp, 'Comparacion')

        # tab convergencia: residuo vs iteraciones
        self._tab_conv = QWidget()
        lay_conv = QVBoxLayout(self._tab_conv)
        self.canvas_conv = FigureCanvas(Figure(figsize=(7, 4), facecolor='#252525'))
        lay_conv.addWidget(self.canvas_conv)
        self.tabs.addTab(self._tab_conv, 'Convergencia')

        # tab barrido omega: v-shape de iteraciones vs omega
        self._tab_omega = QWidget()
        lay_omega = QVBoxLayout(self._tab_omega)
        self.canvas_omega = FigureCanvas(Figure(figsize=(7, 4), facecolor='#252525'))
        lay_omega.addWidget(self.canvas_omega)
        self.tabs.addTab(self._tab_omega, 'Barrido Omega')

        # tab metricas: tabla con psnr/ssim
        self._tab_met = QWidget()
        lay_met = QVBoxLayout(self._tab_met)
        self.tabla_metricas = QTableWidget(0, 5)
        self.tabla_metricas.setHorizontalHeaderLabels(
            ['Modelo', 'PSNR (dB)', 'SSIM', 'GMSD', 'Solver'])
        self.tabla_metricas.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabla_metricas.setAlternatingRowColors(True)
        lay_met.addWidget(self.tabla_metricas)
        self.tabs.addTab(self._tab_met, 'Metricas')

        # botones de exportacion en la parte inferior
        lay_exp = QHBoxLayout()
        self.btn_guardar_img  = QPushButton('Guardar imagen')
        self.btn_guardar_graf = QPushButton('Guardar grafica')
        self.btn_exportar_csv = QPushButton('Exportar CSV')
        for btn in (self.btn_guardar_img, self.btn_guardar_graf, self.btn_exportar_csv):
            btn.setMinimumHeight(28)
            lay_exp.addWidget(btn)
        lay.addLayout(lay_exp)

    # actualizar cada tab

    def mostrar_comparacion(self, original, bicubica, laplace, poisson, tabla,
                             res_poi=None):
        fig = grafica_comparacion_4modelos(original, bicubica, laplace, poisson, tabla)
        self._renderizar_en(self.canvas_comp, fig)
        plt.close(fig)

        self.tabla_metricas_comp.setRowCount(0)
        info_solver = [
            '',
            '',
            (f"iter={res_poi.get('iteraciones','--')}"
             f" t={res_poi.get('tiempo', 0):.3f}s"
             f" omega={res_poi.get('omega', 0):.4f}"
             f" conv={'Si' if res_poi.get('convergio') else 'No'}")
        ] if res_poi else ['', '', '']
        for fila, info in zip(tabla, info_solver):
            row = self.tabla_metricas_comp.rowCount()
            self.tabla_metricas_comp.insertRow(row)
            psnr_v = fila.get('psnr', 0)
            ssim_v = fila.get('ssim', 0)
            gmsd_v = fila.get('gmsd', 0)
            self.tabla_metricas_comp.setItem(row, 0, QTableWidgetItem(fila.get('modelo', '')))
            self.tabla_metricas_comp.setItem(row, 1, QTableWidgetItem(
                'inf' if psnr_v == float('inf') else f'{psnr_v:.2f}'))
            self.tabla_metricas_comp.setItem(row, 2, QTableWidgetItem(f'{ssim_v:.4f}'))
            self.tabla_metricas_comp.setItem(row, 3, QTableWidgetItem(f'{gmsd_v:.4f}'))
            self.tabla_metricas_comp.setItem(row, 4, QTableWidgetItem(info))

        self.tabs.setCurrentIndex(0)

    def mostrar_convergencia(self, historial: np.ndarray, omega: float):
        fig = grafica_convergencia(historial, omega)
        self._renderizar_en(self.canvas_conv, fig)
        self._fig_actual = fig
        plt.close(fig)
        self.tabs.setCurrentIndex(1)

    def mostrar_barrido_omega(self, resultado: dict):
        fig = grafica_barrido_omega(
            resultado['omegas'], resultado['iteraciones'],
            resultado['omega_teorico'])
        self._renderizar_en(self.canvas_omega, fig)
        plt.close(fig)
        self.tabs.setCurrentIndex(2)

    def mostrar_metricas(self, tabla: list, res_solver: dict):
        self.tabla_metricas.setRowCount(0)
        info_solver = [
            '',
            '',
            (f"iter={res_solver.get('iteraciones','--')}  "
             f"t={res_solver.get('tiempo', 0):.3f}s  "
             f"omega={res_solver.get('omega', 0):.4f}  "
             f"conv={'Si' if res_solver.get('convergio') else 'No'}"),
        ]
        for fila, info in zip(tabla, info_solver):
            row = self.tabla_metricas.rowCount()
            self.tabla_metricas.insertRow(row)
            psnr_v = fila.get('psnr', 0)
            ssim_v = fila.get('ssim', 0)
            gmsd_v = fila.get('gmsd', 0)
            self.tabla_metricas.setItem(row, 0, QTableWidgetItem(fila.get('modelo', '')))
            self.tabla_metricas.setItem(row, 1, QTableWidgetItem(
                'inf' if psnr_v == float('inf') else f'{psnr_v:.2f}'))
            self.tabla_metricas.setItem(row, 2, QTableWidgetItem(f'{ssim_v:.4f}'))
            self.tabla_metricas.setItem(row, 3, QTableWidgetItem(f'{gmsd_v:.4f}'))
            self.tabla_metricas.setItem(row, 4, QTableWidgetItem(info))
        self.tabs.setCurrentIndex(3)

    def mostrar_benchmark(self, resultados: list):
        fig = grafica_benchmark(resultados)
        self._renderizar_en(self.canvas_conv, fig)
        self._fig_actual = fig
        plt.close(fig)
        self.tabs.setCurrentIndex(1)

    def _renderizar_en(self, canvas: FigureCanvas, fig: plt.Figure):
        """
        copia los ejes de fig al canvas embebido con colores del tema oscuro.
        necesario porque matplotlib genera figuras independientes que hay que
        transferir al canvas de qt sin abrir ventanas externas.
        """
        canvas.figure.clear()
        canvas.figure.set_facecolor('#252525')

        for ax_src in fig.get_axes():
            rect   = ax_src.get_position()
            ax_dst = canvas.figure.add_axes(rect)

            for line in ax_src.get_lines():
                ax_dst.plot(
                    line.get_xdata(), line.get_ydata(),
                    color=line.get_color(), linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(), label=line.get_label(),
                    marker=line.get_marker(), markersize=line.get_markersize())

            for bc in ax_src.containers:
                for bar in bc:
                    ax_dst.bar(bar.get_x() + bar.get_width() / 2,
                               bar.get_height(), bar.get_width(),
                               color=bar.get_facecolor())

            for img in ax_src.get_images():
                ax_dst.imshow(img.get_array(), cmap=img.get_cmap(),
                              vmin=img.norm.vmin, vmax=img.norm.vmax,
                              aspect='equal', interpolation='nearest')

            title_visible = ax_src.title.get_visible()
            xlabel_visible = ax_src.xaxis.get_label().get_visible()
            ylabel_visible = ax_src.yaxis.get_label().get_visible()
            ticks_visible = any(tick.get_visible() for tick in ax_src.get_xticklabels())
            axis_off = not (title_visible or xlabel_visible or ylabel_visible or ticks_visible)
            if axis_off:
                ax_dst.axis('off')

            # estilos del tema oscuro
            ax_dst.set_facecolor('#252525')
            ax_dst.set_xlabel(ax_src.get_xlabel(), color='#B0B0B0')
            ax_dst.set_ylabel(ax_src.get_ylabel(), color='#B0B0B0')
            ax_dst.set_title(ax_src.get_title(),   color='#E0E0E0', fontsize=9)
            ax_dst.tick_params(colors='#B0B0B0')
            for spine in ax_dst.spines.values():
                spine.set_color('#3A3A3A')
            if ax_src.get_yscale() == 'log':
                ax_dst.set_yscale('log')
            ax_dst.grid(True, linestyle=':', alpha=0.4, color='#555')
            if ax_src.get_legend():
                handles, labels = ax_src.get_legend_handles_labels()
                if handles:
                    ax_dst.legend(labels=labels, fontsize=8,
                                  facecolor='#252525', labelcolor='#E0E0E0')
            ax_dst.set_xlim(ax_src.get_xlim())
            if ax_src.get_yscale() != 'log':
                ax_dst.set_ylim(ax_src.get_ylim())

        canvas.draw()

class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self._imagen_orig  = None  # ndarray float64 [0,1] grises (luminancia), imagen completa
        self._imagen_color = None  # ndarray float64 [0,1] RGB (h,w,3), o None si la imagen es gris
        self._hilo        = None
        self._resultado   = None   # ultimo resultado para exportacion
        self._modo_lupa_activo = False
        self._hilo_lupa        = None
        self._ultima_lupa      = 0.0

        self._construir_ui()
        self._construir_toolbar()
        self._conectar()

        # compilar el kernel numba al fondo para que la primera reconstruccion
        # no tenga la pausa de compilacion jit
        if NUMBA_DISPONIBLE:
            self._hilo_init = QThread()
            self._hilo_init.run = _calentamiento
            self._hilo_init.start()
            self.statusBar().showMessage(
                'Iniciando: compilando kernel Numba en segundo plano...')

    def _construir_ui(self):
        self.setWindowTitle('RZI - Reconstructor de Zoom Iterativo (Poisson SOR)')
        self.setMinimumSize(1300, 780)
        self.resize(1550, 880)

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
        cen_l.setSpacing(4)

        visores_lay = QHBoxLayout()
        visores_lay.setSpacing(6)

        # visor izquierdo: imagen original
        col_orig = QVBoxLayout()
        lbl_orig = QLabel('Original')
        lbl_orig.setStyleSheet('color: #2196F3; font-weight: bold; padding: 2px;')
        lbl_orig.setAlignment(Qt.AlignCenter)
        self.visor_orig = VisorImagen()
        col_orig.addWidget(lbl_orig)
        col_orig.addWidget(self.visor_orig)

        # visor derecho: imagen reconstruida / agrandada
        col_recon = QVBoxLayout()
        lbl_recon = QLabel('Reconstruida')
        lbl_recon.setStyleSheet('color: #4CAF50; font-weight: bold; padding: 2px;')
        lbl_recon.setAlignment(Qt.AlignCenter)
        self.visor_recon = VisorImagen()
        col_recon.addWidget(lbl_recon)
        col_recon.addWidget(self.visor_recon)

        visores_lay.addLayout(col_orig, 1)
        visores_lay.addLayout(col_recon, 1)
        cen_l.addLayout(visores_lay)

        # barra de progreso (oculta en reposo)
        self.barra_progreso = QProgressBar()
        self.barra_progreso.setRange(0, 0)   # modo indeterminado
        self.barra_progreso.setVisible(False)
        cen_l.addWidget(self.barra_progreso)

        
        self.panel_res = PanelResultados()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll_p)
        splitter.addWidget(cen_w)
        splitter.addWidget(self.panel_res)
        splitter.setSizes([250, 800, 500])
        lay.addWidget(splitter)

        self.statusBar().showMessage('Listo. Cargue una imagen para comenzar.')

    def _construir_toolbar(self):
        tb = QToolBar('Herramientas')
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_abrir = QAction('Abrir imagen', self)
        self.act_abrir.setShortcut('Ctrl+O')
        tb.addAction(self.act_abrir)

        self.act_ajustar = QAction('Ajustar vistas', self)
        tb.addAction(self.act_ajustar)

        tb.addSeparator()

        self.act_guardar_img  = QAction('Guardar imagen reconstruida', self)
        self.act_guardar_graf = QAction('Guardar grafica activa', self)
        self.act_export_csv   = QAction('Exportar CSV', self)
        for act in (self.act_guardar_img, self.act_guardar_graf, self.act_export_csv):
            tb.addAction(act)

        tb.addSeparator()
        self.act_salir = QAction('Salir', self)
        self.act_salir.setShortcut('Ctrl+Q')
        tb.addAction(self.act_salir)

    def _conectar(self):
        self.act_abrir.triggered.connect(self._abrir_imagen)
        self.act_ajustar.triggered.connect(self._ajustar_vistas)
        self.act_salir.triggered.connect(self.close)
        self.act_guardar_img.triggered.connect(self._guardar_imagen)
        self.act_guardar_graf.triggered.connect(self._guardar_grafica)
        self.act_export_csv.triggered.connect(self._exportar_csv)

        self.panel_params.reconstruir_solicitado.connect(self._reconstruir)
        self.panel_params.comparar_solicitado.connect(self._comparar)
        self.panel_params.barrido_omega_solicitado.connect(self._barrido_omega)
        self.panel_params.benchmark_solicitado.connect(self._benchmark)

        self.panel_params.lupa_solicitado.connect(self._toggle_modo_lupa)
        self.panel_params.animar_solicitado.connect(self._animar_convergencia)

        self.panel_res.btn_guardar_img.clicked.connect(self._guardar_imagen)
        self.panel_res.btn_guardar_graf.clicked.connect(self._guardar_grafica)
        self.panel_res.btn_exportar_csv.clicked.connect(self._exportar_csv)

    # cargar imagen

    def _abrir_imagen(self):
        ruta, _ = QFileDialog.getOpenFileName(
            self, 'Abrir imagen', '',
            'Imagenes (*.png *.jpg *.jpeg *.bmp *.tiff *.webp);;Todos (*)')
        if not ruta:
            return
        try:
            pil_original = Image.open(ruta)
            es_color = pil_original.mode not in ('L', '1', 'I', 'F')

            pil_img = pil_original.convert('RGB')
            arr = np.asarray(pil_img, dtype=np.float64) / 255.0

            if es_color:
                self._imagen_color = arr
                self._imagen_orig  = imagen_a_grises(arr)
                self.visor_orig.establecer_imagen(self._imagen_color)
            else:
                self._imagen_color = None
                self._imagen_orig  = imagen_a_grises(arr)
                self.visor_orig.establecer_imagen(self._imagen_orig)

            self._resultado = None
            self.visor_recon.limpiar()
            modo_txt = 'color' if es_color else 'grises'
            self.statusBar().showMessage(
                f'Imagen cargada ({modo_txt}): {os.path.basename(ruta)}  '
                f'({self._imagen_orig.shape[1]}x{self._imagen_orig.shape[0]} px)')
        except Exception as exc:
            QMessageBox.critical(self, 'Error al cargar imagen', str(exc))

    def _ajustar_vistas(self):
        self.visor_orig.ajustar_a_ventana()
        self.visor_recon.ajustar_a_ventana()

    # extraccion del tile visible en el visor izquierdo

    def _tile_desde_viewport(self) -> tuple | None:
        """
        extrae la region de la imagen actualmente visible en el visor izquierdo.
        retorna (tile, x, y) donde (x,y) es la esquina superior izquierda
        en coordenadas de imagen completa, o none si hay error.
        """
        if self._imagen_orig is None:
            QMessageBox.warning(self, 'Sin imagen', 'Cargue una imagen primero.')
            return None

        region = self.visor_orig.obtener_viewport_imagen()

        if region is None:
            # si no hay region valida, usar la imagen completa
            region = (0, 0,
                      self._imagen_orig.shape[1],
                      self._imagen_orig.shape[0])

        x, y, w, h = region

        # si la imagen cargada es a color, se recorta y reconstruye a color
        # (luminancia-PDE); _imagen_orig (gris) sigue usandose para
        # benchmark/lupa-fallback y como base de las herramientas de analisis.
        fuente = self._imagen_color if self._imagen_color is not None else self._imagen_orig
        tile = fuente[y:y+h, x:x+w, ...]

        if tile.shape[0] * tile.shape[1] < 64:
            QMessageBox.warning(self, 'Region muy pequeña',
                                'Haga zoom fuera o cargue una imagen mas grande.')
            return None

        return tile, x, y, w, h

    # gestion del hilo de calculo

    def _iniciar_hilo(self, funcion, *args, **kwargs) -> bool:
        if self._hilo and self._hilo.isRunning():
            QMessageBox.warning(self, 'Ocupado', 'Ya hay un calculo en progreso.')
            return False
        self.barra_progreso.setVisible(True)
        self.panel_params.deshabilitar(True)
        self._hilo = HiloCalculo(funcion, *args, **kwargs)
        self._hilo.error.connect(self._en_error)
        self._hilo.finished.connect(self._en_fin_hilo)
        self._hilo.start()
        return True

    def _en_error(self, msg: str):
        QMessageBox.critical(self, 'Error en calculo', msg)

    def _en_fin_hilo(self):
        self.barra_progreso.setVisible(False)
        self.panel_params.deshabilitar(False)

    # modo lupa matematica
    def _toggle_modo_lupa(self, activado: bool):
        if activado:
            if self._imagen_orig is None:
                QMessageBox.warning(self, 'Sin imagen', 'Cargue una imagen primero.')
                self.panel_params.btn_lupa.setChecked(False)
                return
            self._modo_lupa_activo = True
            self.visor_orig.setCursor(Qt.CrossCursor)
            self.visor_orig.mouse_moved.connect(self._actualizar_lupa)
            self.statusBar().showMessage('Modo Lupa activo: mueva el mouse sobre la imagen')
        else:
            self._modo_lupa_activo = False
            self.visor_orig.unsetCursor()
            try:
                self.visor_orig.mouse_moved.disconnect(self._actualizar_lupa)
            except Exception:
                pass
            self.statusBar().showMessage('Modo Lupa desactivado')

    def _actualizar_lupa(self, x_img: int, y_img: int):
        if not self._modo_lupa_activo or self._imagen_orig is None:
            return

        import time
        now = time.time()
        if (now - self._ultima_lupa) < 0.125:
            return
        self._ultima_lupa = now

        if self._hilo_lupa is not None and self._hilo_lupa.isRunning():
            return

        # El radio de la lupa se adapta al zoom de la imagen original: se toma
        # como una fraccion de la region actualmente visible. Con zoom in la
        # region visible es chica -> radio chico -> la lupa amplia justo lo que
        # se esta mirando. Con zoom out el radio queda topado (RADIO_MAX) para
        # no terminar ampliando casi toda la imagen.
        FRACCION_LUPA = 0.18
        RADIO_MIN     = 20
        RADIO_MAX     = 80
        vp = self.visor_orig.obtener_viewport_imagen()
        if vp is None:
            radio = 60
        else:
            _, _, vw, vh = vp
            radio = int(FRACCION_LUPA * min(vw, vh))
            radio = max(RADIO_MIN, min(radio, RADIO_MAX))

        x0 = max(0, x_img - radio)
        y0 = max(0, y_img - radio)
        x1 = min(self._imagen_orig.shape[1], x_img + radio)
        y1 = min(self._imagen_orig.shape[0], y_img + radio)

        fuente = self._imagen_color if self._imagen_color is not None else self._imagen_orig
        tile = fuente[y0:y1, x0:x1, ...]
        if tile.shape[0] * tile.shape[1] < 64:
            return

        factor = 3
        params = self.panel_params.obtener_params()

        def _tarea_lupa(t=tile, f=factor, p=params):
            # Modo lupa: solo el solve screened, sin shock/difusion (se
            # prioriza la velocidad en tiempo real). A color, se resuelve
            # solo la luminancia y el croma se recombina via bicubica.
            if t.ndim == 3:
                y, cb, cr = rgb_a_ycbcr(t)
                problema = construir_problema_screened(y, f, lam=p['lam'],
                                                         beta=p['beta'], tau=p['tau'])
                res = resolver_screened(problema, max_iter=400, tol=1e-4)
                y_up = res['imagen_recon']
                new_h, new_w = y_up.shape
                cb_up = resize_float(cb, new_w, new_h, Image.BICUBIC)
                cr_up = resize_float(cr, new_w, new_h, Image.BICUBIC)
                return ycbcr_a_rgb(y_up, np.clip(cb_up, 0.0, 1.0), np.clip(cr_up, 0.0, 1.0))

            problema = construir_problema_screened(t, f, lam=p['lam'],
                                                     beta=p['beta'], tau=p['tau'])
            res = resolver_screened(problema, max_iter=400, tol=1e-4)
            return res['imagen_recon']

        self._hilo_lupa = HiloCalculo(_tarea_lupa)
        self._hilo_lupa.terminado.connect(self._mostrar_lupa)
        self._hilo_lupa.start()

    def _mostrar_lupa(self, imagen_upscaled):
        if not self._modo_lupa_activo:
            return
        self.visor_recon.establecer_imagen(imagen_upscaled)
        self.statusBar().showMessage(f'Lupa actualizada: {imagen_upscaled.shape[1]}x{imagen_upscaled.shape[0]} px')

    # animacion didactica del proceso completo de reconstruccion
    def _animar_convergencia(self):
        info = self._tile_desde_viewport()
        if info is None:
            return

        tile   = info[0]
        params = self.panel_params.obtener_params()

        def _tarea_animacion(t=tile, p=params):
            es_color = t.ndim == 3
            if es_color:
                y, cb, cr = rgb_a_ycbcr(t)
                canal = y
            else:
                canal = t

            # generar_frames_pipeline orquesta TODO el proceso: NN, bicubica,
            # SOR por etapa, back-projection, anti-ringing, difusion, shock,
            # USM y resultado final. El ultimo frame es identico a Reconstruir.
            gen = generar_frames_pipeline(
                canal, p['factor'],
                lam=p['lam'], beta=p['beta'], tau=p['tau'],
                usar_back_projection=p['back_projection'],
                usar_anti_ringing=p['anti_ringing'],
                usar_multiescala=p['multiescala'],
                usar_difusion=p['usar_difusion'],
                usar_shock=p['usar_shock'],
                usm_pct=p['usm_pct'],
                tol=p['tol'], max_iter=p['max_iter'],
                back_gain=p['back_gain'],
            )

            if es_color:
                # el croma no participa en el pipeline: se reescala con
                # bicubica una sola vez y se recombina con cada frame 2D de Y.
                th, tw   = canal.shape
                out_h    = th * p['factor']
                out_w    = tw * p['factor']
                cb_disp  = np.clip(resize_float(cb, out_w, out_h, Image.BICUBIC), 0.0, 1.0)
                cr_disp  = np.clip(resize_float(cr, out_w, out_h, Image.BICUBIC), 0.0, 1.0)
                from core.image_ops import ycbcr_a_rgb as _ycbcr_a_rgb
                frames_vis = [_ycbcr_a_rgb(f, cb_disp, cr_disp)
                              for f in gen['frames']]
            else:
                frames_vis = gen['frames']

            return {
                'frames':          frames_vis,
                'pasos':           gen['pasos'],
                'pasos_de_frame':  gen['pasos_de_frame'],
                'subtitulos':      gen['subtitulos'],
                'omega':           gen['omega'],
                'iteraciones':     gen['iteraciones'],
                'tiempo':          gen['tiempo'],
                'etapas':          gen['etapas'],
            }

        ok = self._iniciar_hilo(_tarea_animacion)
        if not ok:
            return

        self.statusBar().showMessage(
            'Generando animacion del proceso completo de reconstruccion...')
        self._hilo.terminado.connect(self._mostrar_dialogo_animacion)

    def _mostrar_dialogo_animacion(self, resultado: dict):
        from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                       QPushButton, QLabel, QSlider,
                                       QComboBox, QListWidget,
                                       QListWidgetItem, QSizePolicy)
        from PySide6.QtCore import QTimer, Qt

        frames = resultado.get('frames', [])
        if not frames:
            QMessageBox.warning(self, 'Error', 'No se generaron frames.')
            return

        pasos          = resultado['pasos']
        pasos_de_frame = resultado['pasos_de_frame']
        subtitulos     = resultado['subtitulos']
        n_frames       = len(frames)
        n_pasos        = len(pasos)

        dialogo = QDialog(self)
        dialogo.setWindowTitle('Proceso completo de reconstruccion RZI')
        dialogo.resize(1150, 680)
        dialogo.setStyleSheet(ESTILO)

        root = QVBoxLayout(dialogo)
        root.setSpacing(6)

        # --- cabecera con metadatos del proceso ---
        etapas_txt = ' -> '.join(f'x{e}' for e in resultado.get('etapas', []))
        lbl_info = QLabel(
            f'Frames: {n_frames}  |  Pasos: {n_pasos}  |  '
            f'Cascada: {etapas_txt}  |  '
            f'Omega: {resultado["omega"]:.3f}  |  '
            f'Iters SOR: {resultado["iteraciones"]}  |  '
            f'Tiempo SOR: {resultado["tiempo"]:.2f}s'
        )
        lbl_info.setStyleSheet('color: #2196F3; font-weight: bold; padding: 4px;')
        root.addWidget(lbl_info)

        # --- cuerpo: lista de pasos (izquierda) + visor (derecha) ---
        cuerpo = QHBoxLayout()
        cuerpo.setSpacing(8)

        # lista de pasos (esquema del proceso)
        lista_pasos = QListWidget()
        lista_pasos.setFixedWidth(240)
        lista_pasos.setStyleSheet("""
            QListWidget {
                background: #252525;
                border: 1px solid #444;
                border-radius: 4px;
                font-size: 11px;
                color: #C0C0C0;
            }
            QListWidget::item { padding: 5px 8px; }
            QListWidget::item:selected {
                background: #1565C0;
                color: #FFFFFF;
                font-weight: bold;
            }
        """)
        for i, nombre in enumerate(pasos):
            item = QListWidgetItem(f'{i + 1}. {nombre}')
            lista_pasos.addItem(item)
        # el usuario puede hacer click para saltar a ese paso
        def _saltar_a_paso(row):
            # buscar el primer frame de ese paso
            for fi, pi in enumerate(pasos_de_frame):
                if pi == row:
                    slider.setValue(fi)
                    break
        lista_pasos.currentRowChanged.connect(_saltar_a_paso)

        col_lista = QVBoxLayout()
        lbl_lista_titulo = QLabel('Proceso de reconstruccion')
        lbl_lista_titulo.setAlignment(Qt.AlignCenter)
        lbl_lista_titulo.setStyleSheet('color: #90A4AE; font-size: 11px; padding-bottom: 2px;')
        col_lista.addWidget(lbl_lista_titulo)
        col_lista.addWidget(lista_pasos)

        cuerpo.addLayout(col_lista)

        # visor unico (ocupa el resto del espacio)
        col_visor = QVBoxLayout()
        lbl_visor_titulo = QLabel('Reconstruccion RZI')
        lbl_visor_titulo.setAlignment(Qt.AlignCenter)
        lbl_visor_titulo.setStyleSheet('color: #90A4AE; font-size: 11px; padding-bottom: 2px;')
        col_visor.addWidget(lbl_visor_titulo)
        visor = VisorImagen()
        col_visor.addWidget(visor, stretch=1)

        cuerpo.addLayout(col_visor, stretch=1)
        root.addLayout(cuerpo, stretch=1)

        # --- leyenda del frame actual ---
        lbl_leyenda = QLabel('')
        lbl_leyenda.setWordWrap(True)
        lbl_leyenda.setAlignment(Qt.AlignCenter)
        lbl_leyenda.setStyleSheet(
            'color: #E0E0E0; background: #2A2A2A; border-radius: 4px; '
            'padding: 6px; font-size: 11px;')
        lbl_leyenda.setMinimumHeight(44)
        root.addWidget(lbl_leyenda)

        # --- slider ---
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, n_frames - 1)
        slider.setValue(0)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #3A3A3A; height: 8px; border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3; width: 18px; margin: -5px 0;
                border-radius: 9px;
            }
        """)
        root.addWidget(slider)

        lbl_frame = QLabel(f'Frame 1 / {n_frames}')
        lbl_frame.setAlignment(Qt.AlignCenter)
        lbl_frame.setStyleSheet('color: #808080; font-size: 10px;')
        root.addWidget(lbl_frame)

        # --- controles ---
        ctrl = QHBoxLayout()
        lbl_vel = QLabel('Velocidad:')
        lbl_vel.setStyleSheet('color: #B0B0B0;')
        combo_vel = QComboBox()
        # ms por frame; valores mas lentos para que se vea todo el proceso.
        VELOCIDADES = {'Muy lento': 1200, 'Lento': 800, 'Normal': 450}
        combo_vel.addItems(list(VELOCIDADES.keys()))
        combo_vel.setCurrentText('Lento')

        btn_play  = QPushButton('Reproducir')
        btn_pause = QPushButton('Pausar')
        btn_close = QPushButton('Cerrar')

        ctrl.addWidget(lbl_vel)
        ctrl.addWidget(combo_vel)
        ctrl.addStretch()
        for b in (btn_play, btn_pause, btn_close):
            b.setMinimumHeight(32)
            ctrl.addWidget(b)
        root.addLayout(ctrl)

        # --- logica de animacion ---

        def actualizar_frame(idx):
            visor.establecer_imagen(frames[idx])
            paso_idx  = pasos_de_frame[idx]
            paso_name = pasos[paso_idx]
            sub       = subtitulos[idx]
            es_final  = idx == n_frames - 1
            sufijo    = '  [Reconstruccion final - identica al boton Reconstruir]' if es_final else ''
            lbl_leyenda.setText(
                f'Paso {paso_idx + 1}/{n_pasos}: {paso_name}  |  {sub}{sufijo}')
            lbl_frame.setText(f'Frame {idx + 1} / {n_frames}')
            # resaltar paso en la lista (bloquear la senal para no saltar)
            lista_pasos.blockSignals(True)
            lista_pasos.setCurrentRow(paso_idx)
            lista_pasos.blockSignals(False)

        slider.valueChanged.connect(actualizar_frame)

        timer = QTimer(dialogo)

        # Hold mas largo al comienzo de un paso nuevo para dar tiempo a leer.
        HOLD_PASO_MULT = 2

        def avanzar():
            idx = slider.value()
            if idx >= n_frames - 1:
                # llego al final: detener, no reiniciar
                timer.stop()
                return
            next_idx = idx + 1
            slider.setValue(next_idx)
            # ajustar intervalo: mas largo si es el primer frame de un paso nuevo
            base     = VELOCIDADES[combo_vel.currentText()]
            es_nuevo = (next_idx == 0 or
                        pasos_de_frame[next_idx] != pasos_de_frame[next_idx - 1])
            timer.setInterval(base * HOLD_PASO_MULT if es_nuevo else base)

        timer.timeout.connect(avanzar)

        def iniciar():
            # si estaba en el ultimo frame, reiniciar desde el principio
            if slider.value() >= n_frames - 1:
                slider.setValue(0)
            timer.start(VELOCIDADES[combo_vel.currentText()])

        def cambiar_velocidad():
            if timer.isActive():
                base     = VELOCIDADES[combo_vel.currentText()]
                idx      = slider.value()
                es_nuevo = (idx == 0 or
                            pasos_de_frame[idx] != pasos_de_frame[max(0, idx - 1)])
                timer.setInterval(base * HOLD_PASO_MULT if es_nuevo else base)

        combo_vel.currentTextChanged.connect(cambiar_velocidad)
        btn_play.clicked.connect(iniciar)
        btn_pause.clicked.connect(timer.stop)
        btn_close.clicked.connect(dialogo.accept)

        # mostrar primer frame y arrancar automaticamente
        actualizar_frame(0)
        iniciar()
        dialogo.exec()

    # helper compartido: ampliacion para mostrar (Reconstruir y Comparar)

    def _upscale_display(self, tile: np.ndarray, factor: int, params: dict) -> dict:
        """
        Ejecuta el pipeline screened y produce la imagen AMPLIADA lista para
        mostrar (sin reducir de vuelta al tamaño del tile ni incrustar).

        Aplica el stack de nitidez (difusion + shock + USM opcional) y, al
        final, anti-ringing contra una guia bicubica del mismo tamaño: esto
        recorta el overshoot sin perder el realce del modelo screened.

        Si 'tile' es a color (h,w,3), reconstruye con el enfoque
        luminancia-PDE: el canal Y pasa por el solver y el stack de nitidez
        completos, Cb/Cr se reescalan a bicubica y se recombinan
        (finalizar_display_color). Devuelve el mismo dict con 'display',
        'nn' y 'bicubica' en RGB.
        """
        es_color = tile.ndim == 3

        if es_color:
            y, cb, cr = rgb_a_ycbcr(tile)
            canal = y
        else:
            canal = tile

        res = upscale_poisson_screened(
            canal, factor,
            lam=params['lam'], beta=params['beta'], tau=params['tau'],
            usar_back_projection=params['back_projection'],
            usar_anti_ringing=params['anti_ringing'],
            usar_multiescala=params['multiescala'],
            back_gain=params['back_gain'],
            tol=params['tol'], max_iter=params['max_iter'],
        )

        th, tw = canal.shape
        sor_h_fit = th * factor
        sor_w_fit = tw * factor

        kwargs_realce = dict(
            usar_difusion=params['usar_difusion'],
            usar_shock=params['usar_shock'],
        )

        if es_color:
            bic = resize_rgb_float(tile, sor_w_fit, sor_h_fit, Image.BICUBIC)
            nn  = resize_rgb_float(tile, sor_w_fit, sor_h_fit, Image.NEAREST)
            display = finalizar_display_color(
                res['imagen_recon'], cb, cr, y, factor, params['usm_pct'],
                **kwargs_realce)
        else:
            pil_tile = Image.fromarray(
                (np.clip(tile, 0.0, 1.0) * 255).astype(np.uint8), mode='L')
            bic = np.array(pil_tile.resize((sor_w_fit, sor_h_fit), Image.BICUBIC),
                           dtype=np.float64) / 255.0
            nn = np.array(pil_tile.resize((sor_w_fit, sor_h_fit), Image.NEAREST),
                          dtype=np.float64) / 255.0
            display = finalizar_display(res['imagen_recon'], tile, factor,
                                          params['usm_pct'], **kwargs_realce)

        return {
            'display':      display,
            'res':          res,
            'nn':           nn,
            'bicubica':     bic,
            'forma_salida': (sor_h_fit, sor_w_fit),
        }

    # reconstruccion poisson sor

    # fraccion del lado menor de la imagen que ocupa el inset (~40 %)
    INSET_FRAC = 0.40

    def _componer_con_inset(self, patch: np.ndarray,
                             region: tuple) -> tuple:
        """
        Construye el contexto del panel derecho: copia de la imagen original
        A SU TAMANO REAL con el tile reconstruido pegado como INSET en la
        esquina mas lejana a la region del zoom.

        La region original NO se modifica (el usuario ve donde se trabajo).
        El inset muestra el tile SR a un tamano de INSET_FRAC * dimension
        de la imagen (preservando aspecto), con un margen de 2% respecto
        al borde.

        Devuelve (contexto, region_orig, region_inset) donde:
          - contexto:      ndarray float64 con original + inset pegado
          - region_orig:   (tx, ty, tw, th) en coords de imagen (para recuadro rojo)
          - region_inset:  (ix, iy, iw, ih) en coords de imagen (para recuadro verde)
        """
        tx, ty, tw, th = region
        orig = (self._imagen_color if self._imagen_color is not None
                else self._imagen_orig)
        H, W = orig.shape[:2]

        # clamp de la region a los limites de la imagen
        tx = max(0, min(tx, W - 1))
        ty = max(0, min(ty, H - 1))
        tw = max(1, min(tw, W - tx))
        th = max(1, min(th, H - ty))

        # tamano del inset preservando el aspecto del patch
        ph, pw = patch.shape[:2]
        aspect = pw / max(ph, 1)
        ih = max(1, round(self.INSET_FRAC * H))
        iw = max(1, round(ih * aspect))
        if iw > self.INSET_FRAC * W:
            iw = max(1, round(self.INSET_FRAC * W))
            ih = max(1, round(iw / aspect))

        # esquina mas lejana al centro de la region
        cx = tx + tw / 2
        cy = ty + th / 2
        m  = max(2, round(0.02 * min(W, H)))
        corners = {
            'TL': (m,         m),
            'TR': (W - iw - m, m),
            'BL': (m,         H - ih - m),
            'BR': (W - iw - m, H - ih - m),
        }
        ix, iy = max(corners.items(),
                     key=lambda kv: (kv[1][0] - cx) ** 2 + (kv[1][1] - cy) ** 2)[1]
        ix = int(max(0, min(ix, W - iw)))
        iy = int(max(0, min(iy, H - ih)))

        # pegar el inset
        base = orig.copy()
        if patch.ndim == 3:
            patch_inset = resize_rgb_float(patch, iw, ih, Image.LANCZOS)
        else:
            patch_inset = resize_float(patch, iw, ih, Image.LANCZOS)
        base[iy:iy + ih, ix:ix + iw] = np.clip(patch_inset, 0.0, 1.0)

        return base, (tx, ty, tw, th), (ix, iy, iw, ih)

    def _reconstruir(self):
        info = self._tile_desde_viewport()
        if info is None:
            return

        tile, tx, ty, tw, th = info
        params = self.panel_params.obtener_params()

        def _tarea():
            factor = params['factor']

            # upscaling con poisson "screened" + amplificacion adaptativa
            # de gradientes (lam_map ancla suavemente las muestras conocidas,
            # sin las discontinuidades duras que causaban el patron de
            # "puntitos" del modelo armonico anterior)
            disp = self._upscale_display(tile, factor, params)
            res  = disp['res']

            return {
                'imagen_upscale':  disp['display'],
                'resultado':       res,
                'tile_original':   tile,
                'region':          (tx, ty, tw, th),
                'forma_salida':    disp['forma_salida'],
                'factor':          factor,
                'detalle_nn':       disp['nn'],
                'detalle_bicubica': disp['bicubica'],
                'detalle_sor':      disp['display'],
            }

        ok = self._iniciar_hilo(_tarea)
        if not ok:
            return

        self.statusBar().showMessage(
            f'Reconstruyendo Poisson SOR (screened) region '
            f'({tx},{ty}) {tw}x{th} px  ->  x{params["factor"]}...')

        try:
            self._hilo.terminado.disconnect()
        except Exception:
            pass
        self._hilo.terminado.connect(self._mostrar_reconstruccion)

    def _mostrar_reconstruccion(self, datos: dict):
        self._resultado = datos
        res    = datos['resultado']
        factor = datos['factor']

        # componer: imagen original a tamano real + inset con la reconstruccion
        # en la esquina mas lejana a la region del zoom. el visor derecho muestra
        # ese contexto con:
        #   recuadro ROJO  -> ubicacion de la region del zoom en el original
        #   recuadro VERDE -> inset con el tile reconstruido a detalle completo
        # el tile SR full-res (tw*factor x th*factor) se guarda con 'Guardar imagen'.
        contexto, region, inset = self._componer_con_inset(
            datos['imagen_upscale'], datos['region'])
        datos['imagen_contexto'] = contexto
        self.visor_recon.establecer_imagen(contexto)
        self.visor_recon.marcar_region(*region, color='#FF5252')   # rojo: donde se trabajo
        self.visor_recon.marcar_region(*inset,  color='#00E676')   # verde: inset SR

        self.panel_res.mostrar_convergencia(res['historial'], res['omega'])

        tx, ty, tw, th = datos['region']
        nh, nw = datos['forma_salida']
        ix, iy, iw, ih = inset
        etapas_txt = 'x'.join(str(e) for e in res.get('etapas', [factor]))
        self.statusBar().showMessage(
            f"Poisson SOR screened x{factor} (etapas {etapas_txt})  |  "
            f"region ({tx},{ty}) {tw}x{th} px -> SR {nw}x{nh}  |  "
            f"inset en ({ix},{iy}) {iw}x{ih}  |  "
            f"iter={res['iteraciones']}  t={res['tiempo']:.3f}s  "
            f"omega={res['omega']:.4f}  "
            f"conv={'Si' if res['convergio'] else 'No (aumentar max_iter)'}")

        # dialogo de comparacion en detalle: zoom nativo vs bicubica vs SOR
        self._mostrar_dialogo_detalle(datos)

    def _mostrar_dialogo_detalle(self, datos: dict):
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton

        factor = datos['factor']
        tx, ty, tw, th = datos['region']

        dialogo = QDialog(self)
        dialogo.setWindowTitle('Comparacion en detalle: zoom nativo vs Poisson SOR')
        dialogo.setStyleSheet(ESTILO)
        dialogo.resize(1000, 460)

        layout = QVBoxLayout(dialogo)

        lbl_info = QLabel(
            f"Region ({tx},{ty}) {tw}x{th} px ampliada x{factor}. La columna "
            f"izquierda es el zoom nativo (vecino mas proximo, con bloques "
            f"pixelados). La derecha es la reconstruccion Poisson SOR "
            f"screened: anclaje suave a las muestras conocidas (lambda) mas "
            f"amplificacion adaptativa de gradientes en los bordes (beta), "
            f"sin los bloques del zoom nativo.")
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet('color: #B0B0B0; padding: 4px;')
        layout.addWidget(lbl_info)

        fila = QHBoxLayout()
        for titulo, color, img in (
            ('Zoom nativo (vecino mas proximo)', '#F44336', datos['detalle_nn']),
            ('Bicubica', '#FF9800', datos['detalle_bicubica']),
            (f'Poisson SOR screened x{factor}', '#4CAF50', datos['detalle_sor']),
        ):
            col = QVBoxLayout()
            lbl = QLabel(titulo)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(f'color: {color}; font-weight: bold; padding: 2px;')
            visor = VisorImagen()
            visor.setMinimumSize(280, 280)
            visor.establecer_imagen(img)
            col.addWidget(lbl)
            col.addWidget(visor)
            fila.addLayout(col)
        layout.addLayout(fila)

        btn_cerrar = QPushButton('Cerrar')
        btn_cerrar.setMinimumHeight(32)
        btn_cerrar.clicked.connect(dialogo.accept)
        layout.addWidget(btn_cerrar)

        dialogo.exec()

    # comparacion de 3 modelos

    def _comparar(self):
        info = self._tile_desde_viewport()
        if info is None:
            return
        tile, tx, ty, tw, th = info
        params = self.panel_params.obtener_params()

        def _tarea():
            factor = params['factor']

            # vista a mostrar: misma ampliacion real que "Reconstruir"
            # (modelo screened completo, con USM + anti-ringing). Si la
            # imagen es a color, disp['display'] queda en RGB.
            disp = self._upscale_display(tile, factor, params)
            res_poi = disp['res']

            # Las metricas (PSNR/SSIM/GMSD) y la tabla comparativa son 2D
            # (luminancia): si el tile es a color, se reduce a gris solo
            # para esta seccion de analisis.
            tile_gris = imagen_a_grises(tile) if tile.ndim == 3 else tile

            # tabla psnr/ssim/gmsd: metodologia degradar/factor -> reconstruir
            # con el modelo screened, sin realce (beta=1, "suave") y con el
            # realce configurado (beta del usuario), medidas contra el tile
            th_lr = max(2, th // factor)
            tw_lr = max(2, tw // factor)
            pil_tile = Image.fromarray(
                (np.clip(tile_gris, 0.0, 1.0) * 255).astype(np.uint8), mode='L')
            pil_lr = pil_tile.resize((tw_lr, th_lr), Image.LANCZOS)
            lr = np.array(pil_lr, dtype=np.float64) / 255.0

            bic = interpolacion_bicubica(tile_gris, factor)

            res_suave = upscale_poisson_screened(
                lr, factor, lam=params['lam'], beta=1.0, tau=params['tau'],
                usar_back_projection=params['back_projection'],
                usar_anti_ringing=params['anti_ringing'],
                usar_multiescala=params['multiescala'],
                back_gain=params['back_gain'],
                tol=params['tol'], max_iter=params['max_iter'])
            res_realce = upscale_poisson_screened(
                lr, factor, lam=params['lam'], beta=params['beta'], tau=params['tau'],
                usar_back_projection=params['back_projection'],
                usar_anti_ringing=params['anti_ringing'],
                usar_multiescala=params['multiescala'],
                back_gain=params['back_gain'],
                tol=params['tol'], max_iter=params['max_iter'])

            def _a_tamano_tile(arr):
                pil = Image.fromarray(
                    (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode='L')
                pil = pil.resize((tw, th), Image.LANCZOS)
                return np.array(pil, dtype=np.float64) / 255.0

            suave  = _a_tamano_tile(res_suave['imagen_recon'])
            realce = _a_tamano_tile(res_realce['imagen_recon'])

            tabla = tabla_comparativa(tile_gris, bic, suave, realce)

            return {
                'imagen_completa': disp['display'],
                'imagen_upscale':  disp['display'],
                'original': tile_gris,
                'bicubica': bic,
                'laplace': suave,
                'poisson': realce,
                'tabla': tabla,
                'res_poi': res_poi,
                'region': (tx, ty, tw, th),
            }

        ok = self._iniciar_hilo(_tarea)
        if not ok:
            return
        self.statusBar().showMessage('Comparando 3 modelos (screened)...')
        self._hilo.terminado.connect(self._mostrar_comparacion)

    def _mostrar_comparacion(self, datos: dict):
        self._resultado = datos

        # ampliacion real (igual que Reconstruir), sin embed ni copia de zoom
        self.visor_recon.establecer_imagen(datos['imagen_completa'])

        self.panel_res.mostrar_comparacion(
            datos['original'], datos['bicubica'],
            datos['laplace'],  datos['poisson'],
            datos['tabla'], res_poi=datos['res_poi'])
        self.panel_res.mostrar_convergencia(
            datos['res_poi']['historial'], datos['res_poi']['omega'])

        self.statusBar().showMessage('Comparacion lista.')

    # barrido omega

    def _barrido_omega(self):
        info = self._tile_desde_viewport()
        if info is None:
            return

        tile = info[0]
        if tile.ndim == 3:
            tile = imagen_a_grises(tile)
        params = self.panel_params.obtener_params()

        def _tarea():
            problema = construir_problema(tile, params['factor'])
            return _barrido(problema, tol=1e-3, max_iter=500, n_puntos=38)

        ok = self._iniciar_hilo(_tarea)
        if not ok:
            return

        self.statusBar().showMessage('Ejecutando barrido de omega...')
        self._hilo.terminado.connect(self._en_barrido_terminado)

    def _en_barrido_terminado(self, resultado: dict):
        self.panel_res.mostrar_barrido_omega(resultado)
        self.statusBar().showMessage(
            f"Barrido omega  |  teorico={resultado['omega_teorico']:.4f}  |  "
            f"empirico={resultado['omegas'][resultado['iteraciones'].index(min(resultado['iteraciones']))]:.4f}")

    # benchmark numba vs python

    def _benchmark(self):
        if self._imagen_orig is None:
            QMessageBox.warning(self, 'Sin imagen', 'Cargue una imagen primero.')
            return

        img = self._imagen_orig

        def _tarea():
            resultados = []
            for size in (64, 128, 256):
                nr, nc = img.shape
                if nr < size or nc < size:
                    continue
                cy, cx = nr // 2, nc // 2
                t = size // 2
                tile     = img[cy-t:cy+t, cx-t:cx+t]
                problema = construir_problema(tile, factor=4)
                bench    = benchmark_numba_vs_python(problema, tol=1e-3, max_iter=100)
                resultados.append(bench)
            return resultados

        ok = self._iniciar_hilo(_tarea)
        if not ok:
            return

        self.statusBar().showMessage('Ejecutando benchmark Numba vs Python...')
        self._hilo.terminado.connect(self.panel_res.mostrar_benchmark)

    # exportacion

    def _guardar_imagen(self):
        if self._resultado is None:
            QMessageBox.warning(self, 'Sin resultado',
                                'Ejecute una reconstruccion primero.')
            return

        img = None
        # Busqueda explicita para evitar evaluar arrays de NumPy como booleanos (ValueError).
        # 'imagen_upscale' contiene el tile reconstruido a resolucion completa
        # (tw*factor x th*factor), que es el resultado de interes para guardar.
        for clave in ('imagen_upscale', 'poisson', 'imagen_completa'):
            if clave in self._resultado and self._resultado[clave] is not None:
                img = self._resultado[clave]
                break

        if img is None and 'resultado' in self._resultado:
            res_dict = self._resultado['resultado']
            if isinstance(res_dict, dict) and 'imagen_recon' in res_dict:
                img = res_dict['imagen_recon']

        if img is None:
            return

        ruta, _ = QFileDialog.getSaveFileName(
            self, 'Guardar imagen reconstruida', 'poisson_recon.png',
            'PNG (*.png);;JPEG (*.jpg)')
        if ruta:
            guardar_imagen(img, ruta)
            QMessageBox.information(self, 'Guardado', f'Imagen guardada en:\n{ruta}')

    def _guardar_grafica(self):
        tab = self.panel_res.tabs.currentWidget()
        canvas = tab.findChild(FigureCanvas) if tab else None
        if canvas is None:
            return
        ruta, _ = QFileDialog.getSaveFileName(
            self, 'Guardar grafica', 'grafica.png', 'PNG (*.png);;PDF (*.pdf)')
        if ruta:
            canvas.figure.savefig(ruta, dpi=150, bbox_inches='tight')
            QMessageBox.information(self, 'Guardado', f'Grafica guardada en:\n{ruta}')

    def _exportar_csv(self):
        if self._resultado is None or 'tabla' not in self._resultado:
            QMessageBox.warning(self, 'Sin datos',
                                'Ejecute "Comparar 3 modelos" primero.')
            return
        ruta, _ = QFileDialog.getSaveFileName(
            self, 'Exportar CSV', 'resultados.csv', 'CSV (*.csv)')
        if ruta:
            exportar_csv(self._resultado['tabla'],
                         self._resultado.get('res_poi', {}), ruta)
            QMessageBox.information(self, 'Exportado', f'Datos exportados a:\n{ruta}')
