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
from PIL import Image, ImageFilter

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
from core.poisson_builder import (construir_problema, construir_problema_upscale,
                                   imagen_a_grises, calcular_omega_optimo)
from core.sor_solver import (resolver, barrido_omega as _barrido,
                              benchmark_numba_vs_python, NUMBA_DISPONIBLE,
                              _calentamiento, resolver_con_snapshots)
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
        self.spin_factor.setRange(2, 8)
        self.spin_factor.setValue(4)
        self.spin_factor.setToolTip(
            'con factor=4 se conoce 1 de cada 16 pixeles')
        gl.addWidget(self.spin_factor, 0, 1)

        lay.addWidget(g_entrada)

        
        g_solver = QGroupBox('Solver SOR Red-Black')
        sl = QGridLayout(g_solver)
        sl.setSpacing(6)

        sl.addWidget(QLabel('Tolerancia:'), 0, 0)
        self.spin_tol = QDoubleSpinBox()
        self.spin_tol.setDecimals(8)
        self.spin_tol.setRange(1e-10, 1e-2)
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
        self.spin_usm.setValue(150)
        self.spin_usm.setSingleStep(25)
        self.spin_usm.setToolTip('0 = sin realce; 150 = nitidez media; 300 = maxima')
        nl.addWidget(self.spin_usm, 0, 1)
        lay.addWidget(g_sharp)

        
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
        self.tabla_metricas = QTableWidget(0, 4)
        self.tabla_metricas.setHorizontalHeaderLabels(
            ['Modelo', 'PSNR (dB)', 'SSIM', 'Solver'])
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

    def mostrar_comparacion(self, original, bicubica, laplace, poisson, tabla):
        fig = grafica_comparacion_4modelos(original, bicubica, laplace, poisson, tabla)
        self._renderizar_en(self.canvas_comp, fig)
        plt.close(fig)
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
            self.tabla_metricas.setItem(row, 0, QTableWidgetItem(fila.get('modelo', '')))
            self.tabla_metricas.setItem(row, 1, QTableWidgetItem(
                'inf' if psnr_v == float('inf') else f'{psnr_v:.2f}'))
            self.tabla_metricas.setItem(row, 2, QTableWidgetItem(f'{ssim_v:.4f}'))
            self.tabla_metricas.setItem(row, 3, QTableWidgetItem(info))
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
                              aspect='auto', interpolation='nearest')

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
        self._imagen_orig = None   # ndarray float64 [0,1] grises, imagen completa
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
        lbl_orig = QLabel('Original  (zoom con rueda, reconstruye la region visible)')
        lbl_orig.setStyleSheet('color: #2196F3; font-weight: bold; padding: 2px;')
        lbl_orig.setAlignment(Qt.AlignCenter)
        self.visor_orig = VisorImagen()
        col_orig.addWidget(lbl_orig)
        col_orig.addWidget(self.visor_orig)

        # visor derecho: imagen reconstruida / agrandada
        col_recon = QVBoxLayout()
        lbl_recon = QLabel('Reconstruida  (region visible refinada con SOR armonico, imagen completa)')
        lbl_recon.setStyleSheet('color: #4CAF50; font-weight: bold; padding: 2px;')
        lbl_recon.setAlignment(Qt.AlignCenter)
        self.visor_recon = VisorImagen()
        col_recon.addWidget(lbl_recon)
        col_recon.addWidget(self.visor_recon)

        visores_lay.addLayout(col_orig)
        visores_lay.addLayout(col_recon)
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
            pil_img = Image.open(ruta).convert('RGB')
            arr = np.asarray(pil_img, dtype=np.float64) / 255.0
            self._imagen_orig = imagen_a_grises(arr)
            self._resultado   = None
            self.visor_orig.establecer_imagen(self._imagen_orig)
            self.visor_recon.limpiar()
            self.statusBar().showMessage(
                f'Imagen cargada: {os.path.basename(ruta)}  '
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
        tile = self._imagen_orig[y:y+h, x:x+w]

        if tile.size < 64:
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

        radio = 60
        x0 = max(0, x_img - radio)
        y0 = max(0, y_img - radio)
        x1 = min(self._imagen_orig.shape[1], x_img + radio)
        y1 = min(self._imagen_orig.shape[0], y_img + radio)

        tile = self._imagen_orig[y0:y1, x0:x1]
        if tile.size < 64:
            return

        factor = 3

        def _tarea_lupa(t=tile, f=factor):
            problema = construir_problema_upscale(t, f)
            res = resolver(problema, ecuacion='laplace', max_iter=400, tol=1e-4)
            return res['imagen_recon']

        self._hilo_lupa = HiloCalculo(_tarea_lupa)
        self._hilo_lupa.terminado.connect(self._mostrar_lupa)
        self._hilo_lupa.start()
        
        def _mostrar_lupa(self, imagen_upscaled):
        # por si el usuario desactivo el modo lupa mientras se calculaba        
            if not self._modo_lupa_activo:
                return
        self.visor_recon.establecer_imagen(imagen_upscaled)
        self.statusBar().showMessage(f'Lupa actualizada: {imagen_upscaled.shape[1]}x{imagen_upscaled.shape[0]} px')

    # animacion de convergencia
    def _animar_convergencia(self):
        info = self._tile_desde_viewport()
        if info is None:
            return

        tile = info[0]
        params = self.panel_params.obtener_params()

        def _tarea_animacion(t=tile, p=params):
            problema = construir_problema_upscale(t, p['factor'])
            return resolver_con_snapshots(
                problema,
                omega=p['omega'],
                tol=p['tol'],
                max_iter=min(p['max_iter'], 600),
                ecuacion='laplace',
                cada_n_iter=15
            )

        ok = self._iniciar_hilo(_tarea_animacion)
        if not ok:
            return

        self.statusBar().showMessage('Generando animacion de convergencia...')
        self._hilo.terminado.connect(self._mostrar_dialogo_animacion)

    def _mostrar_dialogo_animacion(self, resultado: dict):
        from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout,
                                       QPushButton, QLabel, QSlider)
        from PySide6.QtCore import QTimer, Qt

        snapshots = resultado.get('snapshots', [])
        if not snapshots:
            QMessageBox.warning(self, 'Error', 'No se generaron snapshots.')
            return

        dialogo = QDialog(self)
        dialogo.setWindowTitle('Animacion de Convergencia SOR Red-Black')
        dialogo.resize(700, 650)
        dialogo.setStyleSheet(ESTILO)

        layout = QVBoxLayout(dialogo)

        lbl_info = QLabel(f"Snapshots: {len(snapshots)} | "
                          f"Iteraciones totales: {resultado['iteraciones']} | "
                          f"Tiempo: {resultado['tiempo']:.3f}s")
        lbl_info.setStyleSheet('color: #2196F3; font-weight: bold; padding: 4px;')
        layout.addWidget(lbl_info)

        visor_temp = VisorImagen()
        layout.addWidget(visor_temp)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, len(snapshots) - 1)
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
        layout.addWidget(slider)

        lbl_frame = QLabel('Frame 1 / ' + str(len(snapshots)))
        lbl_frame.setAlignment(Qt.AlignCenter)
        lbl_frame.setStyleSheet('color: #B0B0B0;')
        layout.addWidget(lbl_frame)

        btn_layout = QHBoxLayout()
        btn_play  = QPushButton('▶ Reproducir')
        btn_pause = QPushButton('⏸ Pausar')
        btn_close = QPushButton('Cerrar')
        for b in (btn_play, btn_pause, btn_close):
            b.setMinimumHeight(32)
            btn_layout.addWidget(b)
        layout.addLayout(btn_layout)

        def actualizar_frame(idx):
            visor_temp.establecer_imagen(snapshots[idx])
            iter_aprox = idx * resultado.get('cada_n_iter', 15)
            lbl_frame.setText(f'Frame {idx + 1} / {len(snapshots)}  |  '
                              f'Iteracion aproximada: {iter_aprox}')

        slider.valueChanged.connect(actualizar_frame)

        timer = QTimer(dialogo)
        timer.timeout.connect(
            lambda: slider.setValue((slider.value() + 1) % len(snapshots))
        )

        btn_play.clicked.connect(lambda: timer.start(166))
        btn_pause.clicked.connect(timer.stop)
        btn_close.clicked.connect(dialogo.accept)

        actualizar_frame(0)
        dialogo.exec()

    # reconstruccion poisson sor

    def _reconstruir(self):
        info = self._tile_desde_viewport()
        if info is None:
            return

        tile, tx, ty, tw, th = info
        params       = self.panel_params.obtener_params()
        img_completa = self._imagen_orig.copy()

        def _tarea():
            factor = params['factor']

            # upscaling con laplace armonico (b=0)
            problema = construir_problema_upscale(tile, factor)
            res      = resolver(problema,
                                omega=params['omega'],
                                tol=params['tol'],
                                max_iter=params['max_iter'],
                                ecuacion='laplace')
            new_h, new_w = problema['forma']
            big = res['imagen_recon']

            # USM aplicado a resolucion ampliada; radius escalado por factor
            # para equivalencia perceptual (radio fisico = 2 px del tile original)
            pil_big = Image.fromarray(
                (np.clip(big, 0.0, 1.0) * 255).astype(np.uint8), mode='L')
            usm_pct = params['usm_pct']
            if usm_pct > 0:
                pil_big = pil_big.filter(
                    ImageFilter.UnsharpMask(radius=2 * factor,
                                           percent=usm_pct, threshold=2))

            # ajustar SOR a tamano exacto th*factor x tw*factor para embedding limpio
            sor_h_fit = th * factor
            sor_w_fit = tw * factor
            if (new_w, new_h) != (sor_w_fit, sor_h_fit):
                pil_big = pil_big.resize((sor_w_fit, sor_h_fit), Image.LANCZOS)
            sor_fit = np.array(pil_big, dtype=np.float64) / 255.0

            # contexto: extraer region con padding y upscalar con BICUBIC
            # asi el visor derecho muestra la region reconstruida + entorno
            pad = max(20, min(th // 3, tw // 3))
            ctx_x0 = max(0, tx - pad)
            ctx_y0 = max(0, ty - pad)
            ctx_x1 = min(img_completa.shape[1], tx + tw + pad)
            ctx_y1 = min(img_completa.shape[0], ty + th + pad)
            ctx_orig = img_completa[ctx_y0:ctx_y1, ctx_x0:ctx_x1]
            ctx_h_o, ctx_w_o = ctx_orig.shape

            pil_ctx     = Image.fromarray((ctx_orig * 255).astype(np.uint8), mode='L')
            pil_ctx_big = pil_ctx.resize((ctx_w_o * factor, ctx_h_o * factor),
                                         Image.BICUBIC)
            ctx_big = np.array(pil_ctx_big, dtype=np.float64) / 255.0

            # incrustar resultado SOR en el contexto bicubico
            oy = (ty - ctx_y0) * factor
            ox = (tx - ctx_x0) * factor
            ctx_big[oy:oy + sor_h_fit, ox:ox + sor_w_fit] = sor_fit

            return {
                'imagen_visor':  ctx_big,
                'imagen_upscale': big,
                'resultado':     res,
                'tile_original': tile,
                'region':        (tx, ty, tw, th),
                'forma_salida':  (new_h, new_w),
                'factor':        factor,
            }

        ok = self._iniciar_hilo(_tarea)
        if not ok:
            return

        self.statusBar().showMessage(
            f'Reconstruyendo Poisson SOR (armonico) region '
            f'({tx},{ty}) {tw}x{th} px  ->  x{params["factor"]}...')

        self._hilo.terminado.connect(self._mostrar_reconstruccion)

    def _mostrar_reconstruccion(self, datos: dict):
        self._resultado = datos
        res    = datos['resultado']
        factor = datos['factor']

        # mostrar composite upscalado (SOR en centro + contexto bicubico)
        # ajustar_a_ventana() ya es llamado por establecer_imagen; no copiar zoom
        # porque el visor derecho muestra la region x factor mas grande
        self.visor_recon.establecer_imagen(datos['imagen_visor'])

        self.panel_res.mostrar_convergencia(res['historial'], res['omega'])

        tx, ty, tw, th = datos['region']
        nh, nw = datos['forma_salida']
        self.statusBar().showMessage(
            f"SOR armonico x{factor}  |  {tw}x{th} px -> {nw}x{nh}  "
            f"|  iter={res['iteraciones']}  t={res['tiempo']:.3f}s  "
            f"omega={res['omega']:.4f}  "
            f"conv={'Si' if res['convergio'] else 'No (aumentar max_iter)'}")

    # comparacion de 3 modelos

    def _comparar(self):
        info = self._tile_desde_viewport()
        if info is None:
            return

        tile, tx, ty, tw, th = info
        params     = self.panel_params.obtener_params()
        img_completa = self._imagen_orig.copy()

        def _tarea():
            problema = construir_problema(tile, params['factor'])
            bic      = interpolacion_bicubica(tile, params['factor'])
            res_lap  = resolver(problema, omega=params['omega'],
                                tol=params['tol'], max_iter=params['max_iter'],
                                ecuacion='laplace')
            res_poi  = resolver(problema, omega=params['omega'],
                                tol=params['tol'], max_iter=params['max_iter'],
                                ecuacion='poisson')
            tabla = tabla_comparativa(tile, bic,
                                       res_lap['imagen_recon'],
                                       res_poi['imagen_recon'])
            # reconstruccion completa con el mejor metodo (poisson)
            img_recon = img_completa.copy()
            img_recon[ty:ty+th, tx:tx+tw] = res_poi['imagen_recon']
            return {
                'imagen_completa': img_recon,
                'original': tile,
                'bicubica': bic,
                'laplace':  res_lap['imagen_recon'],
                'poisson':  res_poi['imagen_recon'],
                'tabla':    tabla,
                'res_poi':  res_poi,
                'region':   (tx, ty, tw, th),
            }

        ok = self._iniciar_hilo(_tarea)
        if not ok:
            return

        self.statusBar().showMessage('Comparando 3 modelos...')
        self._hilo.terminado.connect(self._mostrar_comparacion)

    def _mostrar_comparacion(self, datos: dict):
        self._resultado = datos

        self.visor_recon.establecer_imagen(datos['imagen_completa'])
        self.visor_recon.copiar_estado_zoom(self.visor_orig)

        self.panel_res.mostrar_comparacion(
            datos['original'], datos['bicubica'],
            datos['laplace'],  datos['poisson'],
            datos['tabla'])
        self.panel_res.mostrar_metricas(datos['tabla'], datos['res_poi'])
        self.panel_res.mostrar_convergencia(
            datos['res_poi']['historial'], datos['res_poi']['omega'])

        self.statusBar().showMessage('Comparacion lista.')

    # barrido omega

    def _barrido_omega(self):
        info = self._tile_desde_viewport()
        if info is None:
            return

        tile = info[0]
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
        # Busqueda explicita para evitar evaluar arrays de NumPy como booleanos (ValueError)
        for clave in ('poisson', 'imagen_upscale', 'imagen_completa'):
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
