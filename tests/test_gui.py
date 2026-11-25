"""
Prueba de humo de la interfaz (sin pantalla, QT_QPA_PLATFORM=offscreen).
Se omite si PySide6 no esta instalado o no puede iniciar.
"""

import os
import sys
import time

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from PySide6.QtWidgets import QApplication, QDialog
    from PySide6.QtCore import QTimer
except Exception:  # PySide6 ausente o sin librerias graficas del sistema
    pytest.skip('PySide6 no disponible', allow_module_level=True)

from PIL import Image


def _esperar(app, ventana, limite=120):
    t0 = time.time()
    while ventana._hilo is not None and ventana._hilo.isRunning():
        app.processEvents()
        time.sleep(0.02)
        assert time.time() - t0 < limite, 'el calculo tardo demasiado'
    for _ in range(10):
        app.processEvents()


def test_flujo_completo_de_la_interfaz(tmp_path):
    from ui.main_window import VentanaPrincipal, ESTILO, PanelResultados

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(ESTILO)
    ventana = VentanaPrincipal()
    ventana.show()
    errores = []
    ventana._en_error = lambda resumen, detalle='': errores.append(resumen)

    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 160)
    img = (0.5 + 0.4 * np.sin(12 * x)[None, :] * np.cos(9 * x)[:, None]
           + 0.05 * rng.standard_normal((160, 160)))
    ruta = tmp_path / 'prueba.png'
    Image.fromarray(np.uint8(np.clip(img, 0, 1) * 255)).save(ruta)
    ventana.cargar_imagen(str(ruta))

    # cerrar automaticamente los dialogos modales que abren Reconstruir/Animar
    def cerrar_dialogos():
        for w in app.topLevelWidgets():
            if isinstance(w, QDialog) and w.isVisible():
                w.accept()
    timer = QTimer()
    timer.timeout.connect(cerrar_dialogos)
    timer.start(200)

    ventana._reconstruir()
    _esperar(app, ventana)
    assert ventana._resultado['imagen_upscale'].shape == (640, 640)

    # probar vista contextual con inset en vivo sin recalcular
    ventana.chk_vista_inset.setChecked(True)
    assert len(ventana.visor_recon._marcas) == 2
    ventana.chk_vista_inset.setChecked(False)
    assert len(ventana.visor_recon._marcas) == 0

    ventana._comparar()
    _esperar(app, ventana)
    assert len(ventana._resultado['tabla']) == 4

    ventana._barrido_omega()
    _esperar(app, ventana)
    assert PanelResultados.TAB_OMEGA in ventana.panel_res.figuras
    # la pestana deja de mostrar el texto de "estado vacio" y muestra la grafica
    assert ventana.panel_res.pilas[PanelResultados.TAB_OMEGA].currentIndex() == 1

    ventana._benchmark()
    _esperar(app, ventana)
    assert PanelResultados.TAB_BENCH in ventana.panel_res.figuras

    ventana._animar_convergencia()
    _esperar(app, ventana)

    timer.stop()
    assert errores == []
    ventana.close()


def _ventana(app):
    from ui.main_window import VentanaPrincipal, ESTILO
    app.setStyleSheet(ESTILO)
    ventana = VentanaPrincipal()
    ventana.show()
    return ventana


def test_estados_vacios_y_presets():
    from ui.main_window import PRESETS, PERSONALIZADO
    app = QApplication.instance() or QApplication(sys.argv)
    ventana = _ventana(app)
    pr = ventana.panel_res
    for idx in range(4):
        assert pr.pilas[idx].currentIndex() == 0      # texto de ayuda visible
    pp = ventana.panel_params
    pp.combo_preset.setCurrentText('Rapido')
    assert pp.obtener_params()['ciclos'] == PRESETS['Rapido']['ciclos']
    pp.spin_beta.setValue(2.2)                         # cambio manual
    assert pp.combo_preset.currentText() == PERSONALIZADO
    ventana.close()


def test_comparar_con_foto_grande(tmp_path):
    """Antes fallaba con 'operands could not be broadcast' en fotos grandes."""
    app = QApplication.instance() or QApplication(sys.argv)
    ventana = _ventana(app)
    errores = []
    ventana._en_error = lambda resumen, detalle='': errores.append(resumen)
    rng = np.random.default_rng(1)
    base = rng.random((120, 160))
    grande = np.asarray(Image.fromarray(np.uint8(base * 255)).resize((1600, 1200),
                                                                     Image.BICUBIC))
    ruta = tmp_path / 'grande.png'
    Image.fromarray(grande).save(ruta)
    ventana.cargar_imagen(str(ruta))
    ventana._comparar()
    _esperar(app, ventana)
    assert errores == []
    assert len(ventana._resultado['tabla']) == 4
    assert ventana.panel_res.tabla.rowCount() == 4
    ventana.close()


def test_errores_en_slots_se_muestran(monkeypatch):
    """Una excepcion al mostrar un resultado ya no queda en silencio."""
    import ui.main_window as mw
    app = QApplication.instance() or QApplication(sys.argv)
    ventana = _ventana(app)
    vistos = []
    monkeypatch.setattr(mw, 'mostrar_error',
                        lambda parent, titulo, resumen, detalle='': vistos.append(resumen))
    ventana._en_barrido_terminado({'omegas': None})   # datos invalidos a proposito
    assert len(vistos) == 1 and 'KeyError' in vistos[0]
    ventana.close()
