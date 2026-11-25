"""
Pruebas automaticas del proyecto RZI.

Ejecutar desde la raiz del proyecto:
    python -m pytest tests -q
"""

import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

from core.poisson_builder import (construir_problema, construir_problema_gradiente,
                                  divergence, gradiente_adelante,
                                  estimar_omega_optimo, calcular_omega_optimo,
                                  omega_opt_rectangular)
from core.sor_solver import (resolver, resolver_screened,
                             resolver_screened_con_snapshots,
                             barrido_omega, benchmark_numba_vs_python)
from core.upscaler import (upscale, upscale_rzi, generar_frames_pipeline,
                           back_projection, error_consistencia)
from core import upscaler as modulo_upscaler
from core.sor_solver import CalculoCancelado
from services.metrics import (degradar, interpolacion_bicubica, psnr, ssim, gmsd,
                              tabla_comparativa)
from services.exporter import exportar_csv

skimage_data = pytest.importorskip('skimage.data')


@pytest.fixture(scope='module')
def camara():
    return skimage_data.camera().astype(np.float64) / 255.0


def test_divergencia_de_gradiente_es_laplaciano_5_puntos():
    rng = np.random.default_rng(1)
    G = rng.random((12, 15))
    lap = divergence(*gradiente_adelante(G))
    esperado = G[1:-1, 2:] + G[1:-1, :-2] + G[2:, 1:-1] + G[:-2, 1:-1] - 4 * G[1:-1, 1:-1]
    assert np.allclose(lap[1:-1, 1:-1], esperado)


def test_signo_de_b_poisson_mejor_que_laplace(camara):
    # Con el signo correcto, guiar con la imagen (Poisson) no puede ser peor
    # que no guiar (Laplace). Con el signo invertido Poisson quedaba ~2 dB peor.
    tile = camara[100:228, 200:328]
    p = construir_problema(tile, 4)
    r_poi = resolver(p, tol=1e-7, max_iter=4000, ecuacion='poisson')
    r_lap = resolver(p, tol=1e-7, max_iter=4000, ecuacion='laplace')
    assert r_poi['convergio'] and r_lap['convergio']
    assert psnr(tile, r_poi['imagen_recon']) > psnr(tile, r_lap['imagen_recon'])


def test_omega_estimado_coincide_con_minimo_del_barrido(camara):
    tile = camara[150:278, 150:278]
    p = construir_problema(tile, 4)
    r = barrido_omega(p, tol=1e-3, max_iter=500, n_puntos=45)
    assert abs(r['omega_empirico'] - p['omega_opt']) < 0.05
    # la formula de malla completa NO aplica con mascara: queda lejos
    assert p['omega_malla_completa'] - r['omega_empirico'] > 0.2


def test_omega_malla_completa_formula_clasica():
    n = 50
    assert np.isclose(calcular_omega_optimo(n + 2, n + 2),
                      2 / (1 + np.sin(np.pi / (n + 1))))
    # el termino lambda reduce el radio espectral y por tanto omega
    assert omega_opt_rectangular(100, 100, 0.3) < omega_opt_rectangular(100, 100, 0.0)


def test_estimacion_omega_sin_mascara_interior_coincide_con_formula():
    m = np.zeros((40, 40), dtype=bool)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = True
    assert abs(estimar_omega_optimo(m, iters=3000) - calcular_omega_optimo(40, 40)) < 0.02


def test_screened_converge_y_satisface_la_ecuacion():
    rng = np.random.default_rng(2)
    t = rng.random((40, 50))
    p = construir_problema_gradiente(t, lam=0.3)
    r = resolver_screened(p, tol=1e-10, max_iter=2000)
    assert r['convergio']
    u = r['imagen_recon']
    lhs = (4.3 * u[1:-1, 1:-1] - u[:-2, 1:-1] - u[2:, 1:-1] - u[1:-1, :-2] - u[1:-1, 2:])
    assert np.allclose(lhs, p['b'][1:-1, 1:-1], atol=1e-7)


def test_kernel_numba_igual_a_python_puro(camara):
    p = construir_problema(camara[:48, :48], 4)
    r = benchmark_numba_vs_python(p, n_iter=5)
    assert r['diferencia_max'] < 1e-10


@pytest.mark.parametrize('factor', [2, 3, 4])
def test_salida_de_tamano_exacto(factor):
    lr = np.random.default_rng(3).random((21, 34))
    assert upscale_rzi(lr, factor, ciclos=2)['imagen_recon'].shape == (21 * factor, 34 * factor)
    rgb = np.random.default_rng(4).random((10, 12, 3))
    assert upscale(rgb, factor, ciclos=2)['imagen_recon'].shape == (10 * factor, 12 * factor, 3)


def test_back_projection_reduce_error_de_consistencia(camara):
    _, lr = degradar(camara[:128, :128], 4)
    u0 = interpolacion_bicubica(lr, 4)
    u1 = back_projection(u0, lr, n_pass=5)
    assert error_consistencia(u1, lr) < 0.2 * error_consistencia(u0, lr)


@pytest.mark.parametrize('factor', [2, 4])
def test_rzi_supera_a_la_bicubica(camara, factor):
    hr, lr = degradar(camara, factor)
    bic = interpolacion_bicubica(lr, factor)
    rzi = upscale_rzi(lr, factor)['imagen_recon']
    assert psnr(hr, rzi) > psnr(hr, bic) + 0.5
    assert ssim(hr, rzi) > ssim(hr, bic)
    assert gmsd(hr, rzi) < gmsd(hr, bic)


def test_csv_incluye_datos_del_solver(tmp_path, camara):
    hr, lr = degradar(camara[:128, :128], 2)
    res = upscale_rzi(lr, 2, ciclos=2)
    tabla = tabla_comparativa(hr, {'Bicubica': interpolacion_bicubica(lr, 2),
                                   'RZI': res['imagen_recon']}, modelos_rzi=('RZI',))
    ruta = tmp_path / 'r.csv'
    exportar_csv(tabla, res, str(ruta))
    filas = list(csv.DictReader(open(ruta, encoding='utf-8')))
    assert filas[0]['Iteraciones_SOR'] == '--'
    assert filas[1]['Iteraciones_SOR'] == str(res['iteraciones'])
    assert filas[1]['Omega'] != '--'


def test_resolver_screened_con_snapshots_equivale_a_resolver():
    rng = np.random.default_rng(7)
    t = rng.random((32, 36))
    p = construir_problema_gradiente(t, lam=0.3)
    r_ref = resolver_screened(p, tol=1e-6, max_iter=100)
    r_snap = resolver_screened_con_snapshots(p, tol=1e-6, max_iter=100, cada_n_iter=25)
    assert r_snap['iteraciones'] == r_ref['iteraciones']
    assert r_snap['convergio'] == r_ref['convergio']
    assert len(r_snap['snapshots']) >= 1
    assert np.allclose(r_snap['imagen_recon'], r_ref['imagen_recon'], atol=1e-12)


def test_generar_frames_pipeline_coincide_con_upscale(camara):
    tile = camara[120:152, 200:232]
    res = upscale(tile, 3, ciclos=3, sor_iters=80, realce=0.4, usar_anti_ringing=True)
    gen = generar_frames_pipeline(tile, 3, ciclos=3, sor_iters=80,
                                  realce=0.4, usar_anti_ringing=True)
    assert len(gen['frames']) > 4
    assert len(gen['frames']) == len(gen['pasos_de_frame']) == len(gen['subtitulos'])
    assert gen['frames'][-1].shape == (96, 96)
    assert np.allclose(gen['frames'][-1], res['imagen_recon'], atol=1e-12)

    rgb = np.random.default_rng(9).random((12, 14, 3))
    res_rgb = upscale(rgb, 2, ciclos=2, sor_iters=40)
    gen_rgb = generar_frames_pipeline(rgb, 2, ciclos=2, sor_iters=40)
    assert gen_rgb['frames'][-1].shape == (24, 28, 3)
    assert np.allclose(gen_rgb['frames'][-1], res_rgb['imagen_recon'], atol=1e-12)


@pytest.mark.parametrize('factor', [2, 4])
def test_estable_con_muchos_ciclos(camara, factor):
    # con amplificacion constante (rho=1) 30 ciclos rompian la imagen (-6 dB en
    # promedio); con amplificacion decreciente sigue por encima de la bicubica
    hr, lr = degradar(camara, factor)
    bic = psnr(hr, interpolacion_bicubica(lr, factor))
    assert psnr(hr, upscale_rzi(lr, factor, ciclos=30)['imagen_recon']) > bic + 0.5


def test_por_bloques_igual_que_completo(camara, monkeypatch):
    hr, lr = degradar(camara[:256, :256], 2)
    completo = upscale_rzi(lr, 2, ciclos=3)
    monkeypatch.setattr(modulo_upscaler, 'MAX_PIXELES_BLOQUE', 5_000)
    monkeypatch.setattr(modulo_upscaler, 'BLOQUE_LR', 48)
    bloques = upscale_rzi(lr, 2, ciclos=3)
    assert bloques['por_bloques'] and bloques['bloques'] > 1
    assert bloques['imagen_recon'].shape == completo['imagen_recon'].shape
    assert abs(psnr(hr, bloques['imagen_recon']) - psnr(hr, completo['imagen_recon'])) < 0.1


def test_region_grande_sale_del_tamano_exacto():
    lr = np.random.default_rng(5).random((600, 450))   # salida 4.3 MP > 4 MP
    r = upscale_rzi(lr, 4, ciclos=1, sor_iters=20, bp_final=1)
    assert r['por_bloques']
    assert r["imagen_recon"].shape == (2400, 1800)


def test_salida_demasiado_grande_da_mensaje_claro():
    with pytest.raises(modulo_upscaler.SalidaDemasiadoGrande, match='MP'):
        upscale(np.zeros((3000, 4000, 3)), 2)


def test_cancelacion():
    lr = np.random.default_rng(6).random((40, 40))
    with pytest.raises(CalculoCancelado):
        upscale_rzi(lr, 2, cancelado=lambda: True)


def test_progreso_llega_a_uno():
    avances = []
    upscale_rzi(np.random.default_rng(7).random((30, 30)), 2, ciclos=3,
                progreso=lambda f, t: avances.append(f))
    assert avances and avances[-1] == 1.0 and avances == sorted(avances)


def test_diagnostico_sin_errores():
    from services.diagnostico import ejecutar_diagnostico
    texto, ok = ejecutar_diagnostico()
    assert ok, texto


def test_estimador_de_ruido(camara):
    from core.denoise import estimar_ruido
    assert estimar_ruido(camara) < 0.006
    ruidosa = np.clip(camara + np.random.default_rng(8).normal(0, 0.02, camara.shape), 0, 1)
    assert abs(estimar_ruido(ruidosa) - 0.02) < 0.004


def test_limpieza_automatica_rescata_entrada_ruidosa(camara):
    hr, lr = degradar(camara, 2)
    lr = np.clip(lr + np.random.default_rng(9).normal(0, 0.02, lr.shape), 0, 1)
    bic = psnr(hr, interpolacion_bicubica(lr, 2))
    sin = upscale_rzi(lr, 2, limpieza='no')
    auto = upscale_rzi(lr, 2, limpieza='auto')
    assert auto['limpieza_aplicada'] == 'ruido'
    assert psnr(hr, sin['imagen_recon']) < bic            # sin limpiar copia el ruido
    assert psnr(hr, auto['imagen_recon']) > bic + 0.5      # limpiando supera a la bicubica


def test_limpieza_automatica_no_toca_imagenes_limpias(camara):
    hr, lr = degradar(camara, 2)
    auto = upscale_rzi(lr, 2, ciclos=3)
    sin = upscale_rzi(lr, 2, ciclos=3, limpieza='no')
    assert auto['limpieza_aplicada'] == 'no'
    assert np.array_equal(auto['imagen_recon'], sin['imagen_recon'])


def test_region_pequena_no_se_limpia_sola_pero_si_con_sigma_global():
    textura = np.random.default_rng(10).random((64, 64))     # parece "ruido"
    assert upscale_rzi(textura, 2, ciclos=1)['limpieza_aplicada'] == 'no'
    r = upscale_rzi(textura, 2, ciclos=1, sigma_ruido=0.03)
    assert r['limpieza_aplicada'] == 'ruido'


def test_animacion_con_limpieza_coincide_con_upscale(camara):
    tile = np.clip(camara[100:228, 100:228]
                   + np.random.default_rng(11).normal(0, 0.03, (128, 128)), 0, 1)
    res = upscale(tile, 2, ciclos=2, sor_iters=40)
    gen = generar_frames_pipeline(tile, 2, ciclos=2, sor_iters=40)
    assert gen['limpieza_aplicada'] == 'ruido'
    assert 'Limpieza de la entrada (variacion total)' in gen['pasos']
    assert np.allclose(gen['frames'][-1], res['imagen_recon'], atol=1e-12)
