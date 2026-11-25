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
from core.sor_solver import (resolver, resolver_screened, barrido_omega,
                             benchmark_numba_vs_python)
from core.upscaler import upscale, upscale_rzi, back_projection, error_consistencia
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
