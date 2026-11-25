"""
Evalua RZI frente a bicubica y Lanczos con las imagenes de imagenes_ejemplo/.

Uso (desde la raiz del proyecto):
    python scripts/evaluar_imagenes.py

1. Condiciones limpias: cada imagen (luminancia, recorte central de hasta
   768 px) se reduce x2 y x4 con bicubica y se vuelve a ampliar.
2. Entrada degradada (el caso real de hacer zoom en una foto de internet):
   la imagen pequena ademas se comprime en JPEG (calidad 75 y 50) o se le
   agrega ruido (sigma 0.01 y 0.02). Se compara RZI sin limpieza, con
   limpieza automatica y con limpieza manual (ruido o JPEG).
3. Zoom real sin verdad de campo: recortes ampliados x4 de algunas imagenes
   (vecino, bicubica, RZI) para revisar artefactos a ojo.

Salidas: docs/resultados/imagenes_ejemplo*.csv y docs/figuras/ejemplos_*.png
"""

import csv
import glob
import io
import os
import sys
import time

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from core.image_ops import resize_rgb_float
from core.poisson_builder import imagen_a_grises
from core.upscaler import upscale, upscale_rzi
from core.denoise import estimar_ruido
from services.metrics import (degradar, psnr, ssim, gmsd, interpolacion_bicubica,
                              interpolacion_lanczos)

CARPETA = os.path.join(RAIZ, 'imagenes_ejemplo')
FIG = os.path.join(RAIZ, 'docs', 'figuras')
RES = os.path.join(RAIZ, 'docs', 'resultados')
LADO = 768
AZUL, VERDE, NARANJA, ROJO, GRIS = '#1976D2', '#2E7D32', '#EF6C00', '#C62828', '#757575'


def cargar():
    imagenes = []
    for ruta in sorted(glob.glob(os.path.join(CARPETA, '*'))):
        try:
            rgb = np.asarray(Image.open(ruta).convert('RGB'), dtype=np.float64) / 255.0
        except Exception as exc:  # formato no soportado por Pillow (p. ej. avif)
            print(f'  omitida {os.path.basename(ruta)}: {type(exc).__name__}')
            continue
        imagenes.append((os.path.basename(ruta), rgb))
    return imagenes


def recorte_central(a, lado=LADO):
    H, W = a.shape[:2]
    y0, x0 = max(0, (H - lado) // 2), max(0, (W - lado) // 2)
    return a[y0:y0 + lado, x0:x0 + lado]


def jpeg(a, calidad):
    buf = io.BytesIO()
    Image.fromarray(np.uint8(np.clip(a, 0, 1) * 255 + 0.5)).save(buf, 'JPEG', quality=calidad)
    return np.asarray(Image.open(buf), dtype=np.float64) / 255.0


def guardar_csv(nombre, filas):
    with open(os.path.join(RES, nombre), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
        w.writeheader()
        for f in filas:
            w.writerow({k: (f'{v:.4f}' if isinstance(v, float) else v) for k, v in f.items()})


def limpio(imagenes):
    print('Condiciones limpias...')
    filas = []
    for nombre, rgb in imagenes:
        g = recorte_central(imagen_a_grises(rgb))
        for f in (2, 4):
            hr, lr = degradar(g, f)
            bic, lan = interpolacion_bicubica(lr, f), interpolacion_lanczos(lr, f)
            t0 = time.perf_counter()
            r = upscale_rzi(lr, f)
            dt = time.perf_counter() - t0
            u = r['imagen_recon']
            filas.append({
                'imagen': nombre, 'factor': f,
                'psnr_bic': psnr(hr, bic), 'psnr_lan': psnr(hr, lan), 'psnr_rzi': psnr(hr, u),
                'ganancia_vs_bic': psnr(hr, u) - psnr(hr, bic),
                'ganancia_vs_lan': psnr(hr, u) - psnr(hr, lan),
                'ssim_bic': ssim(hr, bic), 'ssim_rzi': ssim(hr, u),
                'gmsd_bic': gmsd(hr, bic), 'gmsd_rzi': gmsd(hr, u),
                'limpieza_auto': r['limpieza_aplicada'], 'tiempo_s': dt,
            })
        print(f'  {nombre:<40} x2 {filas[-2]["ganancia_vs_bic"]:+.2f} dB  '
              f'x4 {filas[-1]["ganancia_vs_bic"]:+.2f} dB', flush=True)
    guardar_csv('imagenes_ejemplo.csv', filas)

    # figura: ganancia por imagen
    nombres = [f['imagen'] for f in filas if f['factor'] == 2]
    g2 = [f['ganancia_vs_bic'] for f in filas if f['factor'] == 2]
    g4 = [f['ganancia_vs_bic'] for f in filas if f['factor'] == 4]
    orden = np.argsort(g2)[::-1]
    y = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(nombres) + 1.5))
    ax.barh(y - 0.2, np.array(g2)[orden], 0.4, color=AZUL, label='x2')
    ax.barh(y + 0.2, np.array(g4)[orden], 0.4, color=NARANJA, label='x4')
    ax.set_yticks(y)
    ax.set_yticklabels([nombres[i][:30] for i in orden], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Ganancia de PSNR de RZI sobre la bicubica (dB)')
    ax.set_title(f'Imagenes de ejemplo del equipo: RZI gana en {sum(x > 0 for x in g2)}/{len(g2)} (x2) '
                 f'y {sum(x > 0 for x in g4)}/{len(g4)} (x4)')
    ax.legend()
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'ejemplos_ganancia.png'), dpi=150)
    plt.close(fig)
    return filas


def robustez(imagenes):
    print('Entrada con JPEG o ruido...')
    rng = np.random.default_rng(0)
    escenarios = {
        'JPEG calidad 75': lambda lr: jpeg(lr, 75),
        'JPEG calidad 50': lambda lr: jpeg(lr, 50),
        'Ruido 0.01': lambda lr: np.clip(lr + rng.normal(0, 0.01, lr.shape), 0, 1),
        'Ruido 0.02': lambda lr: np.clip(lr + rng.normal(0, 0.02, lr.shape), 0, 1),
    }
    filas = []
    for esc, fn in escenarios.items():
        manual = 'jpeg' if esc.startswith('JPEG') else 'ruido'
        for f in (2, 4):
            vals = {'sin': [], 'auto': [], 'manual': []}
            activas = 0
            for nombre, rgb in imagenes:
                g = recorte_central(imagen_a_grises(rgb), 512)
                hr, lr = degradar(g, f)
                lr = fn(lr)
                pb = psnr(hr, interpolacion_bicubica(lr, f))
                vals['sin'].append(psnr(hr, upscale_rzi(lr, f, limpieza='no')['imagen_recon']) - pb)
                ra = upscale_rzi(lr, f, limpieza='auto')
                activas += ra['limpieza_aplicada'] != 'no'
                vals['auto'].append(psnr(hr, ra['imagen_recon']) - pb)
                vals['manual'].append(psnr(hr, upscale_rzi(lr, f, limpieza=manual)['imagen_recon']) - pb)
            fila = {'escenario': esc, 'factor': f, 'limpieza_manual': manual,
                    'activa_auto': activas, 'n': len(imagenes)}
            for k, v in vals.items():
                fila[f'{k}_media'] = float(np.mean(v))
                fila[f'{k}_peor'] = float(np.min(v))
            filas.append(fila)
            print(f"  {esc:<16} x{f}: sin limpieza {fila['sin_media']:+.2f}  "
                  f"auto {fila['auto_media']:+.2f} ({activas}/{len(imagenes)})  "
                  f"manual ({manual}) {fila['manual_media']:+.2f}", flush=True)
    guardar_csv('imagenes_ejemplo_robustez.csv', filas)

    etiquetas = [f"{f['escenario']}\nx{f['factor']}" for f in filas]
    x = np.arange(len(filas))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    for k, (clave, color, texto) in enumerate((('sin', ROJO, 'RZI sin limpieza'),
                                               ('auto', AZUL, 'RZI limpieza automatica'),
                                               ('manual', VERDE, 'RZI limpieza manual'))):
        ax.bar(x + (k - 1) * 0.27, [f[f'{clave}_media'] for f in filas], 0.27,
               color=color, label=texto)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=8)
    ax.set_ylabel('Ganancia sobre la bicubica (dB)')
    ax.set_title('Entrada comprimida o con ruido: la limpieza previa recupera la ventaja de RZI')
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle=':', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'ejemplos_robustez.png'), dpi=150)
    plt.close(fig)
    return filas


def zoom_real(imagenes, elegidas=('brain x-ray.jpeg', '10.jpeg', 'matthias-schmidt-gas-giant-02.jpg',
                                  '7.jpeg')):
    """Zoom x4 real (sin verdad de campo) sobre la zona con mas bordes."""
    print('Zoom real (comparacion visual)...')
    from services.exporter import elegir_recorte
    dic = dict(imagenes)
    elegidas = [e for e in elegidas if e in dic]
    if not elegidas:
        return
    fig, ejes = plt.subplots(len(elegidas), 3, figsize=(11, 3.7 * len(elegidas)), squeeze=False)
    for fila, nombre in enumerate(elegidas):
        rgb = dic[nombre]
        y0, x0, lado = elegir_recorte(imagen_a_grises(rgb), 64)
        tile = rgb[y0:y0 + lado, x0:x0 + lado]
        # como la aplicacion: el ruido se mide sobre la imagen completa
        res = upscale(tile, 4, sigma_ruido=estimar_ruido(recorte_central(imagen_a_grises(rgb), 1024)))
        W = H = lado * 4
        paneles = (('Zoom nativo (vecino)', resize_rgb_float(tile, W, H, Image.NEAREST)),
                   ('Bicubica', resize_rgb_float(tile, W, H, Image.BICUBIC)),
                   (f"RZI ({res['limpieza_aplicada'].replace('no', 'sin limpieza')})",
                    res['imagen_recon']))
        for col, (titulo, img) in enumerate(paneles):
            ax = ejes[fila, col]
            ax.imshow(np.clip(img, 0, 1), interpolation='nearest')
            ax.set_title(f'{nombre[:26]}: {titulo}' if col == 0 else titulo, fontsize=9)
            ax.axis('off')
    fig.suptitle('Zoom x4 real sobre las imagenes del equipo (64x64 px -> 256x256 px)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'ejemplos_zoom_real.png'), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    t0 = time.time()
    imagenes = cargar()
    print(f'{len(imagenes)} imagenes cargadas')
    filas = limpio(imagenes)
    for f in (2, 4):
        sel = [x for x in filas if x['factor'] == f]
        g = [x['ganancia_vs_bic'] for x in sel]
        gl = [x['ganancia_vs_lan'] for x in sel]
        print(f'  x{f}: media {np.mean(g):+.2f} dB (min {min(g):+.2f}, max {max(g):+.2f}); '
              f'gana a bicubica en {sum(v > 0 for v in g)}/{len(g)}, a Lanczos en '
              f'{sum(v > 0 for v in gl)}/{len(gl)}; SSIM mejor en '
              f"{sum(x['ssim_rzi'] > x['ssim_bic'] for x in sel)}, GMSD mejor en "
              f"{sum(x['gmsd_rzi'] < x['gmsd_bic'] for x in sel)}; limpieza auto activada en "
              f"{sum(x['limpieza_auto'] != 'no' for x in sel)}")
    robustez(imagenes)
    zoom_real(imagenes)
    print(f'Listo en {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()
