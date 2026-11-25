# RZI - Reconstructor de Zoom Iterativo

## Proyecto de Algoritmos Numericos - Universidad de Pamplona

Aplicacion de escritorio que amplia imagenes resolviendo una ecuacion diferencial parcial (Poisson) sobre la malla de pixeles, en lugar de usar solo interpolacion geometrica. El sistema lineal resultante se resuelve con **SOR Red-Black** sin construir la matriz (*matrix-free*) y acelerado con **Numba**.

No usa inteligencia artificial: no hay entrenamiento ni datos externos. Todo sale de la ecuacion y de los pixeles de la propia imagen.

**Documento de estudio completo (para la sustentacion):** [`docs/RZI_Documento_de_Estudio.pdf`](docs/RZI_Documento_de_Estudio.pdf)

## Resultados

Protocolo de verdad de campo: se reduce una imagen real por el factor indicado (bicubica con antialias) y se vuelve a ampliar con cada metodo. Promedio sobre 4 imagenes estandar de scikit-image (camera, astronaut, coffee, chelsea):

| Factor | Bicubica | Lanczos | **RZI** | Ganancia sobre bicubica |
|:---:|:---:|:---:|:---:|:---:|
| x2 | 30.90 dB | 31.30 dB | **32.17 dB** | +1.27 dB |
| x3 | 28.46 dB | 28.73 dB | **29.44 dB** | +0.98 dB |
| x4 | 27.00 dB | 27.25 dB | **28.01 dB** | +1.01 dB |

RZI tambien mejora SSIM y GMSD en las 12 combinaciones imagen/factor. Si la imagen se redujo con otro nucleo (Lanczos, desenfoque gaussiano, promedio de area), RZI sigue ganando, aunque menos (+0.4 a +0.8 dB). Tablas completas en `docs/resultados/` y figuras en `docs/figuras/`.

![Comparacion x4](docs/figuras/comparacion_camera_x4.png)

## El modelo

Se alternan dos pasos (10 ciclos por defecto):

1. **Back-projection** (consistencia con los datos): `t = u + Up(f - Down(u))`.
2. **Poisson "screened"** con gradientes reforzados en los bordes, resuelta con SOR Red-Black:

   `(4 + lam) u[i,j] - (suma de 4 vecinos) = -div(v)[i,j] + lam * t[i,j]`

   donde `v = s(|grad t|) grad t` y `s(x) = 1 + (beta - 1) x / (x + tau)`.

La malla esta centrada en pixeles (la salida mide exactamente `h*k x w*k`). A color, la luminancia (Y) pasa por el modelo y la crominancia (Cb, Cr) por bicubica con back-projection.

**Omega optimo.** Se usa la formula de Young, `omega = 2 / (1 + sqrt(1 - rho_J^2))`. En el problema clasico con pixeles fijos dentro de la malla, `rho_J` se estima con el metodo de la potencia: la formula de malla completa (`2 / (1 + sin(pi/(n+1)))`) no aplica ahi y da casi 2, cuando el optimo real con factor 4 es cerca de 1.51 (20 iteraciones contra unas 350).

## Arquitectura (3 capas)

| Capa | Carpeta | Contenido |
|:---|:---|:---|
| Interfaz | `ui/` | Ventana PySide6, visores con zoom sincronizado, calculos en `QThread`. |
| Nucleo | `core/` | `poisson_builder.py` (problemas y omega), `sor_solver.py` (kernels SOR Red-Black con Numba), `upscaler.py` (modelo RZI), `image_ops.py` (reescalado y color en float64). |
| Servicios | `services/` | `metrics.py` (PSNR, SSIM, GMSD, nitidez, protocolo de degradacion), `exporter.py` (CSV y graficas). |

Ademas: `tests/` (pruebas automaticas), `scripts/generar_resultados.py` (reproduce tablas y figuras) y `docs/estudio/` (fuente del documento PDF).

## Instalacion y uso

Requisitos: Python 3.10 o superior.

```bash
python -m venv venv
```

```bash
source venv/bin/activate        # Windows: venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
python main.py
```

Tambien estan `instalar.sh` / `instalar.bat` y `ejecutar.sh` / `ejecutar.bat`. La primera ejecucion tarda unos segundos mas porque Numba compila los kernels y los guarda en cache.

### Modos de la aplicacion

- **Reconstruir (RZI):** haga zoom sobre una region del visor izquierdo y pulse el boton. Se amplia la region visible y se abre un dialogo con zoom nativo, bicubica y RZI lado a lado (zoom sincronizado), con nitidez y consistencia de cada uno.
- **Comparar modelos:** toma la region visible como verdad de campo, la reduce y la vuelve a ampliar con bicubica, Lanczos, RZI y RZI con realce. Muestra PSNR, SSIM, GMSD y nitidez; se puede exportar a CSV.
- **Barrido Omega:** curva de iteraciones contra omega, con el minimo medido, el omega estimado con la mascara y el de la formula de malla completa.
- **Benchmark:** Numba contra Python puro haciendo exactamente el mismo trabajo.
- **Modo Lupa** y **Animar convergencia:** herramientas de demostracion.

El control **Realce perceptual** (0 a 1, apagado por defecto) marca mas los bordes a la vista a cambio de bajar un poco el PSNR.

### Pruebas y resultados

```bash
python -m pytest tests -q
```

```bash
python scripts/generar_resultados.py
```

```bash
bash docs/estudio/construir_pdf.sh
```

El ultimo comando regenera el PDF. Requiere pandoc y Node.js con Playwright/Chromium.

## Limitaciones

- No recupera detalle fino que desaparecio al reducir la imagen: mejora sobre todo los bordes.
- Supone un modelo de degradacion bicubico. Con otros la ganancia es menor.
- Las ganancias (0.5 a 1.7 dB sobre la bicubica) son reales pero moderadas: los metodos con redes neuronales logran mas, a costa de entrenamiento y de poder inventar detalle.

## Creditos

**Materia:** Algoritmos Numericos para Ingenieria (Corte 3)
**Universidad:** Universidad de Pamplona

**Equipo de desarrollo:**
- Anderson Fabian Gonzalez Aparicio
- Wilmer Sneyder Florez Peña
- Gonzalo Javier Niño Amaris
