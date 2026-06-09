# RZI - Reconstructor de Zoom Iterativo

## Proyecto de Algoritmos Numericos - Universidad de Pamplona

Herramienta de escritorio que reemplaza la interpolacion geometrica tradicional por una reconstruccion matematica basada en Ecuaciones Diferenciales Parciales (EDP). Al hacer zoom sobre una region, el software resuelve un sistema lineal disperso mediante el metodo SOR Red-Black acelerado con Numba para calcular los pixeles intermedios preservando la nitidez de los bordes.

**Aclaracion Conceptual:** Este proyecto no utiliza Redes Neuronales Generativas (IA) que "inventan" informacion inexistente. Los metodos numericos estacionarios implementados interpolan y propagan campos escalares basandose estrictamente en la informacion de los pixeles conocidos.

## Caracteristicas Principales

- **Metodo Unico SOR Red-Black:** Descarte de Jacobi, Gauss-Seidel y SSOR en favor del acelerador espectral optimo para matrices Simetricas Definidas Positivas (SPD).
- **Aceleracion JIT con Numba:** Compilacion del nucleo matematico a codigo maquina LLVM nativo, logrando aceleraciones de hasta 400x frente a bucles de Python puro.
- **Enfoque Matrix-Free:** El operador Laplaciano (estencil de 5 puntos) se aplica directamente sobre la malla 2D. La matriz del sistema (A) nunca se construye en memoria, mitigando el riesgo de Out of Memory (OOM) en imagenes grandes.
- **Modelo de Poisson y Laplace:** Reconstruccion guiada por la divergencia del campo de gradientes (Poisson) para preservar bordes, o interpolacion armonica pura (Laplace) para upscaling sin efecto de ringing.
- **Interfaz Moderna (PySide6):** Tema oscuro, navegacion fluida con Zoom y Pan mediante QGraphicsView.
- **Validacion Academica (Enfoque B):** Comparacion contra verdad de campo (Ground Truth) submuestreada utilizando metricas objetivas (PSNR y SSIM).
- **Pruebas de Rendimiento:** Graficas de convergencia espectral (Barrido de Omega) y benchmarking Numba vs Python puro.
- **Exportacion:** Tablas comparativas en CSV, imagenes reconstruidas y graficas en alta resolucion.

## Modelo Matematico

El proyecto resuelve la Ecuacion de Poisson discreta para cada pixel desconocido (i, j):

`4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] = b[i,j]`

Donde:
- El lado izquierdo es el estencil de 5 puntos del operador Laplaciano.
- `b[i,j]` es la divergencia discreta del campo de gradientes de la imagen guia (para Poisson) o cero (para Laplace/interpolacion armonica).
- Los pixeles conocidos actuan como condiciones de frontera de Dirichlet.

**Parametro de Relajacion Optimo (omega_opt):**
Para una malla de n x n, el minimo teorico de iteraciones (acelerador espectral) ocurre en:
`omega_opt = 2 / (1 + sin(pi / (n + 1)))`

## Arquitectura de Software (3 Capas)

El codigo sigue una arquitectura estricta de tres capas para maximizar el desacoplamiento y la mantenibilidad:

1. **Capa UI (`ui/`):** Interfaz grafica en PySide6, visores con zoom/pan, paneles de parametros y gestion de hilos (QThread) para no bloquear el event loop de Qt.
2. **Capa Core (`core/`):** Nucleo matematico puro. Constructor del problema de Poisson (`poisson_builder.py`) y solver SOR Red-Black compilado con `@njit` (`sor_solver.py`). No tiene conocimiento de la interfaz.
3. **Capa Services (`services/`):** Servicios de apoyo. Calculo de metricas PSNR/SSIM, interpolacion bicubica de referencia, y generacion/exportacion de graficas con Matplotlib.

## Instalacion

### Requisitos
- Python 3.10 o superior
- pip (gestor de paquetes de Python)

### Pasos

1. Crear y activar entorno virtual (recomendado):
   - Windows: `python -m venv venv && venv\Scripts\activate`
   - Linux/Mac: `python3 -m venv venv && source venv/bin/activate`

2. Instalar dependencias:
   `pip install -r requirements.txt`

3. Ejecutar la aplicacion:
   `python main.py`

*Nota tecnica:* La primera ejecucion tardara entre 5 y 15 segundos adicionales debido a que Numba compila el kernel SOR y lo almacena en cache. Las ejecuciones posteriores seran instantaneas.

## Guia de Uso

La aplicacion opera en dos modos principales:

**1. Modo Reconstruir (Upscaling Armonico):**
- Cargue una imagen y haga zoom con la rueda del mouse sobre la region de interes en el visor izquierdo.
- Pulse "Reconstruir (Poisson SOR)".
- El visor derecho mostrara la region visible ampliada y suavizada matematicamente, incrustada en su contexto original. Ideal para eliminar el efecto de "bloques pixelados" del zoom nativo.

**2. Modo Comparar 3 Modelos (Validacion con Verdad de Campo):**
- Pulse "Comparar 3 modelos".
- El sistema degrada la region visible (submuestreo) y la reconstruye usando tres metodos: Interpolacion Bicubica (Pillow), Laplace Puro (b=0) y Poisson SOR.
- El panel derecho mostrara la comparacion visual lado a lado y una tabla con las metricas PSNR (dB) y SSIM contra la imagen original de alta resolucion.

**3. Pruebas Tecnicas:**
- **Barrido Omega:** Genera la curva de convergencia espectral en forma de V, demostrando empiricamente el omega_optimo teorico.
- **Benchmark Numba vs Python:** Compara los tiempos de ejecucion del kernel compilado frente a bucles nativos de Python.

## Dependencias

- **PySide6:** Interfaz grafica y motores de renderizado Qt.
- **Numba:** Compilador JIT para aceleracion de bucles numericos.
- **NumPy:** Algebra lineal y manipulacion de arrays.
- **SciPy:** Transformadas de distancia (para relleno de pixeles) y filtros gaussianos.
- **Pillow:** Interpolacion bicubica de referencia y manipulacion de imagenes.
- **Matplotlib:** Generacion de graficas de convergencia y comparacion.
- **scikit-image:** Calculo certificado de metricas PSNR y SSIM.

## Troubleshooting

- **Error de modulo faltante:** Ejecutar `pip install -r requirements.txt`.
- **La interfaz se congela al calcular:** Los calculos pesados se ejecutan en un QThread separado. El sistema incluye un limite de seguridad de 1.2 millones de pixeles para el upscaling con el fin de evitar bloqueos por saturacion de RAM.
- **Numba no acelera:** Verificar que la libreria este correctamente instalada. El panel izquierdo de la interfaz indica si Numba esta activo o no. Si no lo esta, el solver funcionara pero sera significativamente mas lento.
- **La reconstruccion se ve igual que la original:** El modo "Reconstruir" no inventa informacion. Su objetivo es eliminar la interpolacion de vecino mas cercano (bloques) del zoom digital. Para ver diferencias notorias, haga zoom hasta que los pixeles originales sean visibles como cuadrados en el visor izquierdo.

## Creditos

**Materia:** Algoritmos Numericos para Ingenieria (Corte 3)
**Universidad:** Universidad de Pamplona

**Equipo de desarrollo:**
- Anderson Fabian Gonzalez Aparicio
- Wilmer Sneyder Florez Peña
- Gonzalo Javier Niño Amaris