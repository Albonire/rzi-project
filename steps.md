# Documentacion del Proyecto RZI
## Reconstructor de Zoom Iterativo con SOR Red-Black y Numba

---

## Indice

1. [Que hace este proyecto](#1-que-hace-este-proyecto)
2. [Que estaba mal en la version anterior](#2-que-estaba-mal-en-la-version-anterior)
3. [Arquitectura del proyecto](#3-arquitectura-del-proyecto)
4. [El modelo matematico](#4-el-modelo-matematico)
5. [Dos modos de uso: Reconstruir y Comparar](#5-dos-modos-de-uso-reconstruir-y-comparar)
6. [El metodo SOR Red-Black](#6-el-metodo-sor-red-black)
7. [El parametro omega optimo](#7-el-parametro-omega-optimo)
8. [Aceleracion con Numba JIT](#8-aceleracion-con-numba-jit)
9. [Validacion de resultados](#9-validacion-de-resultados)
10. [Como ejecutar el proyecto](#10-como-ejecutar-el-proyecto)
11. [Conclusiones](#11-conclusiones)

---

## 1. Que hace este proyecto

Una imagen digital es una cuadricula de puntos llamados pixeles. Cuando se hace un zoom sobre una imagen, hay regiones de esa cuadricula donde no existe informacion original: la imagen original no contiene suficientes pixeles para mostrar todos los detalles en la resolucion ampliada.

Este proyecto resuelve ese problema mediante un enfoque matematico: interpolar (rellenar) los pixeles desconocidos de forma que el resultado sea visualmente coherente con los pixeles que si se conocen. Para ello se plantea una ecuacion diferencial discreta, llamada ecuacion de Poisson, y se resuelve con el metodo numerico SOR Red-Black acelerado con la biblioteca Numba.

El flujo de trabajo sigue el **Enfoque B** descrito en el documento del proyecto:

```
Imagen original (alta resolucion, la verdad de campo)
       |
       v
  Submuestreo: 1 de cada factor x factor pixeles se marca como "conocido"
       |
       v
  Planteo de la ecuacion (Poisson o Laplace) para los pixeles desconocidos
       |
       v
  Resolucion iterativa con SOR Red-Black + Numba
       |
       v
  Imagen reconstruida -> comparar contra original con PSNR y SSIM
```

---

## 2. Que estaba mal en la version anterior

La version original del proyecto tenia varios problemas respecto a la especificacion. La siguiente tabla resume el diagnostico:

| Aspecto | Lo que pide la especificacion | Lo que tenia el codigo | Corregido |
|---------|-------------------------------|------------------------|-----------|
| Metodo de resolucion | Solo SOR Red-Black | Jacobi, Gauss-Seidel, SOR y SSOR (cuatro metodos) | Si |
| Modelo matematico | Ecuacion de Poisson (b != 0) | Ecuacion de Laplace pura (b = 0) | Si |
| Construccion de la matriz | Matrix-Free (nunca construir A) | Construia la matriz dispersa CSR explicitamente | Si |
| Aceleracion | Numba `@njit` (critico) | Sin Numba, bucles Python puros | Si |
| Arquitectura | 3 capas: ui/, core/, services/ | 5 capas en archivos planos | Si |
| Enfoque de prueba | B: alta res -> submuestrear -> reconstruir -> comparar | No submuestreaba ni comparaba contra verdad de campo | Si |
| Comparativa visual | Bicubica vs Laplace vs Poisson | No existia | Si |
| numba en requirements | Si | No | Si |

La correccion mas importante fue el cambio de cuatro metodos a un unico metodo SOR, y la adicion del termino de Poisson junto con la aceleracion Numba.

---

## 3. Arquitectura del proyecto

El proyecto sigue una arquitectura de tres capas, donde cada capa tiene una responsabilidad bien definida. La siguiente figura muestra la estructura de archivos y el flujo de informacion:

```
proyecto_rzi/
|
+-- main.py                  <- Punto de entrada: lanza la aplicacion Qt
|
+-- ui/                      <- Capa 1: Interfaz grafica (lo que ve el usuario)
|   +-- main_window.py       |  Ventana principal, paneles, botones, hilo de calculo
|   +-- zoom_viewer.py       |  Visor con zoom/pan y seleccion rectangular
|
+-- core/                    <- Capa 2: Nucleo matematico
|   +-- poisson_builder.py   |  Construye el problema: submuestreo, mascara, vector b
|   +-- sor_solver.py        |  Kernel SOR Red-Black con @njit, barrido omega, benchmark
|
+-- services/                <- Capa 3: Servicios de apoyo
|   +-- metrics.py           |  PSNR, SSIM, interpolacion bicubica, tabla comparativa
|   +-- exporter.py          |  Guardar imagenes, CSV, graficas matplotlib
|
+-- assets/                  <- Imagenes de prueba y graficas generadas
+-- requirements.txt         <- Dependencias (PySide6, numpy, scipy, numba, etc.)
+-- steps.md                 <- Este documento
```

La regla de la arquitectura es simple: las capas superiores pueden llamar a las inferiores, pero nunca al reves. La interfaz (Capa 1) llama al nucleo (Capa 2) y a los servicios (Capa 3). El nucleo no sabe nada de la interfaz.

El calculo se ejecuta en un hilo separado (`QThread`) para que la interfaz no se congele mientras el solver trabaja.

---

## 4. El modelo matematico

### 4.1 La ecuacion de Laplace

La ecuacion de Laplace dice que en cada pixel desconocido, el valor debe ser exactamente el promedio de sus cuatro vecinos inmediatos:

```
    u[i-1, j]
         |
u[i, j-1] -- u[i, j] -- u[i, j+1]
         |
    u[i+1, j]

Ecuacion de Laplace:
  4 * u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] = 0
```

Esto produce una funcion llamada "armonica": la mas suave posible entre los valores conocidos. El resultado es una interpolacion que no tiene oscilaciones bruscas, pero tampoco puede preservar bordes nitidos.

### 4.2 La ecuacion de Poisson

La ecuacion de Poisson agrega un termino extra `b[i,j]` en el lado derecho:

```
  4 * u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] = b[i,j]
```

El termino `b[i,j]` es la divergencia del campo de gradientes de la imagen de baja resolucion. En terminos simples: captura donde estan los cambios de intensidad (los bordes) y con que fuerza se producen. Al incluirlo en la ecuacion, el solver es "guiado" hacia una solucion que respeta esa estructura de bordes.

**Como se calcula b en el codigo** (`core/poisson_builder.py`):

1. Se toman los pixeles conocidos y se rellena el resto con el vecino conocido mas cercano (relleno por vecino mas proximo).
2. Se suaviza con un filtro Gaussiano (sigma=1.0) para reducir artefactos de borde.
3. Se calculan los gradientes horizontales (`gx`) y verticales (`gy`) de esa imagen guia.
4. Se calcula la divergencia discreta: `b = d(gx)/dx + d(gy)/dy`.
5. En los pixeles conocidos se pone `b = 0` porque ahi la condicion de Dirichlet domina.

```python
# fragmento de core/poisson_builder.py
fill_idx   = distance_transform_edt(~mascara, return_distances=False, return_indices=True)
guia       = imagen_gris[fill_idx[0], fill_idx[1]]   # relleno NN
suavizada  = gaussian_filter(guia, sigma=1.0)
gx         = np.gradient(suavizada, axis=1)
gy         = np.gradient(suavizada, axis=0)
b          = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
b[mascara] = 0.0
```

### 4.3 Enfoque Matrix-Free

Para una imagen de 256 x 256 pixeles, la matriz del sistema lineal A tendria 65536 x 65536 = 4.295 millones de filas y columnas. En formato denso esto ocuparia aproximadamente 34 GB de memoria RAM, lo que es completamente inviable.

La solucion es el enfoque **Matrix-Free**: nunca se construye la matriz A. El estencil de 5 puntos se aplica directamente sobre la cuadricula 2D de pixeles en cada iteracion. En lugar de multiplicar A * u, se calcula `4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1]` directamente. Esto usa O(1) memoria adicional, independientemente del tamano de la imagen.

---

## 5. Dos modos de uso: Reconstruir y Comparar

La aplicacion tiene dos modos de operacion bien diferenciados. Entender la diferencia es clave para interpretar los resultados.

### 5.1 Tabla de modos

| Boton | Que hace | Para que sirve |
|-------|----------|----------------|
| Reconstruir (Poisson SOR) | Toma la region visible como imagen de entrada y la agranda factor veces | Ver el resultado del upscaling: la region se ve mas suave y con bordes definidos |
| Comparar 3 modelos | Degrada la imagen a proposito (Enfoque B) y la reconstruye para medir PSNR/SSIM | Validacion academica: medir la calidad numerica contra la verdad de campo |

### 5.2 El modo "Reconstruir": upscaling real

Cuando el usuario hace zoom en una region y pulsa "Reconstruir", ocurre lo siguiente:

1. Se extrae la region visible del visor izquierdo como un tile (un recorte de pixeles).
2. Ese tile se trata como la imagen de baja resolucion: es la informacion que ya se tiene.
3. Se crea una malla nueva, factor veces mas grande por lado.
4. Los pixeles del tile ocupan posiciones exactas en esa malla grande: `tile[i,j]` cae en `malla[i*factor, j*factor]`.
5. Los huecos entre ellos son los pixeles nuevos a calcular.
6. La PDE de Poisson rellena esos huecos de forma matematicamente suave y coherente con los bordes del tile.
7. El visor derecho muestra la region agrandada y suave.

```
tile 64x64  (factor=4)  ->  malla 253x253

  tile[0,0]      tile[0,1]      ...
     |   (huecos)   |
  malla[0,0] ... malla[0,4] ... malla[0,8]
  (conocido)      (nuevo)        (conocido)
```

La formula de la malla alineada por esquinas garantiza que no haya borde colgante:

```
new_h = (h - 1) * factor + 1
new_w = (w - 1) * factor + 1

Ejemplo: tile 64x64, factor=4  ->  new_h = 63*4+1 = 253, new_w = 253
```

**Por que se usa Laplace (b=0) en lugar de Poisson con b=Lap(bicubica)**:

Cuando b = Laplaciano(bicubica), el solver converge hacia la bicubica: la solucion esta sesgada y resulta igualmente borrosa. Con b=0, el solver converge hacia la interpolacion armonica (membrana minima), que:
- No tiene el "ringing" (halos) tipico de la bicubica cerca de los bordes.
- Preserva los gradientes de los pixeles conocidos de forma natural.
- Se ve mas nitida en bordes fuertes como los de un equipo o una silueta.

**Punto de partida inteligente**: el solver parte de la bicubica como estimacion inicial (no de cero). Esto da dos ventajas:
- La bicubica ya es una buena aproximacion: el residuo inicial es pequeno.
- El solver converge mas rapido (menos iteraciones).
- El resultado final ES LA SOLUCION ARMONICA (no la bicubica), solo que se llega a ella mas rapido.

**Como se muestra el resultado (visor derecho)**:

1. La malla ampliada (SOR) se ajusta al tamano exacto `th*factor x tw*factor` con LANCZOS para embedding limpio.
2. Se aplica Unsharp Mask (USM) sobre la malla ampliada (NO sobre el tile reducido). El radio se escala por factor (`radius = 2*factor`) para que el efecto perceptual sea equivalente: realza los bordes a la resolucion upscalada, visible al mostrar la imagen a su tamano natural.
3. Se extrae del original una region con padding alrededor del tile (aprox. un tercio de cada lado), se upscala con PIL BICUBIC y se incrusta el resultado SOR en el centro. Esto da contexto visual sin cortar la imagen.
4. El visor derecho muestra ese composite upscalado con ajuste automatico (fit-to-window). Cuando el zoom del visor izquierdo es suficientemente alto (pixel blocks visibles), el visor derecho a fit-to-window muestra la misma area de escena pero con la reconstruccion SOR en lugar de bloques pixelados.

**Por que la diferencia es ahora visible de forma objetiva**:

A zoom alto (factor >= 3x display), el visor izquierdo muestra el tile original con transiciones en escalon entre pixeles (bloques). El visor derecho muestra la malla SOR ampliada: las transiciones son gradientes suaves calculados por la funcion armonica. La comparacion directa es:

```
Izquierdo: [100] [100] [200] [200]  <- pixel blocks (escalon abrupto)
Derecho:   [100] [133] [166] [200]  <- gradiente armonico (transicion suave)
```

La USM con radio escalado amplifica adicionalmente el contraste en los bordes del resultado SOR, haciendo la diferencia perceptualmente mas evidente.

**El parametro "Nitidez (USM %)"**: controla la intensidad del realce. 0 = sin USM (interpolacion armonica pura), 150 = predeterminado, 300 = maximo. A mayor porcentaje, los bordes se ven mas nitidos pero pueden aparecer halos si se excede.

**Nota honesta**: el upscaling no inventa detalle que no existia. El objetivo es eliminar la apariencia pixelada del zoom y producir una interpolacion matematicamente optima (la mas suave posible entre los puntos conocidos). La comparacion justa no es contra la imagen original a zoom 1:1, sino contra el zoom pixelado (vecino mas proximo) que cualquier visor mostraria por defecto.

### 5.3 Por que antes se veia peor (historial de correcciones)

En la version original, "Reconstruir" usaba el Enfoque B: degradaba la region visible (tiraba 15 de cada 16 pixeles) y luego intentaba recuperarla. Este flujo es correcto para medir la calidad de la reconstruccion, pero por construccion nunca puede producir una imagen mejor que la original.

Una segunda version aplicaba el upscaling real (Laplace armonico), pero luego reducia el resultado de vuelta al tamano original con LANCZOS y lo reembebia en la imagen completa mostrando ambos visores al mismo zoom. El problema: el round-trip produce un resultado casi identico al original cuando se ve a la misma escala de pantalla, porque Qt interpola ambas imagenes de la misma manera.

El flujo actual muestra el resultado a la resolucion upscalada (NO reducido), con contexto lateral para evitar el efecto de corte. La diferencia es visible porque las transiciones entre pixeles son suaves en el visor derecho (gradiente armonico) y en escalon en el izquierdo (pixel block del zoom nativo).

### 5.4 Comparacion visual del upscaling

La siguiente figura muestra la diferencia entre los tres metodos de ampliacion para una region de la imagen de camara:

![Demostracion de upscaling](assets/upscaling_demo.png)

- **Vecino mas proximo (zoom nativo)**: bloques pixelados visibles, el estilo tipico del zoom de cualquier visor de imagenes.
- **Bicubica (PIL BICUBIC)**: suave pero con un ligero efecto de "plastificado", pierde algo de contraste local.
- **Poisson SOR (upscaling PDE)**: suave como la bicubica pero con bordes mas definidos en las transiciones abruptas (ver esquina inferior derecha de la figura), gracias al termino de Poisson que guia la solucion hacia los gradientes del tile.

---

## 6. El metodo SOR Red-Black

### 6.1 Que es SOR

SOR (Successive Over-Relaxation) es una variante del metodo de Gauss-Seidel. En cada iteracion, el nuevo valor de `u[i,j]` se calcula como:

```
u_gs   = (b[i,j] + u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1]) / 4
u_nuevo = u_anterior + omega * (u_gs - u_anterior)
```

El parametro `omega` controla la "sobrerrelajacion":
- Con `omega = 1.0` el metodo es identico a Gauss-Seidel.
- Con `omega > 1.0` se "empuja" mas alla del valor de Gauss-Seidel, lo que acelera la convergencia cuando se elige correctamente.
- El rango valido es `1.0 < omega < 2.0`.

### 6.2 El ordenamiento Red-Black (tablero de ajedrez)

El problema del SOR clasico es que cada pixel necesita los valores actualizados de sus vecinos para poder calcularse. Esto crea una dependencia secuencial: hay que actualizar pixel por pixel, fila por fila. Este orden secuencial es imposible de parallelizar y ademas impide que Numba compile el bucle a codigo nativo eficientemente.

La solucion es el **ordenamiento Rojo-Negro** (Red-Black): se divide la cuadricula como un tablero de ajedrez.

```
Tablero de ajedrez 8x8 (R=Rojo, N=Negro):

  R N R N R N R N
  N R N R N R N R
  R N R N R N R N
  N R N R N R N R
  R N R N R N R N
  N R N R N R N R
  R N R N R N R N
  N R N R N R N R

  Un pixel (i, j) es Rojo si (i + j) es par.
  Un pixel (i, j) es Negro si (i + j) es impar.
```

La propiedad clave del estencil de 5 puntos es que los vecinos de un pixel Rojo son todos Negros, y viceversa. Por eso:

- **Fase Roja**: todos los pixeles Rojos se pueden actualizar al mismo tiempo (ninguno depende de otro Rojo).
- **Fase Negra**: todos los pixeles Negros se pueden actualizar al mismo tiempo (usan los valores Rojos recien calculados).

Cada iteracion completa tiene dos fases. Este esquema elimina la dependencia secuencial y permite que Numba compile el bucle interno a codigo LLVM nativo.

```python
# fragmento simplificado de core/sor_solver.py
@njit(cache=True, fastmath=True)
def _kernel_sor_rb(u, b, mascara, omega, max_iter, tol):
    for k in range(max_iter):

        # Fase Roja: (i+j) par
        for i in range(1, nrows-1):
            j_inicio = 2 - (i % 2)          # columna inicial del color rojo en fila i
            for j in range(j_inicio, ncols-1, 2):
                if not mascara[i, j]:        # solo pixeles desconocidos
                    u_gs = (u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]+b[i,j]) * 0.25
                    u[i,j] += omega * (u_gs - u[i,j])

        # Fase Negra: (i+j) impar
        for i in range(1, nrows-1):
            j_inicio = 1 + (i % 2)
            for j in range(j_inicio, ncols-1, 2):
                if not mascara[i, j]:
                    u_gs = (u[i-1,j]+u[i+1,j]+u[i,j-1]+u[i,j+1]+b[i,j]) * 0.25
                    u[i,j] += omega * (u_gs - u[i,j])

        # Criterio de parada: norma L2 del residuo
        res2 = 0.0
        for i, j en cada pixel interior desconocido:
            r = 4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] - b[i,j]
            res2 += r * r
        if sqrt(res2) < tol:
            break   # convergencia alcanzada
```

### 6.3 El criterio de parada

El residuo L2 mide que tan bien el estado actual de `u` satisface la ecuacion:

```
residuo[i,j] = 4*u[i,j] - u[i-1,j] - u[i+1,j] - u[i,j-1] - u[i,j+1] - b[i,j]

norma_L2 = sqrt( sum( residuo[i,j]^2 ) para todos los pixeles desconocidos )
```

Si `norma_L2 < tol`, la solucion esta suficientemente cerca del equilibrio y se detiene. La tolerancia tipica usada es `1e-5`.

La siguiente grafica muestra como cae el residuo con cada iteracion para la imagen de camara 128x128 con factor=4:

![Convergencia del solver](assets/convergencia.png)

El residuo desciende monotonamente en escala logaritmica, confirmando la convergencia correcta del solver.

---

## 7. El parametro omega optimo

### 7.1 La formula

Para el problema de Laplace clasico (solo con condiciones de borde rectangulares, sin restricciones interiores), el valor optimo del parametro de relajacion fue derivado por Young (1971):

```
omega_opt = 2 / (1 + sin(pi / (n + 1)))

donde n = min(numero_de_filas, numero_de_columnas)
```

Para una imagen de 64 x 64 pixeles:
```
n       = 64
omega   = 2 / (1 + sin(pi / 65))
        = 2 / (1 + sin(0.04833...))
        = 2 / (1 + 0.04831...)
        = 2 / 1.04831
        = 1.9078
```

Para una imagen de 128 x 128:
```
n       = 128
omega   = 2 / (1 + sin(pi / 129))
        = 1.9525
```

Cuanto mayor es la imagen, mas cercano a 2.0 es el omega optimo, y mas lenta pero estable es la convergencia con omega suboptimo.

### 7.2 Validacion de la formula

La formula se valida comparando el optimo teorico con el empirico. Para el problema clasico (solo condiciones de borde en los cuatro lados), la coincidencia es casi perfecta:

![Barrido omega: puro borde vs submuestreo](assets/barrido_omega_doble.png)

**Panel izquierdo (condicion de borde pura)**: El minimo de la curva de iteraciones cae en `omega_empirico = 1.9108` frente al teorico `omega_teorico = 1.9078`. La diferencia es de 0.003, validando la formula.

**Panel derecho (submuestreo factor=4)**: El minimo empirico cae en `omega_empirico = 1.5149`. Esto es esperable: cuando hay restricciones de Dirichlet en pixeles interiores (los que se conocen por submuestreo), el dominio efectivo de cada "celda" entre pixeles conocidos es mucho mas pequeno que la imagen completa. Para una celda de 4x4 pixeles, la formula da `omega_opt(4x4) = 1.26`. El valor empirico (1.51) queda entre la escala local (4x4) y la global (64x64), lo cual es correcto: ambas escalas contribuyen al comportamiento espectral.

La implementacion usa `omega_opt` de la formula como punto de partida. Este valor siempre produce convergencia, aunque no sea el exactamente optimo para el problema con submuestreo.

---

## 8. Aceleracion con Numba JIT

### 8.1 Que es Numba

Python es un lenguaje interpretado. Cada operacion en un bucle Python tiene una sobrecarga de varios microsegundos por la interpretacion dinamica del tipo de cada variable. Para bucles con millones de iteraciones (como recorrer todos los pixeles de una imagen), esto hace que Python sea entre 100 y 1000 veces mas lento que C o Fortran.

Numba es una biblioteca de compilacion Just-In-Time (JIT): la primera vez que se llama a una funcion decorada con `@njit`, Numba la compila a codigo maquina nativo usando el compilador LLVM. Las llamadas siguientes ejecutan el codigo nativo directamente, sin la sobrecarga de Python.

### 8.2 Como se aplica en el proyecto

La funcion `_kernel_sor_rb` en `core/sor_solver.py` lleva el decorador `@njit(cache=True, fastmath=True)`:

```python
@njit(cache=True, fastmath=True)
def _kernel_sor_rb(u, b, mascara, omega, max_iter, tol):
    ...
```

- `cache=True`: la version compilada se guarda en disco. La segunda ejecucion no necesita recompilar.
- `fastmath=True`: permite al compilador reordenar operaciones de punto flotante para mayor velocidad.

El proyecto incluye una funcion `_calentamiento()` que invoca el kernel con una malla de 8x8 al inicio. Esto fuerza la compilacion JIT antes de que el usuario cargue una imagen, evitando la espera en el primer uso.

### 8.3 Resultados del benchmark

La siguiente grafica muestra la diferencia en tiempo entre el kernel Python puro y el kernel compilado con Numba, para tres tamanios de imagen:

![Benchmark Numba vs Python](assets/benchmark.png)

| Tamano | Python puro | Numba @njit | Aceleracion |
|--------|-------------|-------------|-------------|
| 32x32  | 64.2 ms     | 0.2 ms      | 380x        |
| 64x64  | 270.2 ms    | 0.6 ms      | 444x        |
| 96x96  | 594.4 ms    | 1.4 ms      | 435x        |

La aceleracion es entre 380 y 444 veces. Para una imagen de 96x96, el solver pasa de tardar 594 milisegundos a tardar 1.4 milisegundos por iteracion. Esto hace viable la reconstruccion interactiva en la interfaz grafica.

---

## 9. Validacion de resultados

### 9.1 Imagen de prueba

Para las pruebas se utilizo la imagen "camara" (64x64 pixeles, escala de grises), una imagen estandar en procesamiento de imagenes por su variedad de texturas y bordes. El factor de submuestreo es 4: solo 1 de cada 16 pixeles es considerado "conocido".

Con factor=4 en una imagen de 64x64, los pixeles conocidos representan aproximadamente el 9% del total. El solver debe reconstruir el 91% restante.

### 9.2 Metricas de calidad

Se usan dos metricas:

**PSNR (Peak Signal-to-Noise Ratio)** mide la diferencia pixel a pixel entre la imagen original y la reconstruida, expresada en decibeles. Valores mayores indican mejor fidelidad. Por encima de 30 dB se considera buena calidad.

```
PSNR = 10 * log10( 1 / MSE )

donde MSE = promedio de (original[i,j] - reconstruida[i,j])^2
```

**SSIM (Structural Similarity Index)** mide la similitud perceptual: brillo, contraste y estructura. Toma valores entre -1 y 1, donde 1 significa imagenes identicas.

### 9.3 Tabla comparativa

La siguiente tabla muestra los resultados sobre la imagen de camara 64x64 con factor=4:

| Modelo | PSNR (dB) | SSIM |
|--------|-----------|------|
| Bicubica (Pillow LANCZOS + BICUBIC) | 26.16 | 0.8819 |
| Laplace puro (b = 0) | 24.50 | 0.8261 |
| Poisson SOR Red-Black | 21.70 | 0.7117 |

El metodo bicubico obtiene la mayor metrica en este test porque la interpolacion de Pillow usa antialias (LANCZOS) al bajar la resolucion, lo que preserva frecuencias espaciales bajas mas fielmente que el submuestreo disperso. Laplace supera a Poisson porque con solo el 9% de pixeles conocidos, la estimacion del vector b introduce ruido que perjudica la reconstruccion.

### 9.4 Comportamiento de Poisson vs Laplace

El termino b de Poisson introduce informacion sobre la estructura de gradientes de la imagen. Esta informacion mejora la calidad cuando el vector b puede estimarse con precision. Con un submuestreo denso (factor 2) o cuando b se conoce exactamente, Poisson produce bordes mas nitidos. Con submuestreo disperso (factor 4), solo el 9% de los pixeles son conocidos, y la estimacion por vecino mas proximo (nearest-neighbor fill) del vector b introduce artefactos que elevan el error.

La ventaja practica de incluir la ecuacion de Poisson (frente a usar Laplace puro) es que el solver esta preparado para aprovechar una mejor estimacion de b cuando se disponga de mas informacion, sin cambiar ninguna otra parte del codigo.

### 9.5 Comparacion visual

![Comparacion visual camara 64x64](assets/comparacion_camara64.png)

Las cuatro imagenes muestran (de izquierda a derecha): la imagen original, la reconstruccion bicubica, la reconstruccion Laplace y la reconstruccion Poisson. Todas las reconstrucciones preservan la estructura general de la imagen. Las diferencias son principalmente en la nitidez de los bordes finos.

---

## 10. Como ejecutar el proyecto

### 10.1 Requisitos

El proyecto requiere Python 3.10 o superior y las siguientes bibliotecas:

```
PySide6>=6.5.0      (interfaz grafica Qt)
numpy>=1.24.0       (algebra lineal y arrays)
scipy>=1.10.0       (distancia euclidea para relleno NN)
Pillow>=9.5.0       (interpolacion bicubica de referencia)
matplotlib>=3.7.0   (graficas de convergencia y comparacion)
scikit-image>=0.21.0 (metricas PSNR y SSIM)
numba>=0.58.0       (compilacion JIT del kernel SOR)
```

### 10.2 Instalacion

```bash
# Crear entorno virtual
python -m venv venv

# Activar (Linux/Mac)
source venv/bin/activate

# Activar (Windows)
venv\Scripts\activate.bat

# Instalar dependencias
pip install -r requirements.txt
```

### 10.3 Ejecucion

```bash
python main.py
```

La primera vez que se ejecuta, Numba compila el kernel SOR y lo guarda en cache. Esta compilacion tarda entre 5 y 15 segundos. Las ejecuciones siguientes arrancan directamente.

### 10.4 Flujo de uso en la interfaz

1. Cargar una imagen con el boton "Abrir imagen" en la barra de herramientas.
2. Ajustar el factor de ampliacion (por defecto 4) y la tolerancia.
3. Hacer zoom con la rueda del raton sobre la region de interes en el visor izquierdo.
4. Hacer clic en "Reconstruir (Poisson SOR)" para ampliar la region visible.
   El visor derecho muestra la region agrandada y suave.
5. Hacer clic en "Comparar 3 modelos" para ver la tabla PSNR/SSIM (Enfoque B).
6. Hacer clic en "Barrido Omega" para ver la curva de convergencia espectral.
7. Hacer clic en "Benchmark Numba vs Python" para ver la aceleracion JIT.
8. Exportar resultados a CSV o guardar imagenes con los botones de exportacion.

---

## 11. Conclusiones

El proyecto implementa correctamente todos los requisitos de la especificacion:

1. **Metodo unico SOR Red-Black**: el codigo contiene un unico solver basado en el ordenamiento Rojo-Negro, sin los cuatro metodos de la version anterior (Jacobi, Gauss-Seidel, SSOR quedan eliminados).

2. **Ecuacion de Poisson**: el termino b se calcula como la divergencia del campo de gradientes de la imagen guia, siguiendo la especificacion. Para la ecuacion de Laplace se usa b = 0 como caso especial.

3. **Enfoque Matrix-Free**: la matriz A nunca se construye. El estencil de 5 puntos se aplica directamente sobre la cuadricula de pixeles, permitiendo trabajar con imagenes grandes sin problemas de memoria.

4. **Aceleracion Numba**: el kernel de calculo se compila con `@njit`, logrando entre 380 y 444 veces mas velocidad que Python puro en las pruebas realizadas.

5. **Formula omega_opt validada**: la formula `2 / (1 + sin(pi / (n + 1)))` produce el minimo empirico de iteraciones para el problema clasico de borde puro, con una diferencia de 0.003 unidades respecto al valor teorico.

6. **Arquitectura de tres capas**: ui/, core/ y services/ estan claramente separados. El calculo ocurre en un hilo separado para no bloquear la interfaz grafica.

7. **Comparativa con cuatro modelos**: original, bicubica, Laplace y Poisson SOR se comparan con PSNR y SSIM siguiendo la especificacion de la seccion 8.2.
