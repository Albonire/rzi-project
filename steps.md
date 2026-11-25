# Documentacion del Proyecto RZI

Este archivo describia una version anterior del proyecto y quedo desactualizado.

La documentacion vigente esta en:

- `README.md`: resumen, instalacion, uso y resultados.
- `docs/RZI_Documento_de_Estudio.pdf`: documento de estudio completo para la sustentacion
  (fuente editable en `docs/estudio/documento.md`).

## Integracion adicional en este repositorio

- `core/upscaler.py::generar_frames_pipeline` y `core/sor_solver.py::resolver_screened_con_snapshots`:
  generacion didactica paso a paso sobre el modelo RZI vigente (NN -> bicubica u0 -> ciclos
  back-projection + SOR Red-Black -> back-projection final -> anti-ringing / realce opcional),
  conectada al dialogo interactivo de animacion en `ui/main_window.py` con lista lateral de etapas,
  leyenda matematica y comparacion sincronizada contra la referencia bicubica.
- `ui/zoom_viewer.py` (`marcar_region`, `enfocar_region`) y opcion `Vista con inset` en
  `ui/main_window.py`: permiten alternar en vivo entre la salida ampliada directa y la vista
  contextual con recuadro rojo (region fuente) y verde (inset SR).
- `sustentacion/`: material de sustentacion (documento, diapositivas HTML/PDF y proyecto de
  video Remotion en `sustentacion/rzi_video/`).
