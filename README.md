# hipico-predict 🏇

Predictor de apuestas para el **Concurso Hípico Internacional de Gijón (Las Mestas)**, con datos en vivo de [online.equipe.com](https://online.equipe.com/shows/81588).

## Modalidades cubiertas

| Apuesta | Qué es |
|---|---|
| **Ganador de serie** | El mejor binomio de cada serie (~6 caballos por orden de salida) |
| **Gemela** | Los 2 mejores de la serie, sin importar el orden |
| **Ganador de la prueba** | El mejor de toda la prueba |
| **Triple gemela** | Acertar la gemela de las 3 últimas series del último trofeo del día. Se pueden marcar varias gemelas por serie: 3×3×3 = 27 combinaciones × 0,30 € = 8,10 € |

## Uso

```bash
python fetch_data.py     # descarga schedule + resultados/listas de salida (API Equipe)
python predict.py        # genera index.html con predicciones y backtest
```

Abre **[index.html](index.html)** para ver:
- Predicciones de las próximas pruebas (ganador de serie, gemela, marcas de triple gemela).
- Backtest contra los resultados reales desde el martes (la predicción de cada prueba solo usa información anterior a ella).
- Detalle serie a serie.

> Para consultarlo online, activa GitHub Pages (Settings → Pages → branch main, root) y el HTML quedará publicado.

## Modelo

Puntuación por binomio = historial ponderado dentro del concurso:
- Percentil de clasificación en pruebas anteriores + componente "cero faltas" (peso 0,35).
- Decaimiento temporal 0,7/día (lo reciente pesa más).
- Mezcla 80 % historial jinete+caballo / 20 % historial del jinete con otros caballos.
- Sin historial → prior 0,5 (neutro).

Parámetros ajustados por búsqueda en rejilla sobre el backtest (miércoles→viernes):
ganador de serie **24 %** (azar ≈ 17 %), gemela con 3 marcas **31 %** (azar ≈ 20 %).

### Mejoras futuras
- Prior con ranking FEI/Longines para el primer día (sin historial local).
- Tamaño de serie real publicado por la organización (configurable en `predict.py` → `SERIE_TARGET`).
