# hipico-predict 🏇

Predictor de apuestas para el **Concurso Hípico Internacional de Gijón (Las Mestas)**, con datos en vivo de [online.equipe.com](https://online.equipe.com/shows/81588).

## Modalidades cubiertas

| Apuesta | Qué es | Coste |
|---|---|---|
| **Ganador de serie** | El mejor binomio de cada serie (~6 caballos por orden de salida) | — |
| **Gemela** | Los 2 mejores de la serie, sin importar el orden (1 pareja) | 2 € |
| **Combinada de 3** | 3 caballos combinados entre sí = 3 gemelas | 6 € |
| **Ganador de la prueba** | El mejor de toda la prueba | — |
| **Triple gemela** | Solo en las **3 últimas series de la última prueba del día**: acertar la gemela de las 3. Puedes marcar varias gemelas por serie y el coste se multiplica: 3×3×3 = 27 combinaciones × 0,30 € = 8,10 € | 0,30 €/comb. |

## Uso

```bash
python fetch_data.py     # descarga schedule + resultados/listas de salida (API Equipe)
python predict.py        # genera index.html con predicciones y backtest
```

Abre **[index.html](index.html)**: predicciones de las próximas pruebas, backtest contra resultados reales y detalle serie a serie. Para verlo online, activa GitHub Pages (Settings → Pages → branch main, root).

---

## El algoritmo (v2) — documentación para futuras actualizaciones

### 1. Datos de entrada
- API pública de Equipe: `GET /api/v1/meetings/{id}/schedule` (pruebas y fechas) y
  `GET /api/v1/class_sections/{cs_id}` (lista de salida, `rank`, faltas por fase, `position` = orden de salida).
- Solo se usan las pruebas con `class_no` numérico (los trofeos reales, no listas maestras ni avisos).

### 2. Rendimiento por participación
Para cada binomio en cada prueba ya disputada con `n` participantes:

```
percentil = 1 − (rank − 1) / (n − 1)        # 1 = ganó, 0 = último
clear     = 1 si terminó con 0 faltas totales (saltos + tiempo), si no 0
perf      = (1 − W_CLEAR) · percentil + W_CLEAR · clear      # W_CLEAR = 0.35
```
Eliminados / no clasificados → `perf = 0`.

### 3. Puntuación de un binomio antes de una prueba
Tres historiales, cada uno con media ponderada y **suavizado bayesiano** hacia el prior:

```
peso(obs) = DECAY^(días de antigüedad) · (LEVEL_BONUS si mismo nivel CSI4*/CSI2*/CSIYH1*)
smooth(H) = (K_SMOOTH·PRIOR + Σ peso·perf) / (K_SMOOTH + Σ peso)

score = W_COMBO·smooth(jinete+caballo) + W_RIDER·smooth(jinete) + W_HORSE·smooth(caballo)
```

Parámetros actuales (ajustados por grid search sobre el backtest mié→vie):

| Parámetro | Valor | Significado |
|---|---|---|
| `DECAY` | 0.85 | lo reciente pesa más (por día) |
| `LEVEL_BONUS` | 1.5 | historial del mismo nivel pesa ×1.5 |
| `W_COMBO / W_RIDER / W_HORSE` | 0.4 / 0.2 / 0.4 | el caballo importa tanto como el binomio |
| `K_SMOOTH` | 1.0 | 1 observación "ficticia" en el prior |
| `PRIOR` | 0.5 | binomio desconocido = mediocre, ni bueno ni malo |
| `W_CLEAR` | 0.35 | peso del cero-faltas en el rendimiento |
| `SERIE_TARGET` | 6 | tamaño objetivo de serie (orden de salida) |
| `CONF_MIN` | 0.044 | umbral de confianza para apostar |

### 4. Predicciones
- **Series**: la lista de salida se parte en grupos de ~6 por orden de salida.
- **Ganador de serie** = mayor `score` de la serie. **Gemela** = top-2. **Combinada de 3** = top-3 (⇒ 3 parejas).
- **Ganador de prueba** = mayor `score` global.
- **Triple gemela**: en las 3 últimas series de la última prueba del día se marcan los top-3 de cada serie (3 gemelas/serie ⇒ 27 combinaciones, 8,10 €).

### 5. Confianza y estrategia de banca 💰
`confianza = score(top1) − score(top2)` dentro de la serie.

Backtest (mié→vie, sin mirar el futuro): en las series con `confianza ≥ 0.044` el acierto sube a
**~28 % ganador** y **~38 % combinada de 3** (azar: 17 % / 20 %). En series igualadas la ventaja
desaparece → **regla: apostar solo en series FIABLES** (marcadas ⭐ en el HTML) y pasar del resto,
salvo la triple gemela que obliga a jugar las 3 series.

### 6. Resultados del backtest (semana Gijón 2026)
- 10 pruebas, 74 series. Martes sin historial (línea base ≈ azar).
- Desde el miércoles: ganador de serie 24 %, combinada de 3 31 %.
- Series fiables: ganador 28 %, combinada 38 %.

### 7. Ideas para futuras versiones
- [ ] **Prior FEI/Longines**: ranking mundial del jinete y del caballo como prior del primer día
  (base de datos FEI: https://data.fei.org). Sustituiría el `PRIOR = 0.5` plano.
- [ ] **Tiempo relativo**: usar el tiempo normalizado dentro de los cero-faltas como señal fina.
- [ ] **Series oficiales**: leer la composición real de las series que publica la organización
  (ahora se aproxima por orden de salida, `SERIE_TARGET`).
- [ ] **Payouts reales**: registrar los dividendos pagados por gemela/triple para optimizar
  valor esperado (EV) en vez de tasa de acierto.
- [ ] **Re-ajuste continuo**: repetir el grid search al acabar cada concurso y guardar
  histórico multi-concurso (los binomios repiten año a año).
