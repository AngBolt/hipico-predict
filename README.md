# hipico-predict 🏇

Predictor de apuestas para el **Concurso Hípico Internacional de Gijón (Las Mestas)**, con datos en vivo de [online.equipe.com](https://online.equipe.com/shows/81588).

## Modalidades cubiertas

| Apuesta | Qué es | Coste |
|---|---|---|
| **Ganador de serie** | El mejor binomio de cada serie (10 caballos consecutivos por orden de salida; máx. 4 series, solo CSI4*/CSI2*) | — |
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

### 1b. Composición de las series de apuestas (calibrada con el programa real)
- Solo hay apuestas en pruebas **CSI4\*** y **CSI2\*** (los CSIYH1\* de caballos jóvenes no, aunque sí aportan historial al modelo).
- Series = bloques de **10 consecutivos por orden de salida**, máximo **4 series por prueba**, **ancladas al final de la lista**: si sobran caballos, los primeros del orden de salida quedan fuera de las apuestas.
- Verificado con el GP CSI2* del 29/08: 57 inscritos → 4 series de 10 empezando en el dorsal 18 (Teresa Arias Cueva).
- Configurable en `predict.py`: `SERIE_SIZE`, `MAX_SERIES`, `BET_LEVELS`.

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
| `DECAY` | 0.6 | lo reciente pesa más (por día) |
| `LEVEL_BONUS` | 1.0 | sin bonus por nivel (no mejoró) |
| `W_COMBO / W_RIDER / W_HORSE` | 0.4 / 0.1 / 0.5 | el caballo es lo que más pesa |
| `K_SMOOTH` | 0.5 | ½ observación "ficticia" en el prior |
| `PRIOR` | 0.5 | binomio desconocido = mediocre, ni bueno ni malo |
| `W_CLEAR` | 0.5 | el cero-faltas pesa tanto como el percentil |
| `SERIE_SIZE / MAX_SERIES` | 10 / 4 | estructura oficial de las series |
| `CONF_MIN` | 0.025 | umbral de confianza para apostar |

### 4. Predicciones
- **Series**: bloques oficiales de 10 (ver 1b).
- **Ganador de serie** = mayor `score` de la serie. **Gemela** = top-2. **Combinada de 3** = top-3 (⇒ 3 parejas).
- **Ganador de prueba** = mayor `score` global.
- **Triple gemela**: en las 3 últimas series de la última prueba con apuestas del día se marcan los top-3 de cada serie (3 gemelas/serie ⇒ 27 combinaciones, 8,10 €).

### 5. Confianza y estrategia de banca 💰
`confianza = score(top1) − score(top2)` dentro de la serie.

Con series de 10 caballos el azar es duro (ganador 10 %, gemela 2,2 %, combinada 6,7 %).
Backtest mié→vie (27 series evaluadas, muestra pequeña):
- **Gemela simple: ~11 % (×5 el azar)** ← aquí está el valor.
- **Combinada de 3: ~19 % (×3 el azar).**
- Ganador de serie: ~7 % (sin ventaja clara).

**Regla**: jugar gemela/combinada solo en series FIABLES (`confianza ≥ 0.025`, marcadas ⭐),
evitar series igualadas; la triple gemela obliga a marcar las 3 series del final.

### 6. Resultados del backtest (semana Gijón 2026)
- 9 pruebas con apuestas, 35 series (los CSIYH1* quedan fuera). Martes sin historial (línea base).
- Ver tabla completa y detalle serie a serie en `index.html`.

### 7. Ideas para futuras versiones
- [ ] **Prior FEI/Longines**: ranking mundial del jinete y del caballo como prior del primer día
  (base de datos FEI: https://data.fei.org). Sustituiría el `PRIOR = 0.5` plano.
- [ ] **Tiempo relativo**: usar el tiempo normalizado dentro de los cero-faltas como señal fina.
- [ ] **Series oficiales exactas**: confirmar cada día la composición real de las series del
  programa de apuestas (regla actual: bloques de 10 anclados al final, máx. 4; ver 1b).
- [ ] **Payouts reales**: registrar los dividendos pagados por gemela/triple para optimizar
  valor esperado (EV) en vez de tasa de acierto.
- [ ] **Re-ajuste continuo**: repetir el grid search al acabar cada concurso y guardar
  histórico multi-concurso (los binomios repiten año a año).
