# Superseded artefacts in this project

`analysis_results.json` records a run with an adaptive threshold of -2.43 dB
producing 2,255 ha across 42 polygons. `outputs/web/data/flood_polygons.geojson`
records a later run at a fixed -3 dB producing 16.76 km2 across 84 polygons.
Neither is reproduced by the current pipeline and neither records the scene
pair it used.

Both are superseded by `src/processing/sar_change_detection.py`, which pins the
STAC item ids, states the multilook window, and publishes the full threshold
sensitivity. Its results are in `outputs/tables/sar_final.json`.

The important difference is not the number. The earlier runs paired scenes from
different relative orbits, which inflates the change distribution from
sigma 3.27 dB to sigma 5.65 dB and makes any fixed dB threshold unreliable.
