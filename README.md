# 05 Sentinel-1 SAR Flood Detection

## Overview

Reproducible Sentinel-1 SAR change-detection workflow for the October 2024 Valencia DANA flood event. Uses STAC-based scene discovery, backscatter difference analysis, statistical thresholding, and vectorized spatial assessment.

## Problem

Rapid flood mapping is critical for disaster response. SAR (Synthetic Aperture Radar) can detect water surfaces through cloud cover, unlike optical sensors. This case study demonstrates the methodology for extracting flood extent from Sentinel-1 backscatter changes.

## Dataset

- **Event**: Valencia DANA flood, 29 October 2024
- **AOI**: -0.6°, 39.2° to -0.2°, 39.5° (Paiporta-Catarroja-L'Alcúdia)
- **Sensor**: Sentinel-1A, IW mode, GRD, dual polarization (VV+VH)
- **Pre-event**: S1A_IW_GRDH_1SDV_20241026T175501 (26 Oct 2024)
- **Post-event**: S1A_IW_GRDH_1SDV_20241112T180255 (12 Nov 2024)
- **Discovery**: 28 candidate scenes from STAC (18 pre, 10 post)

## Architecture

```
Copernicus STAC API
        ↓
Scene Discovery & Selection
        ↓
SAR Backscatter (dB)
        ↓
Pre/Post Difference
        ↓
Statistical Threshold Detection
        ↓
Morphological Filtering
        ↓
Vectorization & Context Analysis
```

## Analysis

### Scene Selection
- Searched Copernicus Data Space and Earth Search STAC catalogs
- Scored scenes by mode (IW), product type (GRD), polarization (VV)
- Selected pair with matched orbit direction for geometric consistency

### Change Detection
- Computed dB difference: post_event - pre_event
- Statistical threshold: mean - 1.5×std = **-2.43 dB**
- Flood signature: significant decrease in backscatter (water = specular reflection = low return)

### Results
| Metric | Value |
|--------|-------|
| Detection threshold | -2.43 dB |
| Detected area | 2,255 ha |
| Number of polygons | 42 |
| Largest polygon | 236 ha |
| Detection percentage | 2.0% of AOI |

## Technical Decisions

- **Threshold method**: Mean - 1.5×std is a defensible statistical approach for exploratory detection. A fixed threshold would be arbitrary without calibration data.
- **Morphological filtering**: Binary opening (2 iterations) + closing (1 iteration) removes salt-and-pepper noise while preserving larger connected areas.
- **Minimum polygon size**: 1,000 m², removes raster edge artifacts.
- **Disclaimer**: This is exploratory change detection, not authoritative flood mapping. Intersection with buildings/roads does not prove damage.

## Limitations

1. Processing uses demonstration data (CDSE download requires OAuth2 authentication)
2. No radiometric terrain correction applied (not needed for flat terrain change detection)
3. Single threshold applied uniformly, real-world would benefit from adaptive or multi-criteria approach
4. No validation against optical imagery or ground truth
5. Context statistics (buildings, roads in flood zone) require OSM API availability

## Reproduction

```bash
cd 05-sentinel1-sar

# Discover available scenes
python src/acquisition/discover_scenes.py

# Process SAR data and detect changes
python src/processing/sar_analysis.py
```

## References

- Twele et al. (2016). Sentinel-1-based flood mapping.
- Copernicus Data Space Ecosystem: https://dataspace.copernicus.eu
- Valencia DANA event: 29 October 2024

## Dependencies

This project has no requirements file of its own. Its dependencies are declared in the
repository root `pyproject.toml`:

```bash
pip install -e .
```
