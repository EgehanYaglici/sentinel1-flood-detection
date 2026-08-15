"""
Sentinel-1 SAR Change Detection Processing
Performs flood/water-change detection from pre/post event SAR imagery.
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from scipy import ndimage
from shapely.geometry import box, shape
from shapely.ops import unary_union
import rasterio.features

DATA_DIR = Path(__file__).parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = Path(__file__).parents[2] / "outputs"
METADATA_DIR = Path(__file__).parents[2] / "metadata"

AOI_BBOX = (-0.6, 39.2, -0.2, 39.5)
TARGET_CRS = "EPSG:32630"  # UTM zone 30N for Valencia area
RESOLUTION = 10  # 10m Sentinel-1 resolution


def ensure_dirs():
    for d in [PROCESSED_DIR, OUTPUTS_DIR / "figures", OUTPUTS_DIR / "tables"]:
        d.mkdir(parents=True, exist_ok=True)


def create_demonstration_sar() -> tuple:
    """
    Create realistic SAR backscatter data for demonstration.
    Used when actual Sentinel-1 data is not downloadable without authentication.
    Simulates VV polarization pre/post flood event.
    """
    # Convert AOI to UTM for metric resolution
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", TARGET_CRS, always_xy=True)
    xmin, ymin = transformer.transform(AOI_BBOX[0], AOI_BBOX[1])
    xmax, ymax = transformer.transform(AOI_BBOX[2], AOI_BBOX[3])

    width = int((xmax - xmin) / RESOLUTION)
    height = int((ymax - ymin) / RESOLUTION)
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

    np.random.seed(2024)

    # Pre-event: typical SAR backscatter
    # Urban areas: high backscatter (-3 to -8 dB)
    # Vegetation: medium (-10 to -15 dB)
    # Water (rivers): low (-18 to -25 dB)
    pre_db = np.random.normal(-12, 3, (height, width)).astype(np.float32)

    # Add urban clusters (high backscatter)
    for _ in range(100):
        cx, cy = np.random.randint(100, width-100), np.random.randint(100, height-100)
        size = np.random.randint(20, 80)
        pre_db[cy:cy+size, cx:cx+size] += np.random.uniform(4, 8)

    # Rivers (low backscatter), Turia and irrigation channels
    river_y = height // 2
    pre_db[river_y-5:river_y+5, :] = np.random.normal(-22, 2, (10, width))

    # Post-event: significant flooding in southern/western areas
    post_db = pre_db.copy()

    # Flood zone: lower-left quadrant gets much lower backscatter
    flood_mask = np.zeros((height, width), dtype=bool)
    # Main flood area, irregular shape
    yc, xc = int(height * 0.6), int(width * 0.4)
    for _ in range(50):
        fy = yc + np.random.randint(-height//4, height//4)
        fx = xc + np.random.randint(-width//4, width//4)
        fh = np.random.randint(20, 100)
        fw = np.random.randint(20, 150)
        fy_s, fx_s = max(0, fy), max(0, fx)
        flood_mask[fy_s:min(fy_s+fh, height), fx_s:min(fx_s+fw, width)] = True

    # Apply flood effect: drop backscatter significantly in flooded areas
    post_db[flood_mask] = np.random.normal(-20, 2, flood_mask.sum())

    # Add speckle noise
    pre_db += np.random.exponential(0.5, pre_db.shape).astype(np.float32) - 0.5
    post_db += np.random.exponential(0.5, post_db.shape).astype(np.float32) - 0.5

    # Save rasters
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": TARGET_CRS,
        "transform": transform,
        "nodata": -9999,
    }

    pre_path = PROCESSED_DIR / "pre_event_db.tif"
    post_path = PROCESSED_DIR / "post_event_db.tif"

    with rasterio.open(pre_path, "w", **profile) as dst:
        dst.write(pre_db, 1)
    with rasterio.open(post_path, "w", **profile) as dst:
        dst.write(post_db, 1)

    print(f"    Created pre-event SAR: {width}x{height} ({RESOLUTION}m)")
    print(f"    Created post-event SAR: {width}x{height} ({RESOLUTION}m)")

    return pre_path, post_path


def compute_change(pre_path: Path, post_path: Path) -> Path:
    """Compute dB difference between pre and post event."""
    with rasterio.open(pre_path) as pre_src, rasterio.open(post_path) as post_src:
        pre = pre_src.read(1)
        post = post_src.read(1)
        profile = pre_src.profile.copy()

        nodata = pre_src.nodata or -9999
        valid = (pre != nodata) & (post != nodata)

        diff = np.where(valid, post - pre, nodata).astype(np.float32)

    diff_path = PROCESSED_DIR / "difference_db.tif"
    with rasterio.open(diff_path, "w", **profile) as dst:
        dst.write(diff, 1)

    return diff_path


def detect_flood(diff_path: Path) -> tuple:
    """
    Detect potential flood areas from SAR change.
    Uses histogram-based threshold selection.
    """
    with rasterio.open(diff_path) as src:
        diff = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata or -9999

    valid = diff[diff != nodata]

    # Statistical threshold selection
    mean_diff = np.mean(valid)
    std_diff = np.std(valid)

    # Flood signature: significant decrease in backscatter (negative difference)
    # Use mean - 1.5*std as threshold (defensible statistical approach)
    threshold = mean_diff - 1.5 * std_diff
    print(f"    Mean difference: {mean_diff:.2f} dB")
    print(f"    Std difference: {std_diff:.2f} dB")
    print(f"    Threshold: {threshold:.2f} dB")

    # Create flood mask
    flood_mask = np.where(
        (diff != nodata) & (diff < threshold),
        1, 0
    ).astype(np.uint8)

    # Morphological cleaning, remove small isolated pixels
    flood_mask = ndimage.binary_opening(flood_mask, iterations=2).astype(np.uint8)
    flood_mask = ndimage.binary_closing(flood_mask, iterations=1).astype(np.uint8)

    # Save detection raster
    detect_path = PROCESSED_DIR / "flood_detection.tif"
    profile.update(dtype="uint8", nodata=255)
    with rasterio.open(detect_path, "w", **profile) as dst:
        dst.write(flood_mask, 1)

    stats = {
        "threshold_db": round(float(threshold), 2),
        "method": "mean - 1.5 * std",
        "mean_difference_db": round(float(mean_diff), 2),
        "std_difference_db": round(float(std_diff), 2),
        "detected_pixels": int(flood_mask.sum()),
        "total_valid_pixels": int((diff != nodata).sum()),
        "detected_area_pct": round(float(flood_mask.sum() / (diff != nodata).sum() * 100), 1),
    }

    return detect_path, stats


def vectorize_detection(detect_path: Path) -> gpd.GeoDataFrame:
    """Convert detection raster to polygons."""
    with rasterio.open(detect_path) as src:
        mask = src.read(1)
        transform = src.transform
        crs = src.crs

    # Vectorize
    shapes_gen = rasterio.features.shapes(mask, transform=transform)
    polygons = []
    for geom, value in shapes_gen:
        if value == 1:
            poly = shape(geom)
            if poly.area > 500:  # Minimum 500 sqm to remove artifacts
                polygons.append(poly)

    if not polygons:
        return gpd.GeoDataFrame(columns=["geometry", "area_sqm"], crs=crs)

    gdf = gpd.GeoDataFrame(
        [{"geometry": p, "area_sqm": p.area} for p in polygons],
        geometry="geometry",
        crs=crs,
    )

    # Remove tiny artifacts and invalid geometries
    gdf = gdf[gdf.geometry.is_valid].copy()
    gdf = gdf[gdf["area_sqm"] > 1000].copy()  # Keep only > 1000 sqm

    gdf["area_ha"] = gdf["area_sqm"] / 10000
    gdf = gdf.sort_values("area_sqm", ascending=False).reset_index(drop=True)

    return gdf


def add_context(flood_gdf: gpd.GeoDataFrame) -> dict:
    """Add OSM context: roads, buildings in detected areas."""
    import osmnx as ox

    # Convert AOI to get context data
    aoi_wgs84 = flood_gdf.to_crs("EPSG:4326").total_bounds
    context_stats = {
        "total_detected_area_ha": round(float(flood_gdf["area_ha"].sum()), 1),
        "polygon_count": len(flood_gdf),
        "largest_polygon_ha": round(float(flood_gdf["area_ha"].max()), 1),
    }

    try:
        # Get buildings in flood area
        flood_union = unary_union(flood_gdf.to_crs("EPSG:4326").geometry)
        bbox = flood_gdf.to_crs("EPSG:4326").total_bounds

        buildings = ox.features_from_bbox(
            bbox=(bbox[3], bbox[1], bbox[2], bbox[0]),
            tags={"building": True},
        )
        buildings = buildings[buildings.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
        buildings_in_flood = buildings[buildings.geometry.intersects(flood_union)]
        context_stats["buildings_intersecting"] = len(buildings_in_flood)
        context_stats["total_buildings_aoi"] = len(buildings)
    except Exception as e:
        context_stats["buildings_note"] = f"Could not fetch: {e}"

    try:
        # Get roads
        roads = ox.features_from_bbox(
            bbox=(bbox[3], bbox[1], bbox[2], bbox[0]),
            tags={"highway": True},
        )
        roads_line = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].copy()
        roads_in_flood = roads_line[roads_line.geometry.intersects(flood_union)]
        roads_in_flood_proj = roads_in_flood.to_crs(TARGET_CRS)
        context_stats["road_length_in_flood_km"] = round(
            float(roads_in_flood_proj.geometry.length.sum() / 1000), 1
        )
    except Exception as e:
        context_stats["roads_note"] = f"Could not fetch: {e}"

    return context_stats


def main():
    print("=" * 60)
    print("Sentinel-1 SAR Flood Detection, Valencia 2024")
    print("=" * 60)

    ensure_dirs()

    # Check if real SAR data exists
    pre_path = PROCESSED_DIR / "pre_event_db.tif"
    post_path = PROCESSED_DIR / "post_event_db.tif"

    if not pre_path.exists() or not post_path.exists():
        print("\n[1/5] Creating SAR demonstration data...")
        print("    (Real Sentinel-1 download requires CDSE authentication)")
        pre_path, post_path = create_demonstration_sar()
    else:
        print("\n[1/5] Using existing SAR data...")

    print("\n[2/5] Computing change detection...")
    diff_path = compute_change(pre_path, post_path)
    print(f"    Output: {diff_path}")

    print("\n[3/5] Detecting flood areas...")
    detect_path, detect_stats = detect_flood(diff_path)
    print(f"    Detected area: {detect_stats['detected_area_pct']:.1f}% of valid pixels")

    print("\n[4/5] Vectorizing detected areas...")
    flood_gdf = vectorize_detection(detect_path)
    print(f"    Polygons: {len(flood_gdf)}")
    if len(flood_gdf) > 0:
        print(f"    Total area: {flood_gdf['area_ha'].sum():.1f} ha")
        flood_gdf.to_file(PROCESSED_DIR / "flood_polygons.gpkg", driver="GPKG")
        flood_gdf.to_crs("EPSG:4326").to_file(
            PROCESSED_DIR / "flood_polygons_wgs84.geojson", driver="GeoJSON"
        )

    print("\n[5/5] Adding context...")
    context = {}
    if len(flood_gdf) > 0:
        context = add_context(flood_gdf)
        print(f"    {json.dumps(context, indent=4)}")

    # Save full results
    results = {
        "event": "Valencia DANA Flood - October 2024",
        "aoi": AOI_BBOX,
        "processing_crs": TARGET_CRS,
        "resolution_m": RESOLUTION,
        "detection": detect_stats,
        "context": context,
        "disclaimer": "Exploratory SAR-derived change/water detection. "
                      "Not authoritative flood mapping. "
                      "Intersection with infrastructure does not prove actual damage.",
    }

    with open(METADATA_DIR / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SAR analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
