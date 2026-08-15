"""
Sentinel-1 SAR Scene Discovery for Valencia Flood Event (October 2024)
Uses Copernicus Data Space Ecosystem STAC API.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pystac_client import Client
from shapely.geometry import box, mapping

DATA_DIR = Path(__file__).parents[2] / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_DIR = Path(__file__).parents[2] / "metadata"

# Valencia flood AOI, focus on affected areas south/west of Valencia city
# Approximate area: Paiporta, Catarroja, L'Alcúdia region
AOI_BBOX = (-0.6, 39.2, -0.2, 39.5)  # lon_min, lat_min, lon_max, lat_max
AOI_CRS = "EPSG:4326"

# Event date: 29 October 2024 (DANA event)
EVENT_DATE = "2024-10-29"
PRE_EVENT_WINDOW = ("2024-10-01", "2024-10-28")
POST_EVENT_WINDOW = ("2024-10-29", "2024-11-15")

# STAC endpoints to try
STAC_ENDPOINTS = [
    "https://catalogue.dataspace.copernicus.eu/stac",
    "https://earth-search.aws.element84.com/v1",
    "https://planetarycomputer.microsoft.com/api/stac/v1",
]

COLLECTION_IDS = [
    "sentinel-1-grd",
    "SENTINEL-1",
    "sentinel-1-rtc",
]


def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def create_aoi() -> gpd.GeoDataFrame:
    """Create AOI GeoDataFrame."""
    aoi_geom = box(*AOI_BBOX)
    gdf = gpd.GeoDataFrame(
        [{"name": "Valencia Flood AOI", "geometry": aoi_geom}],
        geometry="geometry",
        crs=AOI_CRS,
    )
    gdf.to_file(RAW_DIR / "aoi_valencia.geojson", driver="GeoJSON")
    return gdf


def search_stac(period: tuple, label: str) -> list:
    """Search for Sentinel-1 scenes across multiple STAC catalogs."""
    aoi_geom = box(*AOI_BBOX)
    results = []

    for endpoint in STAC_ENDPOINTS:
        print(f"    Trying: {endpoint}")
        try:
            client = Client.open(endpoint)
            collections = [c.id for c in client.get_collections()]

            # Find matching S1 collection
            s1_collection = None
            for cid in COLLECTION_IDS:
                if cid in collections:
                    s1_collection = cid
                    break

            if not s1_collection:
                # Try partial match
                for c in collections:
                    if "sentinel-1" in c.lower() or "sentinel1" in c.lower():
                        s1_collection = c
                        break

            if not s1_collection:
                print(f"      No Sentinel-1 collection found")
                continue

            print(f"      Collection: {s1_collection}")

            search = client.search(
                collections=[s1_collection],
                intersects=mapping(aoi_geom),
                datetime=f"{period[0]}/{period[1]}",
                max_items=50,
            )

            items = list(search.items())
            print(f"      Found: {len(items)} items")

            for item in items:
                props = item.properties
                results.append({
                    "id": item.id,
                    "datetime": props.get("datetime", ""),
                    "platform": props.get("platform", props.get("constellation", "")),
                    "orbit_direction": props.get("sat:orbit_state", props.get("s1:orbit_source", "")),
                    "relative_orbit": props.get("sat:relative_orbit", ""),
                    "polarization": str(props.get("sar:polarizations", props.get("s1:polarization", ""))),
                    "instrument_mode": props.get("sar:instrument_mode", props.get("s1:instrument_mode", "")),
                    "product_type": props.get("sar:product_type", props.get("s1:product_type", "")),
                    "source": endpoint,
                    "period": label,
                    "bbox": list(item.bbox) if item.bbox else [],
                })

            if results:
                break  # Use first successful source

        except Exception as e:
            print(f"      Error: {e}")
            continue

    return results


def select_best_pair(scenes: list) -> dict:
    """Select best pre/post event scene pair based on consistency."""
    df = pd.DataFrame(scenes)

    if df.empty:
        return {"pre": None, "post": None, "reason": "No scenes found"}

    pre = df[df["period"] == "pre-event"].copy()
    post = df[df["period"] == "post-event"].copy()

    if pre.empty or post.empty:
        return {"pre": None, "post": None, "reason": "Missing pre or post scenes"}

    # Prefer IW GRD VV
    def score_scene(row):
        score = 0
        if "IW" in str(row.get("instrument_mode", "")):
            score += 10
        if "GRD" in str(row.get("product_type", "")):
            score += 5
        if "VV" in str(row.get("polarization", "")):
            score += 3
        return score

    pre["score"] = pre.apply(score_scene, axis=1)
    post["score"] = post.apply(score_scene, axis=1)

    # Try to match orbit direction
    best_pre = pre.sort_values("score", ascending=False).iloc[0]
    matching_post = post[post["orbit_direction"] == best_pre["orbit_direction"]]

    if not matching_post.empty:
        best_post = matching_post.sort_values("score", ascending=False).iloc[0]
    else:
        best_post = post.sort_values("score", ascending=False).iloc[0]

    return {
        "pre": best_pre.to_dict(),
        "post": best_post.to_dict(),
        "reason": "Selected by orbit consistency, mode, and polarization",
    }


def main():
    print("=" * 60)
    print("Sentinel-1 SAR Scene Discovery, Valencia Flood 2024")
    print(f"AOI: {AOI_BBOX}")
    print(f"Event date: {EVENT_DATE}")
    print("=" * 60)

    ensure_dirs()
    create_aoi()

    print("\n[1/3] Searching pre-event scenes...")
    pre_scenes = search_stac(PRE_EVENT_WINDOW, "pre-event")

    print("\n[2/3] Searching post-event scenes...")
    post_scenes = search_stac(POST_EVENT_WINDOW, "post-event")

    all_scenes = pre_scenes + post_scenes

    print(f"\n    Total scenes found: {len(all_scenes)}")
    print(f"    Pre-event: {len(pre_scenes)}")
    print(f"    Post-event: {len(post_scenes)}")

    # Save all candidates
    if all_scenes:
        scenes_df = pd.DataFrame(all_scenes)
        scenes_df.to_csv(METADATA_DIR / "scene_candidates.csv", index=False)

    # Select best pair
    print("\n[3/3] Selecting best scene pair...")
    selection = select_best_pair(all_scenes)

    selection_meta = {
        "event": "Valencia DANA Flood",
        "event_date": EVENT_DATE,
        "aoi_bbox": AOI_BBOX,
        "pre_event_window": PRE_EVENT_WINDOW,
        "post_event_window": POST_EVENT_WINDOW,
        "total_candidates": len(all_scenes),
        "selected_pair": selection,
    }

    with open(METADATA_DIR / "scene_selection.json", "w") as f:
        json.dump(selection_meta, f, indent=2, default=str)

    if selection["pre"] and selection["post"]:
        print(f"\n  Selected pair:")
        print(f"    PRE:  {selection['pre']['id']} ({selection['pre']['datetime']})")
        print(f"    POST: {selection['post']['id']} ({selection['post']['datetime']})")
        print(f"    Reason: {selection['reason']}")
    else:
        print(f"\n  Could not select pair: {selection['reason']}")
        print("  Will proceed with synthetic demonstration data.")

    print("\n" + "=" * 60)
    print("Scene discovery complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
