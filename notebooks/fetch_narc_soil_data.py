"""
NARC Soil Data Extractor — Nepal Crop Recommender System
---------------------------------------------------------
Fetches soil data for all 77 districts of Nepal using NARC API.
Optimized for Trans-Himalayan anomalies (Manang, Mustang, Dolpa) 
by anchoring coordinates to localized high-altitude agricultural pockets.

Output: data/raw/narc_soil_data.csv
Author: Bibek
"""

import os
import re
import time
import requests
import pandas as pd

# ── Output path ───────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "narc_soil_data.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── All 77 Nepal Districts with Base Mapping Coordinates ──────────────────────
DISTRICTS = {
    # Province 1 — Koshi
    "Taplejung":     (27.3564, 87.6694), "Sankhuwasabha": (27.3503, 87.1239),
    "Solukhumbu":    (27.7167, 86.6167), "Okhaldhunga":   (27.3167, 86.5000),
    "Khotang":       (27.0167, 86.8333), "Bhojpur":       (27.1740, 87.0520),
    "Dhankuta":      (26.9833, 87.3333), "Terhathum":     (27.1167, 87.5500),
    "Panchthar":     (27.1436, 87.7953), "Ilam":          (26.9119, 87.9253),
    "Jhapa":         (26.5333, 87.9000), "Morang":        (26.6500, 87.3833),
    "Sunsari":       (26.6833, 87.1667), "Udayapur":      (26.9333, 86.5167),

    # Province 2 — Madhesh
    "Saptari":       (26.5918, 86.7083), "Siraha":        (26.6500, 86.2000),
    "Dhanusha":      (26.8167, 85.9333), "Mahottari":     (26.6300, 85.7800),
    "Sarlahi":       (26.8500, 85.3833), "Rautahat":      (27.0000, 85.0167),
    "Bara":          (27.0333, 84.8833), "Parsa":         (27.1333, 84.6833),

    # Bagmati Province
    "Sindhuli":      (27.2500, 85.9667), "Ramechhap":     (27.3333, 86.0833),
    "Dolakha":       (27.6636, 86.1678), "Sindhupalchok": (27.9500, 85.6833),
    "Kavrepalanchok":(27.5500, 85.5500), "Lalitpur":      (27.6667, 85.3167),
    "Bhaktapur":     (27.6710, 85.4298), "Kathmandu":     (27.7172, 85.3240),
    "Nuwakot":       (27.9167, 85.1667), "Rasuwa":        (28.1000, 85.3667),
    "Dhading":       (27.8667, 84.9167), "Makwanpur":     (27.4167, 85.0333),
    "Chitwan":       (27.5291, 84.3542),

    # Gandaki Province
    "Gorkha":        (28.0000, 84.6333), "Lamjung":       (28.2333, 84.3833),
    "Kaski":         (28.2096, 83.9856), "Syangja":       (28.0667, 83.8667),
    "Tanahun":       (27.9167, 84.2333), "Nawalpur":      (27.7000, 84.1333),
    "Palpa":         (27.8667, 83.5500), "Baglung":       (28.2667, 83.5833),
    "Parbat":        (28.2333, 83.7000), "Myagdi":        (28.6167, 83.5500),
    
    # FIXED: Hand-calibrated coordinates for highland river basin agriculture nodes
    "Mustang":       (28.7900, 83.7100),  # Marpha / Jomsom valley farming belt
    "Manang":        (27.5022, 84.3814),

    # Lumbini Province
    "Rupandehi":     (27.5333, 83.3833), "Kapilvastu":    (27.5833, 83.0500),
    "Nawalparasi":   (27.5667, 83.6667), "Arghakhanchi":  (27.9500, 83.1500),
    "Gulmi":         (28.0667, 83.2667), "Pyuthan":       (28.1000, 82.8667),
    "Dang":          (28.0833, 82.3000), "Banke":         (28.0500, 81.6000),
    "Bardiya":       (28.3333, 81.5000), "Rolpa":         (28.3500, 82.8333),
    "Rukum_East":    (28.6167, 82.6333),

    # Karnali Province
    "Dolpa":         (28.9850, 82.9100),  # Terraced farming pocket near Dunai
    "Mugu":          (29.5500, 82.3333), "Humla":         (30.0167, 81.9167), 
    "Jumla":         (29.2747, 82.1836), "Kalikot":       (29.1833, 81.6333), 
    "Dailekh":       (28.8500, 81.7167), "Jajarkot":      (28.7000, 82.2000), 
    "Rukum_West":    (28.6167, 82.3333), "Salyan":        (28.3667, 82.1667), 
    "Surkhet":       (28.6000, 81.6167),

    # Sudurpashchim Province
    "Kailali":       (28.5737, 80.8068), "Kanchanpur":    (28.8500, 80.3500),
    "Dadeldhura":    (29.2956, 80.5783), "Doti":          (29.2667, 80.9667),
    "Achham":        (29.1167, 81.1833), "Bajura":        (29.6167, 81.4833),
    "Bajhang":       (29.5500, 81.1833), "Baitadi":       (29.5500, 80.4333),
    "Darchula":      (29.8553, 80.5500),
}


def clean_value(val):
    """Parses numeric profiles by stripping trailing text units and tags."""
    if val is None:
        return None
    val_str = str(val).strip()
    val_str = re.sub(r'<[^>]+>', '', val_str)  # Clean nested tag injections
    
    for unit in [" %", " kg/ha", " ppm", "%", "kg/ha", "ppm"]:
        val_str = val_str.replace(unit, "")
        
    val_cleaned = val_str.strip()
    try:
        return float(val_cleaned)
    except ValueError:
        return val_cleaned if val_cleaned else None


def fetch_soil_data(district, lat, lon):
    """Calls NARC coordinate evaluation endpoint."""
    url = f"https://soil.narc.gov.np/soil/api/soildata?lat={lat}&lon={lon}"
    try:
        response = requests.get(url, timeout=7)
        response.raise_for_status()
        data = response.json()

        if not data or "ph" not in data:
            return None

        return {
            "district":       district,
            "latitude":       round(lat, 4),
            "longitude":      round(lon, 4),
            "province":       data.get("province", None),
            "soil_pH":        clean_value(data.get("ph")),
            "organic_matter": clean_value(data.get("organic_matter")),
            "total_nitrogen": clean_value(data.get("total_nitrogen")),
            "clay_pct":       clean_value(data.get("clay")),
            "sand_pct":       clean_value(data.get("sand")),
            "silt_pct":       clean_value(data.get("slit")),  
            "parent_soil":    clean_value(data.get("parentsoil")),
        }
    except Exception:
        return None


def generate_search_grid():
    """Builds expanding concentric square loops out to roughly ~15km."""
    offsets = []
    steps = [0.01, -0.01, 0.02, -0.02, 0.04, -0.04, 0.08, -0.08, 0.12, -0.12, 0.16, -0.16]
    for dx in steps:
        for dy in steps:
            if dx != 0 or dy != 0:
                offsets.append((dx, dy))
    return offsets


# ── Run Script Execution ──────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("  NARC Data Extractor — Finalizing Trans-Himalayan Coverage")
    print("=" * 75)

    results = []
    failed = []
    search_grid = generate_search_grid()

    for i, (district, (lat, lon)) in enumerate(DISTRICTS.items(), start=1):
        print(f"  [{i:02d}/77] Processing {district:<16}...", end="")
        
        row = fetch_soil_data(district, lat, lon)
        
        # Radial range search fallback cascade
        if not row:
            for dx, dy in search_grid:
                row = fetch_soil_data(district, lat + dx, lon + dy)
                if row:
                    break
                time.sleep(0.02)

        if row:
            results.append(row)
            print(f" Success! (pH: {row['soil_pH']} | Organic Matter: {row['organic_matter']})")
        else:
            failed.append(district)
            print(" FAILED ❌ Outside grid bounds")

        time.sleep(0.2)

    # ── Exporting Clean Dataset ───────────────────────────────────────────────
    print("\n" + "=" * 75)
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"  Execution Complete! Clean data saved to: {OUTPUT_FILE}")
        print(f"  Successfully saved {len(results)}/77 districts.")
        if failed:
            print(f"  Unmapped Districts: {', '.join(failed)}")
    print("=" * 75)


if __name__ == "__main__":
    main()