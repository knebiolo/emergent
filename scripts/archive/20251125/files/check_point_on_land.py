"""Check whether given lon/lat points fall on land according to ENC LNDARE layer.

Usage:
  python scripts\check_point_on_land.py --port Baltimore --lon -76.5318 --lat 39.39684
"""
import argparse
from shapely.geometry import Point
from emergent.ship_abm.config import SIMULATION_BOUNDS

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--port', '-p', default='Baltimore')
    p.add_argument('--lon', type=float, required=True)
    p.add_argument('--lat', type=float, required=True)
    args = p.parse_args()

    import json, os
    from emergent.ship_abm import simulation_core

    # create a minimal simulation to load ENC (simulation_core will load ENC lazily)
    sim = simulation_core.simulation(port=args.port, n=1)
    # ensure ENC is loaded
    sim._load_enc()
    land_gdf = sim.enc_data.get('LNDARE')
    if land_gdf is None or land_gdf.empty:
        print("No LNDARE found for this port in ENC data")
    else:
        pt = Point(args.lon, args.lat)
        found = False
        for geom in land_gdf.geometry:
            if geom.contains(pt):
                print(f"Point {args.lon},{args.lat} IS on land (inside LNDARE)")
                found = True
                break
        if not found:
            print(f"Point {args.lon},{args.lat} is NOT on land according to ENC LNDARE")
