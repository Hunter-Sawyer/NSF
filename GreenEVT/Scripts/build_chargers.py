"""
Build a SUMO charging-station file from the City of Palo Alto session data
("Electric Vehicle Charging Station Usage (July 2011 - Dec 2020)", the city
download is named "ChargePoint Data CY20Q4.csv"), with real port counts and
measured port ratings, plus a table linking each sim station to the real one.

    python build_chargers.py --csv "../data/Palo Alto/ChargePoint Data CY20Q4.csv" \
        --net PA.network.net.xml --out chargers_palo_alto.xml --max-radius 400

Placement follows the team's assign_charging_stations_to_streets
(activity_gen_assign_populations.py): each station's lat/lon goes on the
nearest non-internal lane within --radius m (100, as in the team's script)
that allows passenger cars.
Differences:
- Optional second pass (--max-radius) for stations the first pass couldn't
  place. It runs after every near station is placed, so it never moves them.
  The committed Palo Alto file uses 400 m, which places the Cambridge and
  Sherman garages (the network lacks their side streets) about 190-320 m out.
- One sim station per (real station, port type). Stations that mix Level 1
  and Level 2 ports become two sim stations, so each keeps its own rating.
- power = measured per-port rating: the 95th percentile of energy / charging
  time over that group's sessions (falls back to the port type's rating
  across all stations when a group has few sessions). Sessions under 15 min
  are skipped, since short sessions give noisy rates.
- totalPower = ports x power, and the stop is ports x 7.5 m long (one
  electric_vehicle is 5 m plus a 2.5 m gap), so as many cars as the station
  has ports can charge at once.
- Stations sharing a lane get separate, non-overlapping stretches.
- Writes <out>_stations.csv mapping each sim id to the real station name,
  ports, rating and placement, and lists stations that couldn't be placed.
"""
import argparse
import collections
import csv
import os
import statistics
import sys
from xml.sax.saxutils import quoteattr

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
import sumolib  # noqa: E402

CAR_SPACE_M = 7.5        # electric_vehicle: length 5.0 + minGap 2.5
MIN_SESSION_H = 0.25
MIN_SESSIONS = 30
PCTL = 0.95


def _hours(text):
    try:
        h, m, s = text.split(":")
        return int(h) + int(m) / 60 + int(s) / 3600
    except ValueError:
        return 0.0


def _pctl(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def load_groups(csv_path):
    """(station name, port type) -> ports, sessions, rates, location."""
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8-sig")))
    # A port's type is the one most of its sessions report (a few are blank).
    port_types = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["Port Type"]:
            port_types[(r["Station Name"], r["Port Number"])][r["Port Type"]] += 1
    groups = {}
    for r in rows:
        key_port = (r["Station Name"], r["Port Number"])
        if key_port not in port_types:
            continue
        ptype = port_types[key_port].most_common(1)[0][0]
        g = groups.setdefault((r["Station Name"], ptype), {
            "ports": set(), "sessions": 0, "rates": [], "locs": collections.Counter(),
            "address": collections.Counter()})
        g["ports"].add(r["Port Number"])
        g["sessions"] += 1
        g["locs"][(float(r["Latitude"]), float(r["Longitude"]))] += 1
        g["address"][r["Address 1"]] += 1
        h = _hours(r["Charging Time (hh:mm:ss)"])
        if h >= MIN_SESSION_H:
            g["rates"].append(float(r["Energy (kWh)"]) / h)
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True)
    p.add_argument("--net", required=True)
    p.add_argument("--out", default="chargers_palo_alto.xml")
    p.add_argument("--radius", type=float, default=100.0,
                   help="first pass: max distance (m) from a station to its lane; the team's script uses 100")
    p.add_argument("--max-radius", type=float, default=None,
                   help="second pass, for stations the first pass couldn't place (default: no second pass). "
                        "Every station within --radius is placed first, so far ones never take their lanes")
    args = p.parse_args()
    radii = [args.radius] + ([args.max_radius] if args.max_radius and args.max_radius > args.radius else [])

    groups = load_groups(args.csv)
    type_rate = {}
    for ptype in {k[1] for k in groups}:
        rates = [x for k, g in groups.items() if k[1] == ptype for x in g["rates"]]
        type_rate[ptype] = _pctl(rates, PCTL)

    net = sumolib.net.readNet(args.net)
    used = collections.defaultdict(float)    # lane id -> metres already taken
    unplaced = []
    for (name, ptype), g in sorted(groups.items()):
        if len(g["rates"]) >= MIN_SESSIONS:
            rate_kw, rate_src = _pctl(g["rates"], PCTL), "station"
        else:
            rate_kw, rate_src = type_rate[ptype], "port_type"
        (lat, lon), _ = g["locs"].most_common(1)[0]
        unplaced.append({"station_name": name, "port_type": ptype, "ports": len(g["ports"]),
                         "port_numbers": " ".join(sorted(g["ports"], key=lambda s: (len(s), s))),
                         "address": g["address"].most_common(1)[0][0], "lat": lat, "lon": lon,
                         "sessions": g["sessions"], "power_kw": round(rate_kw, 2), "power_source": rate_src})

    placed = []
    for radius in radii:
        left = []
        for info in unplaced:
            need = info["ports"] * CAR_SPACE_M
            x, y = net.convertLonLat2XY(info["lon"], info["lat"])
            spot = None
            for edge, dist in sorted(net.getNeighboringEdges(x, y, radius), key=lambda e: e[1]):
                if edge.getID().startswith(":"):
                    continue
                for lane in edge.getLanes():
                    if lane.allows("passenger") and lane.getLength() - used[lane.getID()] >= need:
                        spot = (lane, dist)
                        break
                if spot:
                    break
            if spot is None:
                left.append(info)
                continue
            lane, dist = spot
            start = used[lane.getID()]
            used[lane.getID()] = start + need
            info.update(lane=lane.getID(), start_pos=round(start, 2), end_pos=round(start + need, 2),
                        distance_m=round(dist, 1))
            placed.append(info)
        unplaced = left
    placed.sort(key=lambda s: (s["station_name"], s["port_type"]))

    with open(args.out, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        # (no "--" allowed inside an XML comment)
        f.write(f"<!-- Built by build_chargers.py from {os.path.basename(args.csv)}, lanes within "
                f"{' then '.join(f'{r:g}' for r in radii)} m; {len(unplaced)} real station groups not placed; "
                f"sim id -> real station in {os.path.basename(args.out)[:-4]}_stations.csv -->\n")
        f.write("<additional>\n")
        for i, s in enumerate(placed):
            s["sim_id"] = str(i)
            w = round(s["power_kw"] * 1000)
            f.write(f'    <chargingStation id="{i}" lane={quoteattr(s["lane"])} '
                    f'startPos="{s["start_pos"]}" endPos="{s["end_pos"]}" friendlyPos="true" '
                    f'power="{w}" totalPower="{w * s["ports"]}" '
                    f'name={quoteattr(s["station_name"] + " (" + s["port_type"] + ")")}/>\n')
        f.write("</additional>\n")

    fields = ["sim_id", "station_name", "port_type", "ports", "port_numbers", "power_kw",
              "power_source", "sessions", "address", "lat", "lon", "lane", "start_pos",
              "end_pos", "distance_m"]
    with open(args.out[:-4] + "_stations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in placed + unplaced:
            w.writerow({k: s.get(k, "") for k in fields})

    total_ports = sum(s["ports"] for s in placed + unplaced)
    print(f"placed {len(placed)} sim stations ({sum(s['ports'] for s in placed)} of {total_ports} ports); "
          f"unplaced {len(unplaced)}")
    print("port type ratings (p95, kW): " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(type_rate.items())))
    for s in unplaced:
        print(f"  not placed (no lane within {radii[-1]:g} m with {s['ports'] * CAR_SPACE_M:g} m free): "
              f"{s['station_name']} ({s['port_type']}, {s['ports']} ports) at {s['lat']}, {s['lon']}")
    print(f"Wrote {args.out} and {args.out[:-4]}_stations.csv")


if __name__ == "__main__":
    main()
