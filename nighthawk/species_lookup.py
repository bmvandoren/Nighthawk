"""Runtime species-candidate lookup for the Nighthawk NFC detector.

Given a recording's lat/lon and the month it was made, returns the set of
model taxa plausibly present in the area — one set per taxonomic level
(species, group, family, order).  This drives the geographic candidate filter
in run_reconstructed_model, which narrows the per-level subselect lists before
detection extraction, exactly mirroring how --ap-mask works.

The lookup table is a per-taxonomy CSV subset of a GBIF/eBird species-presence
cube (built offline by subset_lookup_for_taxonomy.py in nighthawk-training).
Its schema is:

    lat_bin, lon_bin, month, taxon_level, taxon_code, count

where lat_bin/lon_bin are the integer SW-corner degrees of each ~1° DMSG
cell, and taxon_level is one of 'species', 'group', 'family', 'order'.

Spatial pooling (haversine neighbourhood search) happens at query time so the
radius is tunable per-call without rebuilding the table.

Public API
----------
get_candidates(table, lat, lon, month,
               radius_km=200.0, min_count=1) -> dict[str, set[str]]

    table   — path to the taxonomy's species_lookup_table.csv, or a
              preloaded DataFrame.  Pass a preloaded DataFrame when calling
              repeatedly (e.g. from an inference loop) to avoid re-reading
              the file on every call.
    returns — {'species': set, 'group': set, 'family': set, 'order': set}
              Sets contain the taxon_code / name strings that appear in the
              model's *.txt files.  An empty set for a level means no taxa at
              that level were found within radius_km for that month.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Union

import pandas as pd


EARTH_RADIUS_KM = 6371.0
_LEVELS = ("species", "group", "family", "order")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two WGS84 points."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def get_candidates(
    table: Union[str, Path, pd.DataFrame],
    lat: float,
    lon: float,
    month: int,
    radius_km: float = 200.0,
    min_count: int = 1,
) -> dict[str, set[str]]:
    """Return per-level candidate taxon sets for a given location and month.

    Parameters
    ----------
    table     : path to species_lookup_table.csv or preloaded DataFrame.
    lat, lon  : WGS84 decimal degrees of the recording location.
    month     : calendar month (1–12).
    radius_km : neighbourhood pooling radius in km (great-circle distance
                from query point to cell centre = lat_bin+0.5, lon_bin+0.5).
                Default 200 km — a conservative single-night NFC displacement.
    min_count : minimum pooled count to include a taxon (applied to the sum
                across all cells in the neighbourhood for this month).

    Returns
    -------
    dict with keys 'species', 'group', 'family', 'order'; each value is a
    set of taxon_code strings (empty set if none found).
    """
    if not isinstance(table, pd.DataFrame):
        table = pd.read_csv(table)

    month_rows = table[table["month"] == month]
    if month_rows.empty:
        return {lvl: set() for lvl in _LEVELS}

    # Find all distinct cells (by SW-corner bins) and filter to those within
    # radius_km.  Cell centre is at lat_bin+0.5, lon_bin+0.5.
    cells = month_rows[["lat_bin", "lon_bin"]].drop_duplicates()
    nearby = set()
    for _, row in cells.iterrows():
        cell_lat = row["lat_bin"] + 0.5
        cell_lon = row["lon_bin"] + 0.5
        if haversine_km(lat, lon, cell_lat, cell_lon) <= radius_km:
            nearby.add((int(row["lat_bin"]), int(row["lon_bin"])))

    if not nearby:
        return {lvl: set() for lvl in _LEVELS}

    # Filter to nearby cells and pool (sum) counts per (taxon_level, taxon_code).
    mask = month_rows.apply(
        lambda r: (int(r["lat_bin"]), int(r["lon_bin"])) in nearby, axis=1
    )
    pooled = (
        month_rows[mask]
        .groupby(["taxon_level", "taxon_code"])["count"]
        .sum()
        .reset_index()
    )
    pooled = pooled[pooled["count"] >= min_count]

    result: dict[str, set[str]] = {lvl: set() for lvl in _LEVELS}
    for _, row in pooled.iterrows():
        lvl = row["taxon_level"]
        if lvl in result:
            result[lvl].add(row["taxon_code"])

    return result
