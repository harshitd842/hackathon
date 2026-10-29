"""
seed_inventory.py — create the local farm_storehouse.db with sample stock.

Run once:  python seed_inventory.py
Then the app (and the MCP server) can check what the farmer already has.
"""
import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "farm_storehouse.db")

ROWS = [
    # item_name,               category,   quantity_available, unit, unit_cost_inr
    ("Mancozeb 75% WP",        "Fungicide", 0.0,  "kg", 350.0),
    ("Copper Oxychloride",     "Fungicide", 1.5,  "kg", 420.0),
    ("Neem Oil Extract",       "Organic",   5.0,  "L",  180.0),
    ("Propiconazole 25% EC",   "Fungicide", 0.75, "L",  560.0),
]

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()
cur.execute(
    """
    CREATE TABLE IF NOT EXISTS inventory (
        item_name          TEXT PRIMARY KEY,
        category           TEXT,
        quantity_available REAL,
        unit               TEXT,
        unit_cost_inr      REAL
    )
    """
)
cur.executemany("INSERT OR REPLACE INTO inventory VALUES (?, ?, ?, ?, ?)", ROWS)
conn.commit()
conn.close()
print(f"farm_storehouse.db ready at {DB_PATH} with {len(ROWS)} items.")
