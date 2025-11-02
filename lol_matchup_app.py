import os
import sys
import time
import json
import sqlite3
import threading
import traceback
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import requests
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QTextEdit, QMessageBox, QSpacerItem, QSizePolicy
)

# ---------------------------
# CONFIG (your account/region)
# ---------------------------
RIOT_API_KEY = os.environ.get("RIOT_API_KEY", "")
PLATFORM = "NA1"       # Summoner endpoints (platform routing)
REGIONAL = "americas"  # Match/account endpoints (regional routing)
RIOT_NAME = "Child"    # From "Child#630"
RIOT_TAG  = "630"

DB_PATH = "lol_history.db"
USER_AGENT = "LoL-Stats-Student-App/1.0 (contact: local)"

# ---------------------------
# Helpers
# ---------------------------
def riot_headers() -> Dict[str, str]:
    if not RIOT_API_KEY:
        raise RuntimeError("RIOT_API_KEY not set in environment.")
    return {
        "X-Riot-Token": RIOT_API_KEY,
        "User-Agent": USER_AGENT,
    }

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

# ---------------------------
# Riot API client (minimal)
# ---------------------------
class RiotClient:
    def __init__(self, platform: str, regional: str):
        self.platform = platform
        self.regional = regional

    def get_puuid_by_riot_id(self, gameName: str, tagLine: str) -> str:
        # account-v1 by-riot-id (regional routing)
        url = f"https://{self.regional}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{gameName}/{tagLine}"
        r = requests.get(url, headers=riot_headers(), timeout=20)
        r.raise_for_status()
        data = r.json()
        return data["puuid"]

    def get_match_ids(self, puuid: str, start: int = 0, count: int = 100) -> List[str]:
        # match-v5 list by puuid
        url = f"https://{self.regional}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {"start": start, "count": count}
        r = requests.get(url, headers=riot_headers(), params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def get_match(self, match_id: str) -> Dict[str, Any]:
        url = f"https://{self.regional}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        r = requests.get(url, headers=riot_headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def get_latest_ddragon_version(self) -> str:
        url = "https://ddragon.leagueoflegends.com/api/versions.json"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        versions = r.json()
        return versions[0] if versions else "13.1.1"

    def get_champion_mapping(self) -> Dict[int, str]:
        """
        Returns {champion_id (int) : champion_name (str)} using Data Dragon.
        """
        version = self.get_latest_ddragon_version()
        url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        data = r.json()["data"]
        mapping = {}
        for champ_name, info in data.items():
            # info['key'] is a string numeric id
            cid = int(info.get("key", "0"))
            mapping[cid] = champ_name
        return mapping

# ---------------------------
# Database
# ---------------------------
class LolDB:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.ensure_schema()

    def ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS matches (
          match_id TEXT PRIMARY KEY,
          platform TEXT,
          game_start_ts INTEGER,
          queue_id INTEGER,
          game_duration INTEGER,
          patch TEXT
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS participants (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          match_id TEXT,
          puuid TEXT,
          summoner_name TEXT,
          champion_id INTEGER,
          champion_name TEXT,
          team_id INTEGER,
          is_me INTEGER,
          win INTEGER,
          kills INTEGER,
          deaths INTEGER,
          assists INTEGER,
          cs INTEGER,
          gold INTEGER,
          vision_score INTEGER,
          damage_dealt INTEGER,
          FOREIGN KEY(match_id) REFERENCES matches(match_id)
        )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_part_me ON participants(is_me, champion_name)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_part_match ON participants(match_id)")
        self.conn.commit()

    def match_exists(self, match_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM matches WHERE match_id=?", (match_id,))
        return cur.fetchone() is not None

    def insert_match(self, m: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT OR IGNORE INTO matches(match_id, platform, game_start_ts, queue_id, game_duration, patch)
        VALUES (?,?,?,?,?,?)
        """, (
            m["match_id"], m.get("platform"), safe_int(m.get("game_start_ts", 0)),
            safe_int(m.get("queue_id", 0)), safe_int(m.get("game_duration", 0)),
            m.get("patch", "")
        ))
        self.conn.commit()

    def insert_participant(self, p: Dict[str, Any]):
        cur = self.conn.cursor()
        cur.execute("""
        INSERT INTO participants(
          match_id, puuid, summoner_name, champion_id, champion_name, team_id, is_me,
          win, kills, deaths, assists, cs, gold, vision_score, damage_dealt
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            p["match_id"], p.get("puuid"), p.get("summoner_name"),
            safe_int(p.get("champion_id", 0)), p.get("champion_name"),
            safe_int(p.get("team_id", 0)), safe_int(p.get("is_me", 0)),
            safe_int(p.get("win", 0)), safe_int(p.get("kills", 0)),
            safe_int(p.get("deaths", 0)), safe_int(p.get("assists", 0)),
            safe_int(p.get("cs", 0)), safe_int(p.get("gold", 0)),
            safe_int(p.get("vision_score", 0)), safe_int(p.get("damage_dealt", 0))
        ))
        self.conn.commit()

    def best_vs_enemy(self, enemy_champ_query: str) -> Optional[Dict[str, Any]]:
        """
        Returns the best of your champions against an enemy champion name
        (case-insensitive exact or partial). Best = highest winrate then most games.
        """
        q = """
        WITH agg AS (
          SELECT
            p_me.champion_name AS my_champ,
            COUNT(*) AS games,
            SUM(p_me.win) AS wins,
            SUM(p_me.kills) AS k_sum,
            SUM(p_me.deaths) AS d_sum,
            SUM(p_me.assists) AS a_sum
          FROM participants p_me
          JOIN participants p_op
            ON p_me.match_id = p_op.match_id
           AND p_me.team_id != p_op.team_id
          WHERE p_me.is_me = 1
            AND LOWER(p_op.champion_name) LIKE '%' || LOWER(?) || '%'
          GROUP BY p_me.champion_name
        )
        SELECT my_champ,
               games,
               wins,
               (CAST(wins AS FLOAT)/games) AS winrate,
               (CAST(k_sum AS FLOAT)/games) AS k_avg,
               (CAST(d_sum AS FLOAT)/games) AS d_avg,
               (CAST(a_sum AS FLOAT)/games) AS a_avg
        FROM agg
        WHERE games >= 1
        ORDER BY winrate DESC, games DESC
        LIMIT 1
        """
        cur = self.conn.cursor()
        cur.execute(q, (enemy_champ_query.strip(),))
        row = cur.fetchone()
        return dict(row) if row else None

# ---------------------------
# Ingestion / ETL
# ---------------------------
def ingest_matches(db: LolDB, riot: RiotClient, gameName: str, tagLine: str,
                   max_to_fetch: int = 150, sleep_sec: float = 1.2) -> Tuple[int, int]:
    """
    Downloads up to `max_to_fetch` most recent matches for this Riot ID and inserts
    matches + participants. Returns (matches_added, participants_added).
    """
    puuid = riot.get_puuid_by_riot_id(gameName, tagLine)
    # Pull in pages of 100 to be efficient
    total_added_matches = 0
    total_added_participants = 0

    # Champion mapping is optional – match JSON already has championName,
    # but we keep mapping for robustness if needed later.
    try:
        champ_map = riot.get_champion_mapping()
    except Exception:
        champ_map = {}

    fetched = 0
    start = 0
    while fetched < max_to_fetch:
        remaining = max_to_fetch - fetched
        count = 100 if remaining > 100 else remaining
        ids = riot.get_match_ids(puuid, start=start, count=count)
        if not ids:
            break
        start += len(ids)
        fetched += len(ids)

        for mid in ids:
            if db.match_exists(mid):
                continue

            try:
                m = riot.get_match(mid)
            except requests.HTTPError as e:
                # Skip 404/429 gracefully
                print(f"Error fetching {mid}: {e}", file=sys.stderr)
                continue

            info = m.get("info", {})
            meta = m.get("metadata", {})
            # Insert match
            rec_m = {
                "match_id": meta.get("matchId", mid),
                "platform": PLATFORM,
                "game_start_ts": safe_int(info.get("gameStartTimestamp", 0)) // 1000,
                "queue_id": safe_int(info.get("queueId", 0)),
                "game_duration": safe_int(info.get("gameDuration", 0)),
                "patch": info.get("gameVersion", "")
            }
            db.insert_match(rec_m)
            total_added_matches += 1

            for p in info.get("participants", []):
                is_me = 1 if p.get("puuid") == puuid else 0
                cid = safe_int(p.get("championId", 0))
                cname = p.get("championName")
                if (not cname or cname == "None") and cid in champ_map:
                    cname = champ_map[cid]
                # Compute CS
                cs = safe_int(p.get("totalMinionsKilled", 0)) + safe_int(p.get("neutralMinionsKilled", 0))
                rec_p = {
                    "match_id": rec_m["match_id"],
                    "puuid": p.get("puuid"),
                    "summoner_name": p.get("summonerName"),
                    "champion_id": cid,
                    "champion_name": cname,
                    "team_id": safe_int(p.get("teamId", 0)),
                    "is_me": is_me,
                    "win": 1 if p.get("win") else 0,
                    "kills": safe_int(p.get("kills", 0)),
                    "deaths": safe_int(p.get("deaths", 0)),
                    "assists": safe_int(p.get("assists", 0)),
                    "cs": cs,
                    "gold": safe_int(p.get("goldEarned", 0)),
                    "vision_score": safe_int(p.get("visionScore", 0)),
                    "damage_dealt": safe_int(p.get("totalDamageDealtToChampions", 0)),
                }
                db.insert_participant(rec_p)
                total_added_participants += 1

            # be gentle with rate limits
            time.sleep(sleep_sec)

    return total_added_matches, total_added_participants

# ---------------------------
# GUI
# ---------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LoL Best Counter (Your Own History)")
        self.setFixedSize(520, 300)

        self.db = LolDB(DB_PATH)
        self.riot = RiotClient(PLATFORM, REGIONAL)

        layout = QVBoxLayout()
        top = QHBoxLayout()

        self.input_label = QLabel("Enemy champion:")
        self.input = QLineEdit()
        self.input.setPlaceholderText("e.g., Ahri")
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.on_search)

        top.addWidget(self.input_label)
        top.addWidget(self.input)
        top.addWidget(self.search_btn)

        # Sync section
        sync_row = QHBoxLayout()
        self.sync_btn = QPushButton("Sync Matches")
        self.sync_btn.clicked.connect(self.on_sync)
        self.sync_status = QLabel("Ready.")
        sync_row.addWidget(self.sync_btn)
        sync_row.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        sync_row.addWidget(self.sync_status)

        self.out = QTextEdit()
        self.out.setReadOnly(True)
        self.out.setPlaceholderText("Results will appear here...")

        layout.addLayout(top)
        layout.addLayout(sync_row)
        layout.addWidget(self.out)
        self.setLayout(layout)

        # First-run tip
        QTimer.singleShot(250, lambda: self.info("Tip", "Click 'Sync Matches' to pull your latest games if this is your first run."))

    def info(self, title: str, msg: str):
        QMessageBox.information(self, title, msg)

    def error(self, title: str, msg: str):
        QMessageBox.critical(self, title, msg)

    def on_search(self):
        enemy = self.input.text().strip()
        if not enemy:
            self.error("Missing input", "Please enter an enemy champion name.")
            return

        try:
            res = self.db.best_vs_enemy(enemy)
            if not res:
                self.out.setPlainText(f"No data found versus '{enemy}'. "
                                      f"Try syncing matches or check spelling.")
                return
            my_champ = res["my_champ"]
            games = safe_int(res["games"])
            wins = safe_int(res["wins"])
            losses = games - wins
            winrate = round(100 * safe_float(res["winrate"]), 1)
            k = round(safe_float(res["k_avg"]), 2)
            d = round(safe_float(res["d_avg"]), 2)
            a = round(safe_float(res["a_avg"]), 2)

            self.out.setPlainText(
                f"Best Champion vs {enemy} (from your match history)\n"
                f"-----------------------------------------------\n"
                f"Champion: {my_champ}\n"
                f"Winrate: {winrate}%  ({wins}W - {losses}L) over {games} games\n"
                f"Average K/D/A: {k} / {d} / {a}\n"
            )
        except Exception as e:
            traceback.print_exc()
            self.error("Query error", str(e))

    def on_sync(self):
        self.sync_btn.setEnabled(False)
        self.sync_status.setText("Syncing...")
        self.out.setPlainText("Fetching matches… this may take a minute.\n")

        def worker():
            try:
                added_m, added_p = ingest_matches(self.db, self.riot, RIOT_NAME, RIOT_TAG, max_to_fetch=150)
                msg = f"Sync complete. New matches: {added_m}, participants: {added_p}."
            except Exception as e:
                msg = f"Sync failed: {e}"
            # update UI from main thread
            def done():
                self.sync_btn.setEnabled(True)
                self.sync_status.setText(msg)
                self.out.append(msg)
            QTimer.singleShot(0, done)

        threading.Thread(target=worker, daemon=True).start()

# ---------------------------
# main
# ---------------------------
def main():
    if not RIOT_API_KEY:
        print("ERROR: Set RIOT_API_KEY environment variable first.", file=sys.stderr)
        sys.exit(1)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
