import os
import sys
import time
import json
import sqlite3
import threading
import traceback
from collections import deque
from typing import Optional, List, Dict, Any, Tuple

import requests

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QSpacerItem, QSizePolicy, QComboBox, QScrollArea,
    QFrame, QProgressBar
)

# ---------------------------
# CONFIG
# ---------------------------
RIOT_API_KEY = os.environ.get("RIOT_API_KEY", "")
PLATFORM = "NA1"
REGIONAL = "americas"

DB_PATH = "lol_history.db"
USER_AGENT = "LoL-Stats-Student-App/1.4 (contact: local)"

# Summoner's Rift queues only
ALLOWED_QUEUES = {400, 430, 420, 440, 490}

TOPK = 5
MIN_GAMES = 1

LANE_MAP = {
    "Any": None,
    "Top": "TOP",
    "Jungle": "JUNGLE",
    "Mid": "MIDDLE",
    "ADC": "BOTTOM",
    "Support": "UTILITY",
}

# How deep to scan for total (pure /ids paging; no details fetched here).
# Raise if you want to compute the lifetime total; keep reasonable for time.
TOTAL_SCAN_CAP = 3000

# How many newest matches to actually download details for & insert per Sync.
DETAILS_FETCH_LIMIT = 1000

# ---------------------------
# RATE LIMITER (20 / 1s, 100 / 120s)
# ---------------------------
class RiotRateLimiter:
    def __init__(self, per1s=20, per120s=100):
        self.per1s = per1s
        self.per120s = per120s
        self.last_1s = deque()
        self.last_120s = deque()

    def wait(self):
        now = time.time()
        while self.last_1s and now - self.last_1s[0] >= 1.0:
            self.last_1s.popleft()
        while self.last_120s and now - self.last_120s[0] >= 120.0:
            self.last_120s.popleft()

        while len(self.last_1s) >= self.per1s or len(self.last_120s) >= self.per120s:
            t1 = (self.last_1s[0] + 1.0 - now) if self.last_1s else 0.0
            t2 = (self.last_120s[0] + 120.0 - now) if self.last_120s else 0.0
            sleep_for = max(0.01, min(x for x in [t1, t2] if x > 0.0))
            time.sleep(sleep_for)
            now = time.time()
            while self.last_1s and now - self.last_1s[0] >= 1.0:
                self.last_1s.popleft()
            while self.last_120s and now - self.last_120s[0] >= 120.0:
                self.last_120s.popleft()

        self.last_1s.append(time.time())
        self.last_120s.append(time.time())

# ---------------------------
# RIOT API CLIENT
# ---------------------------
def riot_headers() -> Dict[str, str]:
    if not RIOT_API_KEY:
        raise RuntimeError("RIOT_API_KEY not set in environment.")
    return {"X-Riot-Token": RIOT_API_KEY, "User-Agent": USER_AGENT}

class RiotClient:
    def __init__(self, platform: str, regional: str, limiter: RiotRateLimiter):
        self.platform = platform
        self.regional = regional
        self.limiter = limiter
        self._dd_version = None
        self._icon_cache: Dict[str, QPixmap] = {}

    def _get(self, url: str, headers: Dict[str, str], params: Dict[str, Any] = None, timeout=20):
        self.limiter.wait()
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        if r.status_code == 429:
            time.sleep(2.5)
            self.limiter.wait()
            r = requests.get(url, headers=headers, params=params, timeout=timeout)
        r.raise_for_status()
        return r

    # Resolve Riot ID → PUUID
    def get_puuid_by_riot_id(self, gameName: str, tagLine: str) -> str:
        url = f"https://{self.regional}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{gameName}/{tagLine}"
        r = self._get(url, headers=riot_headers())
        return r.json()["puuid"]

    # Get list of match IDs (paged). Returns up to `count` starting at `start`.
    def get_match_ids(self, puuid: str, start: int = 0, count: int = 100) -> List[str]:
        url = f"https://{self.regional}.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        r = self._get(url, headers=riot_headers(), params={"start": start, "count": count})
        return r.json()

    # Match details
    def get_match(self, match_id: str) -> Dict[str, Any]:
        url = f"https://{self.regional}.api.riotgames.com/lol/match/v5/matches/{match_id}"
        r = self._get(url, headers=riot_headers())
        return r.json()

    # Data Dragon — version and champion icons
    def dd_version(self) -> str:
        if self._dd_version:
            return self._dd_version
        url = "https://ddragon.leagueoflegends.com/api/versions.json"
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        versions = r.json()
        self._dd_version = versions[0] if versions else "13.1.1"
        return self._dd_version

    def get_icon(self, champion_name: str) -> Optional[QPixmap]:
        if not champion_name:
            return None
        name = champion_name.replace(" ", "")
        if name in self._icon_cache:
            return self._icon_cache[name]
        try:
            v = self.dd_version()
            url = f"https://ddragon.leagueoflegends.com/cdn/{v}/img/champion/{name}.png"
            r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
            if r.status_code != 200:
                return None
            pm = QPixmap()
            pm.loadFromData(r.content)
            self._icon_cache[name] = pm
            return pm
        except Exception:
            return None

# ---------------------------
# DB
# ---------------------------
def safe_int(x: Any, default: int = 0) -> int:
    try: return int(x)
    except Exception: return default

def safe_float(x: Any, default: float = 0.0) -> float:
    try: return float(x)
    except Exception: return default

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
          position TEXT,
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
        # ensure position column exists (upgrade path)
        cur.execute("PRAGMA table_info(participants)")
        cols = [r["name"] for r in cur.fetchall()]
        if "position" not in cols:
            cur.execute("ALTER TABLE participants ADD COLUMN position TEXT")
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
          match_id, puuid, summoner_name, champion_id, champion_name, team_id, is_me, position,
          win, kills, deaths, assists, cs, gold, vision_score, damage_dealt
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            p["match_id"], p.get("puuid"), p.get("summoner_name"),
            safe_int(p.get("champion_id", 0)), p.get("champion_name"),
            safe_int(p.get("team_id", 0)), safe_int(p.get("is_me", 0)),
            p.get("position"),
            safe_int(p.get("win", 0)), safe_int(p.get("kills", 0)),
            safe_int(p.get("deaths", 0)), safe_int(p.get("assists", 0)),
            safe_int(p.get("cs", 0)), safe_int(p.get("gold", 0)),
            safe_int(p.get("vision_score", 0)), safe_int(p.get("damage_dealt", 0))
        ))
        self.conn.commit()

    def top_counters(self, enemy_query: str, lane_pos: Optional[str], target_puuid: str) -> List[Dict[str, Any]]:
        params: List[Any] = []
        lane_filter_sql = ""
        if lane_pos:
            lane_filter_sql = "AND p_me.position = ?"
            params.append(lane_pos)

        q = f"""
        WITH base AS (
          SELECT p_me.champion_name AS my_champ,
                 m.queue_id AS queue_id,
                 p_me.win AS win,
                 p_me.kills AS k,
                 p_me.deaths AS d,
                 p_me.assists AS a
          FROM participants p_me
          JOIN participants p_op
            ON p_me.match_id = p_op.match_id
           AND p_me.team_id != p_op.team_id
          JOIN matches m
            ON p_me.match_id = m.match_id
          WHERE p_me.puuid = ?
            AND LOWER(p_op.champion_name) LIKE '%' || LOWER(?) || '%'
            {lane_filter_sql}
            AND m.queue_id IN ({",".join(str(q) for q in ALLOWED_QUEUES)})
        ),
        agg AS (
          SELECT my_champ,
                 COUNT(*) AS games,
                 SUM(win) AS wins,
                 AVG(CAST(k AS FLOAT)) AS k_avg,
                 AVG(CAST(d AS FLOAT)) AS d_avg,
                 AVG(CAST(a AS FLOAT)) AS a_avg
          FROM base
          GROUP BY my_champ
        )
        SELECT my_champ, games, wins,
               (CAST(wins AS FLOAT)/games) AS winrate,
               k_avg, d_avg, a_avg
        FROM agg
        WHERE games >= ?
        ORDER BY winrate DESC, games DESC
        LIMIT {TOPK}
        """
        params = [target_puuid, enemy_query.strip()] + params + [MIN_GAMES]
        cur = self.conn.cursor()
        cur.execute(q, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]

# ---------------------------
# SCAN + INGEST (with real progress)
# ---------------------------

def scan_all_ids(riot: RiotClient, puuid: str, cap: int, progress_cb=None) -> List[str]:
    """
    Page /ids until empty (or cap). Returns the full list we discovered.
    Calls progress_cb(current_ids_discovered) as it goes.
    """
    ids: List[str] = []
    start = 0
    while len(ids) < cap:
        batch = riot.get_match_ids(puuid, start=start, count=100)
        if not batch:
            break
        ids.extend(batch)
        start += len(batch)
        if progress_cb:
            progress_cb(len(ids))
    return ids[:cap]

def ingest_details_for_ids(db: LolDB, riot: RiotClient, puuid: str,
                           ids: List[str], progress_cb=None) -> Tuple[int, int]:
    added_m = 0
    added_p = 0
    for idx, mid in enumerate(ids, 1):
        if progress_cb:
            progress_cb(idx, len(ids))
        if db.match_exists(mid):
            continue
        try:
            m = riot.get_match(mid)
        except requests.HTTPError:
            continue
        info = m.get("info", {})
        meta = m.get("metadata", {})
        queue_id = safe_int(info.get("queueId", 0))
        if queue_id not in ALLOWED_QUEUES:
            continue
        rec_m = {
            "match_id": meta.get("matchId", mid),
            "platform": PLATFORM,
            "game_start_ts": safe_int(info.get("gameStartTimestamp", 0)) // 1000,
            "queue_id": queue_id,
            "game_duration": safe_int(info.get("gameDuration", 0)),
            "patch": info.get("gameVersion", "")
        }
        db.insert_match(rec_m)
        added_m += 1

        for p in info.get("participants", []):
            cs = safe_int(p.get("totalMinionsKilled", 0)) + safe_int(p.get("neutralMinionsKilled", 0))
            rec_p = {
                "match_id": rec_m["match_id"],
                "puuid": p.get("puuid"),
                "summoner_name": p.get("summonerName"),
                "champion_id": safe_int(p.get("championId", 0)),
                "champion_name": p.get("championName"),
                "team_id": safe_int(p.get("teamId", 0)),
                "is_me": 1 if p.get("puuid") == puuid else 0,
                "position": p.get("individualPosition"),
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
            added_p += 1
    return added_m, added_p

# ---------------------------
# GUI pieces
# ---------------------------
class CounterRow(QFrame):
    def __init__(self, icon: Optional[QPixmap], my_champ: str, winrate: float,
                 games: int, wins: int, k: float, d: float, a: float, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFrameShadow(QFrame.Shadow.Raised)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(10)

        icon_label = QLabel()
        if icon:
            icon_label.setPixmap(icon.scaled(48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            icon_label.setText("🛡️")
            icon_label.setFixedWidth(24)
        name_label = QLabel(f"<b>{my_champ}</b>")
        stats_label = QLabel(f"Winrate: {round(100*winrate,1)}%  ({wins}W/{games-wins}L)  |  Avg K/D/A: {k:.2f}/{d:.2f}/{a:.2f}")

        lay.addWidget(icon_label)
        lay.addWidget(name_label)
        lay.addStretch(1)
        lay.addWidget(stats_label)

# ---------------------------
# MAIN WINDOW
# ---------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LoL Best Counters (Your History)")
        self.setFixedSize(720, 560)

        self.db = LolDB(DB_PATH)
        self.limiter = RiotRateLimiter()
        self.riot = RiotClient(PLATFORM, REGIONAL, self.limiter)

        self.current_name = "Child"
        self.current_tag = "630"
        self.current_puuid: Optional[str] = None

        root = QVBoxLayout(self)

        # User row
        user_row = QHBoxLayout()
        self.name_input = QLineEdit(self.current_name)
        self.name_input.setPlaceholderText("Riot Name (e.g., Child)")
        self.tag_input = QLineEdit(self.current_tag)
        self.tag_input.setPlaceholderText("Tag (e.g., 630)")
        self.set_user_btn = QPushButton("Set User")
        self.set_user_btn.clicked.connect(self.on_set_user)
        self.user_status = QLabel("User: (not resolved)")
        user_row.addWidget(QLabel("Riot ID:"))
        user_row.addWidget(self.name_input)
        user_row.addWidget(QLabel("#"))
        user_row.addWidget(self.tag_input)
        user_row.addWidget(self.set_user_btn)
        user_row.addWidget(self.user_status)

        # Query row
        query_row = QHBoxLayout()
        self.enemy_input = QLineEdit()
        self.enemy_input.setPlaceholderText("Enemy champion (e.g., Ahri)")
        self.lane_combo = QComboBox()
        self.lane_combo.addItems(list(LANE_MAP.keys()))
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.on_search)
        query_row.addWidget(QLabel("Enemy:"))
        query_row.addWidget(self.enemy_input, 2)
        query_row.addSpacing(8)
        query_row.addWidget(QLabel("Lane:"))
        query_row.addWidget(self.lane_combo)
        query_row.addSpacing(8)
        query_row.addWidget(self.search_btn)

        # Sync row with deterministic progress
        sync_row = QHBoxLayout()
        self.sync_btn = QPushButton("Sync Matches")
        self.sync_btn.clicked.connect(self.on_sync)
        self.progress = QProgressBar()
        self.progress.setMinimum(0); self.progress.setMaximum(100); self.progress.setValue(0)
        self.sync_status = QLabel("Ready.")
        sync_row.addWidget(self.sync_btn)
        sync_row.addItem(QSpacerItem(10, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        sync_row.addWidget(QLabel("Progress:"))
        sync_row.addWidget(self.progress, 1)
        sync_row.addWidget(self.sync_status)

        # Results
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setContentsMargins(8, 8, 8, 8)
        self.results_layout.setSpacing(6)
        self.scroll.setWidget(self.results_container)

        root.addLayout(user_row)
        root.addLayout(query_row)
        root.addLayout(sync_row)
        root.addWidget(self.scroll)

        QTimer.singleShot(300, lambda: self.info("Tip", "Set User → Sync (computes total, then fetches SR matches) → Search."))

    # Helpers
    def info(self, title: str, msg: str): QMessageBox.information(self, title, msg)
    def error(self, title: str, msg: str): QMessageBox.critical(self, title, msg)
    def clear_results(self):
        while self.results_layout.count():
            w = self.results_layout.takeAt(0).widget()
            if w: w.deleteLater()

    # User
    def on_set_user(self):
        name = self.name_input.text().strip()
        tag = self.tag_input.text().strip()
        if not name or not tag:
            self.error("Missing Riot ID", "Enter both Riot Name and Tag (e.g., Child#630).")
            return
        self.current_name, self.current_tag = name, tag
        try:
            self.current_puuid = self.riot.get_puuid_by_riot_id(name, tag)
            self.user_status = QLabel(f"User: {name}#{tag} (resolved)")
            self.info("User Set", f"Resolved Riot ID for {name}#{tag}. Sync to ingest matches.")
        except Exception as e:
            traceback.print_exc()
            self.current_puuid = None
            self.error("Set User failed", str(e))

    # Search
    def on_search(self):
        if not self.current_puuid:
            try:
                self.current_puuid = self.riot.get_puuid_by_riot_id(self.current_name, self.current_tag)
            except Exception:
                self.error("Need a user", "Press 'Set User' first.")
                return

        enemy = self.enemy_input.text().strip()
        if not enemy:
            self.error("Missing input", "Enter an enemy champion.")
            return

        lane_label = self.lane_combo.currentText()
        lane_pos = LANE_MAP[lane_label]

        try:
            rows = self.db.top_counters(enemy, lane_pos, self.current_puuid)
            self.clear_results()
            if not rows:
                msg = f"No results for '{enemy}'" + (f" on {lane_label}" if lane_pos else "") + ". Try syncing more."
                self.results_layout.addWidget(QLabel(msg))
                return

            header = QLabel(f"<b>Top {len(rows)} counters vs {enemy}"
                            + (f" on {lane_label}" if lane_pos else "")
                            + f" — for {self.current_name}#{self.current_tag}</b>")
            self.results_layout.addWidget(header)

            for r in rows:
                my_champ = r["my_champ"]
                games = safe_int(r["games"])
                wins = safe_int(r["wins"])
                winrate = safe_float(r["winrate"])
                k = safe_float(r["k_avg"]); d = safe_float(r["d_avg"]); a = safe_float(r["a_avg"])
                icon = self.riot.get_icon(my_champ)
                self.results_layout.addWidget(CounterRow(icon, my_champ, winrate, games, wins, k, d, a, self))
            self.results_layout.addStretch(1)
        except Exception as e:
            traceback.print_exc()
            self.error("Query error", str(e))

    # Sync (two-phase progress)
    def on_sync(self):
        try:
            self.current_puuid = self.riot.get_puuid_by_riot_id(self.current_name, self.current_tag)
        except Exception:
            self.error("Need a user", "Press 'Set User' first.")
            return

        self.sync_btn.setEnabled(False)
        self.progress.setValue(0)
        self.sync_status.setText("Computing total…")

        def set_progress(pct: int):
            QTimer.singleShot(0, lambda: self.progress.setValue(max(0, min(100, pct))))

        def worker():
            try:
                # Phase 1 — list all IDs (up to TOTAL_SCAN_CAP)
                ids_collected = 0
                def on_ids_progress(cur_ids):
                    nonlocal ids_collected
                    ids_collected = cur_ids
                    # Map the ID scan to the first 50% of the bar
                    pct = int(min(100, (cur_ids / TOTAL_SCAN_CAP) * 50))
                    set_progress(pct)

                ids = scan_all_ids(self.riot, self.current_puuid, TOTAL_SCAN_CAP, progress_cb=on_ids_progress)
                total_ids = len(ids)

                # Decide how many to actually download details for this run
                wanted = min(total_ids, DETAILS_FETCH_LIMIT)

                # Phase 2 — fetch details for the newest `wanted` ids
                self.sync_status.setText(f"Found {total_ids} total matches. Fetching details for {wanted}…")
                # Use the newest first — ids are returned newest-first already
                ids_to_fetch = ids[:wanted]

                def on_details_progress(cur, total):
                    # Map details phase to the last 50% of the bar
                    # 50% + (cur/total)*50%
                    pct = 50 + int((cur / max(1, total)) * 50)
                    set_progress(pct)

                added_m, added_p = ingest_details_for_ids(self.db, self.riot, self.current_puuid,
                                                          ids_to_fetch, progress_cb=on_details_progress)
                msg = f"Sync done. Total matches seen: {total_ids}. New SR matches stored: {added_m}. Participants: {added_p}."
            except Exception as e:
                msg = f"Sync failed: {e}"

            def done():
                self.sync_btn.setEnabled(True)
                self.sync_status.setText(msg)
            QTimer.singleShot(0, done)

        threading.Thread(target=worker, daemon=True).start()

# ---------------------------
# MAIN
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
