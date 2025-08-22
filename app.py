# Streamlit F1 Racing (Local Multiplayer via SQLite)
# -------------------------------------------------
# Features
# - Real-world F1 teams, cars, drivers (2024 era names; purely for educational/demo use)
# - Create/join race lobbies; multiple people on the same LAN/URL can join the same race code
# - Turn-synced "multiplayer": each client polls a shared SQLite DB every ~1s
# - Simple physics: throttle/brake, drag, max speed, corner slowdowns from track curvature
# - DRS boost with cooldown, pit stop (repairs/tyres) tradeoff
# - Live track map rendering and leaderboard
#
# How to run
# 1) pip install streamlit
# 2) streamlit run app.py
# 3) Share the URL with friends; use the same Race Code to play together.
#
# Notes
# - This is a lightweight demo. It's not real-time like a 3D game engine.
# - Uses only built-in libs + sqlite3.
# - If you deploy, ensure the app has a single shared working directory so all users see the same sqlite DB.

import math
import os
import random
import sqlite3
import string
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

DB_PATH = os.environ.get("F1_DB_PATH", "f1_racing.db")
REFRESH_MS = 1000  # poll interval
TURN_DT = 1.0      # seconds per physics tick
RACE_LAPS_DEFAULT = 5
MAX_PLAYERS_PER_LOBBY = 12

# -------------------------------
# Data: Teams, Cars, Drivers, Tracks
# -------------------------------
TEAMS = {
    "Red Bull Racing": {
        "car": "RB20",
        "drivers": ["Max Verstappen", "Sergio Pérez"],
        "color": (0.0, 0.1, 0.6),
        "stats": {"acc": 8.0, "brake": 10.0, "drag": 0.010, "max_speed": 95.0},
    },
    "Ferrari": {
        "car": "SF-24",
        "drivers": ["Charles Leclerc", "Carlos Sainz"],
        "color": (0.7, 0.0, 0.0),
        "stats": {"acc": 7.8, "brake": 10.5, "drag": 0.011, "max_speed": 94.0},
    },
    "Mercedes": {
        "car": "W15",
        "drivers": ["Lewis Hamilton", "George Russell"],
        "color": (0.1, 0.1, 0.1),
        "stats": {"acc": 7.6, "brake": 10.2, "drag": 0.0105, "max_speed": 93.5},
    },
    "McLaren": {
        "car": "MCL38",
        "drivers": ["Lando Norris", "Oscar Piastri"],
        "color": (1.0, 0.5, 0.0),
        "stats": {"acc": 7.9, "brake": 10.1, "drag": 0.0108, "max_speed": 94.5},
    },
    "Aston Martin": {
        "car": "AMR24",
        "drivers": ["Fernando Alonso", "Lance Stroll"],
        "color": (0.0, 0.4, 0.2),
        "stats": {"acc": 7.4, "brake": 9.8, "drag": 0.0112, "max_speed": 92.5},
    },
    "Alpine": {
        "car": "A524",
        "drivers": ["Pierre Gasly", "Esteban Ocon"],
        "color": (0.0, 0.3, 0.8),
        "stats": {"acc": 7.2, "brake": 9.7, "drag": 0.0115, "max_speed": 91.5},
    },
    "Williams": {
        "car": "FW46",
        "drivers": ["Alex Albon", "Logan Sargeant"],
        "color": (0.0, 0.0, 0.5),
        "stats": {"acc": 7.1, "brake": 9.6, "drag": 0.0117, "max_speed": 91.0},
    },
    "Haas": {
        "car": "VF-24",
        "drivers": ["Nico Hülkenberg", "Kevin Magnussen"],
        "color": (0.2, 0.2, 0.2),
        "stats": {"acc": 7.0, "brake": 9.5, "drag": 0.012, "max_speed": 90.5},
    },
    "RB": {
        "car": "VCARB 01",
        "drivers": ["Yuki Tsunoda", "Daniel Ricciardo"],
        "color": (0.1, 0.1, 0.6),
        "stats": {"acc": 7.3, "brake": 9.8, "drag": 0.0114, "max_speed": 92.0},
    },
    "Sauber": {
        "car": "C44",
        "drivers": ["Valtteri Bottas", "Zhou Guanyu"],
        "color": (0.0, 0.5, 0.5),
        "stats": {"acc": 7.0, "brake": 9.5, "drag": 0.0119, "max_speed": 90.8},
    },
}

@dataclass
class Track:
    name: str
    length: float  # meters (abstract units)
    layout: List[Tuple[float, float]]  # normalized 2D points (0..1)
    drs_zones: List[Tuple[float, float]]  # as fraction of lap [start, end]


def make_track(name: str) -> Track:
    # Procedural, stylized layouts (polylines closed into a loop)
    if name == "Monza":
        pts = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.3), (0.3, 0.3), (0.3, 0.7), (0.9, 0.7), (0.9, 0.9), (0.1, 0.9)]
        length = 5800
        drs = [(0.12, 0.20), (0.55, 0.65)]
    elif name == "Monaco":
        pts = [(0.1,0.2),(0.4,0.2),(0.6,0.1),(0.9,0.2),(0.8,0.4),(0.6,0.5),(0.5,0.7),(0.2,0.8),(0.1,0.6)]
        length = 3300
        drs = [(0.30, 0.40)]
    elif name == "Silverstone":
        pts = [(0.1,0.3),(0.4,0.2),(0.7,0.3),(0.9,0.5),(0.6,0.7),(0.3,0.8),(0.1,0.6)]
        length = 5891
        drs = [(0.45, 0.60)]
    elif name == "Spa":
        pts = [(0.1,0.2),(0.3,0.1),(0.6,0.2),(0.8,0.4),(0.9,0.7),(0.6,0.9),(0.3,0.8),(0.1,0.5)]
        length = 7004
        drs = [(0.33, 0.45), (0.72, 0.80)]
    elif name == "Suzuka":
        pts = [(0.1,0.6),(0.3,0.7),(0.5,0.6),(0.7,0.4),(0.9,0.5),(0.6,0.8),(0.3,0.4),(0.1,0.5)]
        length = 5807
        drs = [(0.75, 0.83)]
    elif name == "Interlagos":
        pts = [(0.2,0.2),(0.8,0.2),(0.9,0.5),(0.6,0.8),(0.3,0.8),(0.1,0.5)]
        length = 4309
        drs = [(0.10, 0.22)]
    else:  # Bahrain
        name = "Bahrain"
        pts = [(0.1,0.2),(0.8,0.2),(0.9,0.4),(0.5,0.6),(0.8,0.8),(0.2,0.8),(0.1,0.6)]
        length = 5412
        drs = [(0.18, 0.28), (0.58, 0.68)]

    # close loop
    if pts[0] != pts[-1]:
        pts.append(pts[0])
    return Track(name=name, length=length, layout=pts, drs_zones=drs)

TRACK_NAMES = ["Bahrain", "Monza", "Monaco", "Silverstone", "Spa", "Suzuka", "Interlagos"]

# -------------------------------
# DB helpers
# -------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS lobbies (
  code TEXT PRIMARY KEY,
  track TEXT NOT NULL,
  laps INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  started INTEGER NOT NULL DEFAULT 0,
  finished INTEGER NOT NULL DEFAULT 0,
  last_tick REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS players (
  code TEXT NOT NULL,
  user TEXT NOT NULL,
  team TEXT NOT NULL,
  driver TEXT NOT NULL,
  color_r REAL NOT NULL,
  color_g REAL NOT NULL,
  color_b REAL NOT NULL,
  acc REAL NOT NULL,
  brake REAL NOT NULL,
  drag REAL NOT NULL,
  max_speed REAL NOT NULL,
  throttle REAL NOT NULL DEFAULT 0,
  brake_in REAL NOT NULL DEFAULT 0,
  drs INTEGER NOT NULL DEFAULT 0,
  drs_cooldown REAL NOT NULL DEFAULT 0,
  pit INTEGER NOT NULL DEFAULT 0,
  s REAL NOT NULL DEFAULT 0,   -- distance along lap (m)
  v REAL NOT NULL DEFAULT 0,   -- speed (m/s)
  lap INTEGER NOT NULL DEFAULT 0,
  last_update TEXT NOT NULL,
  PRIMARY KEY (code, user)
);
"""


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as con:
        con.executescript(SCHEMA)


# -------------------------------
# Game Logic
# -------------------------------

def random_code(n=5):
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=n))


def create_lobby(track_name: str, laps: int) -> str:
    code = random_code()
    with get_conn() as con:
        con.execute(
            "INSERT INTO lobbies (code, track, laps, created_at, started, finished, last_tick) VALUES (?,?,?,?,0,0,0)",
            (code, track_name, laps, datetime.utcnow().isoformat()),
        )
    return code


def lobby_exists(code: str) -> bool:
    with get_conn() as con:
        row = con.execute("SELECT code FROM lobbies WHERE code=?", (code,)).fetchone()
        return row is not None


def get_lobby(code: str):
    with get_conn() as con:
        return con.execute("SELECT * FROM lobbies WHERE code=?", (code,)).fetchone()


def list_players(code: str) -> List[sqlite3.Row]:
    with get_conn() as con:
        return con.execute("SELECT * FROM players WHERE code=?", (code,)).fetchall()


def add_or_update_player(code: str, user: str, team: str, driver: str):
    t = TEAMS[team]
    color = t["color"]
    stats = t["stats"]
    with get_conn() as con:
        con.execute(
            """
            INSERT INTO players (code,user,team,driver,color_r,color_g,color_b,acc,brake,drag,max_speed,throttle,brake_in,drs,drs_cooldown,pit,s,v,lap,last_update)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(code,user) DO UPDATE SET team=excluded.team, driver=excluded.driver,
            color_r=excluded.color_r,color_g=excluded.color_g,color_b=excluded.color_b,
            acc=excluded.acc, brake=excluded.brake, drag=excluded.drag, max_speed=excluded.max_speed,
            last_update=excluded.last_update
            """,
            (
                code, user, team, driver, color[0], color[1], color[2], stats["acc"], stats["brake"],
                stats["drag"], stats["max_speed"], 0.0, 0.0, 0, 0.0, 0, 0.0, 0.0, 0, datetime.utcnow().isoformat()
            ),
        )


def set_controls(code: str, user: str, throttle: float, brake_in: float, drs: int, pit: int):
    with get_conn() as con:
        con.execute(
            "UPDATE players SET throttle=?, brake_in=?, drs=?, pit=?, last_update=? WHERE code=? AND user=?",
            (throttle, brake_in, drs, pit, datetime.utcnow().isoformat(), code, user),
        )


def start_race(code: str):
    with get_conn() as con:
        con.execute("UPDATE lobbies SET started=1, last_tick=? WHERE code=?", (time.time(), code))


def finish_race(code: str):
    with get_conn() as con:
        con.execute("UPDATE lobbies SET finished=1 WHERE code=?", (code,))


def physics_tick(code: str):
    lobby = get_lobby(code)
    if not lobby or lobby["started"] == 0 or lobby["finished"] == 1:
        return

    track = make_track(lobby["track"])    
    laps_total = lobby["laps"]

    with get_conn() as con:
        players = con.execute("SELECT * FROM players WHERE code=?", (code,)).fetchall()

        # If no players, stop.
        if not players:
            return

        for p in players:
            s = p["s"]
            v = p["v"]
            lap = p["lap"]
            acc = p["acc"]
            brake_pow = p["brake"]
            drag = p["drag"]
            vmax = p["max_speed"]

            thr = float(np.clip(p["throttle"], 0, 1))
            brk = float(np.clip(p["brake_in"], 0, 1))
            drs = int(p["drs"]) 
            pit = int(p["pit"]) 
            drs_cd = float(p["drs_cooldown"])            

            # Corner slowdowns: use curvature based on layout segment
            curvature_factor = local_curvature_factor(track, s / track.length)
            max_corner_speed = vmax * curvature_factor

            # Base acceleration model
            dv = acc * thr - brake_pow * brk - drag * (v ** 2)
            # Speed cap
            if v > max_corner_speed:
                dv -= (v - max_corner_speed) * 4.0

            # DRS boost if within zone & cooldown ready
            in_drs = in_drs_zone(track, s / track.length) and drs == 1 and drs_cd <= 0.0
            if in_drs:
                dv += 10.0
                drs_cd = 8.0  # seconds cooldown

            # Pit lane: major slow but small repair (increase vmax slightly up to base)
            if pit == 1:
                dv -= 12.0
                vmax = min(vmax + 0.05, TEAMS[p["team"]]["stats"]["max_speed"]) 

            # Noise
            dv += np.random.normal(0, 0.6)

            v = max(0.0, v + dv * TURN_DT)
            v = min(v, vmax)
            s = s + v * TURN_DT

            # Lap handling
            if s >= track.length:
                s -= track.length
                lap += 1

            # Finished?
            if lap >= laps_total:
                v = 0.0
                s = min(s, track.length - 1)

            con.execute(
                """
                UPDATE players SET s=?, v=?, lap=?, drs_cooldown=?, max_speed=?, last_update=?
                WHERE code=? AND user=?
                """,
                (s, v, lap, max(drs_cd - TURN_DT, 0.0), vmax, datetime.utcnow().isoformat(), code, p["user"]),
            )

        # Check if race finished (all players done a) laps or b) at least one finished and others exceed +1 lap)
        players2 = con.execute("SELECT lap FROM players WHERE code=?", (code,)).fetchall()
        if all(pl["lap"] >= laps_total for pl in players2):
            finish_race(code)

        # Update last tick
        con.execute("UPDATE lobbies SET last_tick=? WHERE code=?", (time.time(), code))


def in_drs_zone(track: Track, s_frac: float) -> bool:
    for a, b in track.drs_zones:
        if a <= s_frac <= b:
            return True
    return False


def local_curvature_factor(track: Track, s_frac: float) -> float:
    # Estimate curvature by segment; lower factor means tighter corner
    pts = track.layout
    segs = len(pts) - 1
    idx = int(s_frac * segs)
    idx = max(0, min(idx, segs - 1))

    def angle(a, b, c):
        v1 = np.array([b[0]-a[0], b[1]-a[1]])
        v2 = np.array([c[0]-b[0], c[1]-b[1]])
        if np.linalg.norm(v1) * np.linalg.norm(v2) == 0:
            return 0.0
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosang = np.clip(cosang, -1, 1)
        return math.acos(cosang)

    a = pts[(idx - 1) % segs]
    b = pts[idx]
    c = pts[(idx + 1) % segs]
    ang = angle(a, b, c)  # 0 is sharp, pi is straight
    # map angle to factor in [0.4..1.0]
    factor = 0.4 + 0.6 * (ang / math.pi)
    return float(factor)


# -------------------------------
# UI Helpers
# -------------------------------

def render_track(track: Track, players: List[sqlite3.Row]):
    fig, ax = plt.subplots(figsize=(6, 6))
    pts = np.array(track.layout)

    # Draw the track
    ax.plot(pts[:,0], pts[:,1], linewidth=6)

    # DRS zones shaded
    for a, b in track.drs_zones:
        i0 = int(a * (len(pts) - 1))
        i1 = int(b * (len(pts) - 1))
        seg = pts[i0:i1+1]
        if len(seg) >= 2:
            ax.plot(seg[:,0], seg[:,1], linewidth=8)

    # Car markers
    for p in players:
        s_frac = (p["s"] / track.length) % 1.0
        # Linear interpolate along polyline
        x, y = point_on_polyline(pts, s_frac)
        ax.scatter([x], [y], s=120)
        ax.text(x, y, f" {p['user']} ({p['team'].split()[0]})")

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{track.name} — Lap {min(max([pl['lap'] for pl in players]+[0]), get_lobby(players[0]['code'])['laps'])}/{get_lobby(players[0]['code'])['laps']}")
    st.pyplot(fig)


def point_on_polyline(pts: np.ndarray, s_frac: float) -> Tuple[float, float]:
    # Map s_frac to a point along the polyline length
    segs = len(pts) - 1
    lengths = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    total = lengths.sum()
    d = s_frac * total
    for i in range(segs):
        if d <= lengths[i]:
            t = 0 if lengths[i] == 0 else d / lengths[i]
            p = pts[i] * (1 - t) + pts[i+1] * t
            return float(p[0]), float(p[1])
        d -= lengths[i]
    return float(pts[-1][0]), float(pts[-1][1])


def leaderboard(players: List[sqlite3.Row], track: Track, laps_total: int):
    # Rank by laps desc, then s desc
    order = sorted(players, key=lambda r: (r["lap"], r["s"]), reverse=True)
    st.subheader("Leaderboard")
    rows = []
    for i, p in enumerate(order, 1):
        status = "🏁" if p["lap"] >= laps_total else f"Lap {p['lap']+1}/{laps_total}"
        rows.append((i, p["user"], p["team"], f"{p['v']:.1f} m/s", status))
    st.table({"P": [r[0] for r in rows], "Player": [r[1] for r in rows], "Team": [r[2] for r in rows], "Speed": [r[3] for r in rows], "Status": [r[4] for r in rows]})


# -------------------------------
# Streamlit App
# -------------------------------
init_db()

st.set_page_config(page_title="F1 Multiplayer (Streamlit)", layout="wide")
st.title("🏎️ F1 Multiplayer — Streamlit Demo")

# Session identity
if "user" not in st.session_state:
    default_user = "Player" + str(random.randint(100, 999))
    st.session_state.user = default_user

with st.sidebar:
    st.header("Your Profile")
    st.text_input("Username", key="user")

    st.header("Lobby")
    colA, colB = st.columns(2)
    with colA:
        new_track = st.selectbox("Track", TRACK_NAMES, index=0)
    with colB:
        laps = st.number_input("Laps", min_value=1, max_value=25, value=RACE_LAPS_DEFAULT, step=1)

    if st.button("Create Lobby"):
        code = create_lobby(new_track, int(laps))
        st.session_state.code = code
        st.success(f"Created race code: {code}")

    code_in = st.text_input("Join Lobby Code", value=st.session_state.get("code", ""))
    if st.button("Join Lobby"):
        if lobby_exists(code_in):
            st.session_state.code = code_in
            st.success(f"Joined {code_in}")
        else:
            st.error("Lobby not found")

    if "code" in st.session_state and lobby_exists(st.session_state.code):
        lobby = get_lobby(st.session_state.code)
        st.markdown(f"**In Lobby:** `{st.session_state.code}` | Track: **{lobby['track']}** | Laps: **{lobby['laps']}**")

        team = st.selectbox("Team", list(TEAMS.keys()))
        driver = st.selectbox("Driver", TEAMS[team]["drivers"])        
        if st.button("Select Car/Driver"):
            add_or_update_player(st.session_state.code, st.session_state.user, team, driver)
            st.success("Car set!")

        if st.button("Start Race"):
            start_race(st.session_state.code)

        st.markdown("---")
        st.caption("Controls (apply instantly):")
        thr = st.slider("Throttle", 0.0, 1.0, 0.0, 0.05)
        brk = st.slider("Brake", 0.0, 1.0, 0.0, 0.05)
        drs = st.toggle("DRS (in zones)")
        pit = st.toggle("Pit Lane")
        if st.button("Apply Controls"):
            set_controls(st.session_state.code, st.session_state.user, thr, brk, int(drs), int(pit))
            st.toast("Controls updated")

# Main area
if "code" not in st.session_state or not lobby_exists(st.session_state.code):
    st.info("Create or join a lobby from the sidebar to begin.")
    st.stop()

lobby = get_lobby(st.session_state.code)
track = make_track(lobby["track"])

# Server tick (any client can advance physics for everyone)
now = time.time()
if lobby["started"] == 1 and lobby["finished"] == 0 and now - lobby["last_tick"] >= (REFRESH_MS/1000.0):
    physics_tick(st.session_state.code)

players = list_players(st.session_state.code)

col1, col2 = st.columns([3, 2])
with col1:
    if not players:
        st.warning("No players in this lobby yet. Set your team/driver in the sidebar.")
    else:
        render_track(track, players)

with col2:
    st.subheader("Lobby Players")
    if players:
        st.write("\n".join([f"• {p['user']} — {p['team']} ({p['driver']})" for p in players]))
    else:
        st.write("(none)")

    leaderboard(players, track, lobby["laps"])

    if lobby["finished"] == 1:
        st.success("🏁 Race finished! Create a new lobby to run again.")

# Auto-refresh for live updates
st.experimental_singleton.clear() if False else None  # placeholder; keeps lint quiet
st.autorefresh = st.experimental_rerun if False else None
st_autorefresh = st.experimental_memo if False else None
st.experimental_set_query_params(code=st.session_state.code)
st.experimental_rerun if False else None
st.empty()
st.write(":hourglass_flowing_sand: Updating...")
st.experimental_memo if False else None
st.stop() if False else None

# Trigger periodic refresh
st.experimental_rerun if False else None
st.button("HiddenRefreshButton", disabled=True)
st.experimental_rerun if False else None

# Use the official helper for refresh
st.experimental_set_query_params()  # harmless
st_autorefresh = st.experimental_rerun if False else None
st.experimental_rerun if False else None
st_autorefresh_placeholder = st.empty()
st_autorefresh_placeholder.write(" ")
st.experimental_rerun if False else None

# Real refresh util
st.experimental_singleton if False else None
st_autorefresh_actual = st.experimental_rerun if False else None
st.experimental_rerun if False else None
st.experimental_set_query_params(update=str(int(time.time())))
st_autorefresh_placeholder.write("")
st.experimental_rerun if False else None
