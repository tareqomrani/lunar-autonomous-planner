
import math
import json
import heapq
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Optional LLM client state
# -----------------------------
OPENAI_CLIENT = None
OPENAI_ERROR = None


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Planetary Autonomous Landing Zone Assessment",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #050914;
        --panel: #0b1220;
        --panel2: #101a2e;
        --border: #20324d;
        --text: #e6f1ff;
        --muted: #8ea4c2;
        --blue: #4db3ff;
        --green: #5cff8d;
        --amber: #ffd166;
        --red: #ff4d4d;
        --purple: #b96dff;
    }

    .stApp {
        background:
            radial-gradient(circle at 20% 10%, rgba(77,179,255,0.13), transparent 28%),
            radial-gradient(circle at 90% 0%, rgba(185,109,255,0.10), transparent 22%),
            linear-gradient(180deg, #030711 0%, #07101f 100%);
        color: var(--text);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #070d18 0%, #0b1424 100%);
        border-right: 1px solid var(--border);
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 900;
        letter-spacing: 0.05em;
        line-height: 1.05;
        margin-bottom: 0.15rem;
        color: #f4f8ff;
        text-shadow: 0 0 22px rgba(77,179,255,0.25);
    }

    .subtitle {
        color: var(--blue);
        font-size: 1.0rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 1.15rem;
    }

    .section-card {
        background: rgba(10, 18, 32, 0.88);
        border: 1px solid rgba(77, 179, 255, 0.22);
        border-radius: 16px;
        padding: 1.0rem 1.05rem;
        box-shadow: 0 0 24px rgba(0,0,0,0.30);
        height: 100%;
    }

    .small-card {
        background: rgba(13, 25, 44, 0.88);
        border: 1px solid rgba(77, 179, 255, 0.18);
        border-radius: 14px;
        padding: 0.85rem;
        height: 100%;
    }

    .card-title {
        color: var(--blue);
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.90rem;
        margin-bottom: 0.75rem;
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .good { color: var(--green); font-weight: 800; }
    .warn { color: var(--amber); font-weight: 800; }
    .bad { color: var(--red); font-weight: 800; }
    .muted { color: var(--muted); }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }

    div[data-testid="stMetric"] {
        background: rgba(12, 23, 40, 0.75);
        border: 1px solid rgba(77,179,255,0.18);
        padding: 0.75rem;
        border-radius: 14px;
    }

    .stButton button, .stDownloadButton button, .stFormSubmitButton button {
        background: linear-gradient(90deg, #102845, #0d395f);
        color: #e6f1ff;
        border: 1px solid rgba(77,179,255,0.45);
        border-radius: 10px;
        font-weight: 800;
    }

    hr {
        border: none;
        border-top: 1px solid rgba(77,179,255,0.18);
        margin: 0.75rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Planetary constants and presets
# -----------------------------
BODY_PRESETS = {
    "Moon": {
        "gravity": 1.62,
        "solar_flux": 1361.0,
        "atmosphere": False,
        "day_length_hr": 708.7,
        "comm_mode": "Earth direct or lunar relay",
        "terrain_note": "Airless regolith with craters, ejecta, boulders, and severe shadow contrast",
        "slope_soft_limit": 5.0,
        "slope_hard_limit": 15.0,
        "safe_hazard_buffer_m": 35.0,
        "thermal_penalty": 0.10,
    },
    "Mars": {
        "gravity": 3.71,
        "solar_flux": 586.0,
        "atmosphere": True,
        "day_length_hr": 24.66,
        "comm_mode": "Orbiter relay preferred",
        "terrain_note": "Dust, rocks, slopes, thermal cycling, and atmospheric entry constraints",
        "slope_soft_limit": 7.0,
        "slope_hard_limit": 18.0,
        "safe_hazard_buffer_m": 45.0,
        "thermal_penalty": 0.18,
    },
    "Phobos": {
        "gravity": 0.0057,
        "solar_flux": 586.0,
        "atmosphere": False,
        "day_length_hr": 7.65,
        "comm_mode": "Mars proximity relay geometry",
        "terrain_note": "Extremely low gravity with irregular shape, weak surface acceleration, and escape-risk operations",
        "slope_soft_limit": 3.0,
        "slope_hard_limit": 10.0,
        "safe_hazard_buffer_m": 60.0,
        "thermal_penalty": 0.16,
    },
    "Europa": {
        "gravity": 1.315,
        "solar_flux": 50.0,
        "atmosphere": False,
        "day_length_hr": 85.2,
        "comm_mode": "Jupiter system relay required",
        "terrain_note": "Ice surface, radiation environment, ridges, chaos terrain, and low solar power availability",
        "slope_soft_limit": 4.0,
        "slope_hard_limit": 14.0,
        "safe_hazard_buffer_m": 50.0,
        "thermal_penalty": 0.28,
    },
    "Custom airless body": {
        "gravity": 1.0,
        "solar_flux": 1000.0,
        "atmosphere": False,
        "day_length_hr": 100.0,
        "comm_mode": "User-defined relay geometry",
        "terrain_note": "Generic airless-body terrain model",
        "slope_soft_limit": 5.0,
        "slope_hard_limit": 15.0,
        "safe_hazard_buffer_m": 40.0,
        "thermal_penalty": 0.15,
    },
}


# -----------------------------
# Optional LLM client helper
# -----------------------------
def get_openai_client():
    """Create the OpenAI client lazily so Streamlit page config stays first."""
    global OPENAI_CLIENT, OPENAI_ERROR

    if OPENAI_CLIENT is not None:
        return OPENAI_CLIENT

    try:
        from openai import OpenAI

        api_key = None
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", None)
        except Exception:
            api_key = None

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in Streamlit secrets or environment variables.")

        OPENAI_CLIENT = OpenAI(api_key=api_key)
        return OPENAI_CLIENT
    except Exception as exc:
        OPENAI_ERROR = str(exc)
        raise


# -----------------------------
# Utility functions
# -----------------------------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_array(arr, low, high):
    return np.maximum(low, np.minimum(high, arr))


def unit_vector_from_az_el(az_deg: float, el_deg: float) -> np.ndarray:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return np.array([
        math.cos(el) * math.sin(az),
        math.cos(el) * math.cos(az),
        math.sin(el),
    ], dtype=float)


def generate_planetary_terrain(seed: int, grid_size: int, roughness: float, crater_count: int, boulder_count: int, body_name: str):
    rng = np.random.default_rng(seed)
    x = np.linspace(-500, 500, grid_size)
    y = np.linspace(-500, 500, grid_size)
    xx, yy = np.meshgrid(x, y)

    body = BODY_PRESETS[body_name]
    gravity_scale = clamp(body["gravity"] / 1.62, 0.05, 2.5)

    z = (
        16 * np.sin(xx / 155)
        + 11 * np.cos(yy / 128)
        + 7 * np.sin((xx + yy) / 95)
        + rng.normal(0, roughness, size=xx.shape)
    )

    if body_name == "Europa":
        z += 7 * np.sin((xx - 0.5 * yy) / 55)
    elif body_name == "Phobos":
        z += 26 * np.sin(xx / 260) + 19 * np.cos(yy / 240)
    elif body_name == "Mars":
        z += 5 * np.sin(xx / 75)

    craters = []
    for i in range(crater_count):
        cx, cy = rng.uniform(-420, 420, 2)
        radius = rng.uniform(28, 95)
        depth = rng.uniform(8, 32)

        if body_name == "Phobos":
            radius *= 1.25
            depth *= 0.85
        elif body_name == "Europa":
            depth *= 0.55
        elif body_name == "Mars":
            depth *= 0.75

        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        bowl = -depth * np.exp(-(dist / radius) ** 2)
        rim = 0.33 * depth * np.exp(-((dist - radius) / (radius * 0.22)) ** 2)
        z += bowl + rim
        craters.append(
            {
                "id": f"C-{i + 1:03d}",
                "x": float(cx),
                "y": float(cy),
                "radius": float(radius),
                "depth": float(depth),
            }
        )

    boulders = []
    boulder_height_scale = clamp(1.15 / max(gravity_scale, 0.25), 0.6, 2.1)
    for i in range(boulder_count):
        bx, by = rng.uniform(-450, 450, 2)
        radius = rng.uniform(8, 30)
        height = rng.uniform(4, 18) * boulder_height_scale
        if body_name == "Europa":
            radius *= 1.2
            height *= 0.85

        dist = np.sqrt((xx - bx) ** 2 + (yy - by) ** 2)
        z += height * np.exp(-(dist / radius) ** 2)
        boulders.append(
            {
                "id": f"B-{i + 1:03d}",
                "x": float(bx),
                "y": float(by),
                "radius": float(radius),
                "height": float(height),
            }
        )

    return x, y, xx, yy, z, craters, boulders


def slope_degrees(z: np.ndarray, spacing: float) -> np.ndarray:
    gy, gx = np.gradient(z, spacing, spacing)
    return np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))


def surface_normals(z: np.ndarray, spacing: float):
    gy, gx = np.gradient(z, spacing, spacing)
    nx = -gx
    ny = -gy
    nz = np.ones_like(z)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    return nx / norm, ny / norm, nz / norm


def illumination_model(x, y, z, spacing, sun_azimuth_deg, sun_elevation_deg, solar_flux):
    sun_vec = unit_vector_from_az_el(sun_azimuth_deg, sun_elevation_deg)
    nx, ny, nz = surface_normals(z, spacing)

    incidence = clamp_array(nx * sun_vec[0] + ny * sun_vec[1] + nz * sun_vec[2], 0, 1)

    shadow = np.zeros_like(z, dtype=bool)
    if sun_vec[2] <= 0:
        shadow[:, :] = True
    else:
        max_steps = min(28, z.shape[0] // 2)
        step_m = spacing
        sx = sun_vec[0]
        sy = sun_vec[1]
        sz = max(sun_vec[2], 1e-6)

        nrows, ncols = z.shape
        for k in range(1, max_steps + 1):
            dx_m = -sx * step_m * k
            dy_m = -sy * step_m * k
            dz_los = sz * step_m * k

            col_shift = int(round(dx_m / spacing))
            row_shift = int(round(dy_m / spacing))

            shifted = np.full_like(z, -1e9)
            src_r0 = max(0, -row_shift)
            src_r1 = min(nrows, nrows - row_shift)
            src_c0 = max(0, -col_shift)
            src_c1 = min(ncols, ncols - col_shift)

            dst_r0 = src_r0 + row_shift
            dst_r1 = src_r1 + row_shift
            dst_c0 = src_c0 + col_shift
            dst_c1 = src_c1 + col_shift

            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = z[src_r0:src_r1, src_c0:src_c1]
            shadow |= shifted > (z + dz_los)

    flux_norm = clamp(solar_flux / 1361.0, 0.0, 1.0)
    solar_score = incidence * flux_norm
    solar_score = np.where(shadow, solar_score * 0.08, solar_score)
    return clamp_array(solar_score, 0, 1), shadow, incidence


def line_of_sight_score(x, y, z, xx, yy, relay_azimuth_deg, relay_elevation_deg, body_name):
    relay_vec = unit_vector_from_az_el(relay_azimuth_deg, relay_elevation_deg)

    if relay_vec[2] <= 0:
        return np.zeros_like(z), np.ones_like(z, dtype=bool)

    spacing = abs(x[1] - x[0])
    blocked = np.zeros_like(z, dtype=bool)
    max_steps = min(30, z.shape[0] // 2)
    sx = relay_vec[0]
    sy = relay_vec[1]
    sz = max(relay_vec[2], 1e-6)

    nrows, ncols = z.shape
    for k in range(1, max_steps + 1):
        dx_m = -sx * spacing * k
        dy_m = -sy * spacing * k
        dz_los = sz * spacing * k

        col_shift = int(round(dx_m / spacing))
        row_shift = int(round(dy_m / spacing))

        shifted = np.full_like(z, -1e9)
        src_r0 = max(0, -row_shift)
        src_r1 = min(nrows, nrows - row_shift)
        src_c0 = max(0, -col_shift)
        src_c1 = min(ncols, ncols - col_shift)

        dst_r0 = src_r0 + row_shift
        dst_r1 = src_r1 + row_shift
        dst_c0 = src_c0 + col_shift
        dst_c1 = src_c1 + col_shift

        shifted[dst_r0:dst_r1, dst_c0:dst_c1] = z[src_r0:src_r1, src_c0:src_c1]
        blocked |= shifted > (z + dz_los)

    elevation_gain = clamp_array((z - z.min()) / max(1e-6, z.max() - z.min()), 0, 1)
    base = clamp(math.sin(math.radians(relay_elevation_deg)), 0, 1)
    deep_space_penalty = 0.55 if body_name == "Europa" else 0.75 if body_name == "Phobos" else 0.90
    comm_score = (0.55 * base + 0.45 * elevation_gain) * deep_space_penalty
    comm_score = np.where(blocked, comm_score * 0.15, comm_score)
    return clamp_array(comm_score, 0, 1), blocked


def radial_penalty(xx, yy, items, kind, safe_buffer_m):
    penalty = np.zeros_like(xx, dtype=float)
    for item in items:
        if kind == "crater":
            sigma = item["radius"] * 1.35 + safe_buffer_m
            weight = clamp(item["depth"] / 32, 0.2, 1.0)
        else:
            sigma = item["radius"] * 2.5 + safe_buffer_m
            weight = clamp(item["height"] / 18, 0.2, 1.0)

        dist = np.sqrt((xx - item["x"]) ** 2 + (yy - item["y"]) ** 2)
        penalty += weight * np.exp(-(dist / sigma) ** 2)

    return clamp_array(penalty, 0, 1)


def landing_suitability(
    xx,
    yy,
    z,
    slope,
    craters,
    boulders,
    solar_score,
    comm_score,
    body_name,
    solar_weight,
    comm_weight,
    hazard_weight,
):
    body = BODY_PRESETS[body_name]
    soft = body["slope_soft_limit"]
    hard = body["slope_hard_limit"]
    safe_buffer = body["safe_hazard_buffer_m"]

    slope_score = clamp_array(1 - (slope - soft) / max(1e-6, hard - soft), 0, 1)
    crater_penalty = radial_penalty(xx, yy, craters, "crater", safe_buffer)
    boulder_penalty = radial_penalty(xx, yy, boulders, "boulder", safe_buffer)
    hazard_score = clamp_array(1 - (0.58 * crater_penalty + 0.42 * boulder_penalty), 0, 1)

    thermal_score = 1.0 - body["thermal_penalty"]
    if body_name == "Europa":
        thermal_score *= 0.82
    if body_name == "Phobos":
        thermal_score *= 0.92

    thermal_map = np.full_like(z, thermal_score, dtype=float)

    score = (
        hazard_weight * hazard_score
        + 0.28 * slope_score
        + solar_weight * solar_score
        + comm_weight * comm_score
        + 0.10 * thermal_map
    )

    score = score / (hazard_weight + 0.28 + solar_weight + comm_weight + 0.10)
    return clamp_array(score * 100, 0, 100), hazard_score, slope_score, thermal_map


def nearest_grid_index(x, y, target_x, target_y):
    ix = int(np.argmin(np.abs(x - target_x)))
    iy = int(np.argmin(np.abs(y - target_y)))
    return iy, ix


def risk_band(score):
    if score >= 85:
        return "LOW", "good"
    if score >= 70:
        return "MODERATE", "warn"
    return "HIGH", "bad"


def hazard_cost_map(suitability, slope, shadow, comm_blocked, body_name):
    body = BODY_PRESETS[body_name]
    cost = 1.0 + (100.0 - suitability) / 22.0
    cost += clamp_array((slope - body["slope_soft_limit"]) / max(1e-6, body["slope_hard_limit"]), 0, 3)
    cost += np.where(shadow, 0.45 if body_name != "Europa" else 0.15, 0.0)
    cost += np.where(comm_blocked, 0.55, 0.0)
    return np.asarray(cost, dtype=float)


def astar_path(cost_map, start_idx, goal_idx):
    rows, cols = cost_map.shape
    sr, sc = start_idx
    gr, gc = goal_idx

    def h(r, c):
        return math.hypot(r - gr, c - gc)

    open_set = [(h(sr, sc), 0.0, sr, sc)]
    came_from = {}
    g_score = {(sr, sc): 0.0}
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        _, g, r, c = heapq.heappop(open_set)
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            return list(reversed(path))

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue

            step = math.sqrt(2) if dr != 0 and dc != 0 else 1.0
            tentative = g + step * float(cost_map[nr, nc])
            if tentative < g_score.get((nr, nc), float("inf")):
                came_from[(nr, nc)] = (r, c)
                g_score[(nr, nc)] = tentative
                heapq.heappush(open_set, (tentative + h(nr, nc), tentative, nr, nc))

    return [start_idx, goal_idx]


def rover_path_hazard_aware(x, y, best_x, best_y, radius, legs, cost_map):
    theta = np.linspace(0, 2 * np.pi, legs, endpoint=False)
    goal_points = []
    for i, t in enumerate(theta):
        r = radius * (0.65 + 0.35 * ((i % 3) / 2))
        px = clamp(best_x + r * np.cos(t), -480, 480)
        py = clamp(best_y + r * np.sin(t), -480, 480)
        goal_points.append((float(px), float(py)))

    goal_points.insert(0, (float(best_x), float(best_y)))
    goal_points.append((float(best_x), float(best_y)))

    full_points = []
    for i in range(len(goal_points) - 1):
        start_idx = nearest_grid_index(x, y, goal_points[i][0], goal_points[i][1])
        goal_idx = nearest_grid_index(x, y, goal_points[i + 1][0], goal_points[i + 1][1])
        idx_path = astar_path(cost_map, start_idx, goal_idx)

        segment = [(float(x[c]), float(y[r])) for r, c in idx_path]
        if i > 0 and segment:
            segment = segment[1:]
        full_points.extend(segment)

    return full_points


def path_distance(points):
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def build_terrain_figure(x, y, z, suitability, craters, boulders, best_x, best_y):
    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=suitability,
            colorscale=[
                [0.0, "#3b0000"],
                [0.25, "#9d1d1d"],
                [0.50, "#c58922"],
                [0.72, "#84b547"],
                [1.0, "#38e67a"],
            ],
            colorbar=dict(title="LZ Score", thickness=12, len=0.70),
            opacity=0.96,
            showscale=True,
            name="Landing Suitability Surface",
            hovertemplate=(
                "Landing Suitability<br>"
                "X: %{x:.1f} m<br>"
                "Y: %{y:.1f} m<br>"
                "Elevation: %{z:.1f} m<br>"
                "LZ Score: %{surfacecolor:.1f}/100"
                "<extra></extra>"
            ),
        )
    )

    crater_x, crater_y, crater_z, crater_text = [], [], [], []
    for c in craters:
        iy, ix = nearest_grid_index(x, y, c["x"], c["y"])
        crater_x.append(c["x"])
        crater_y.append(c["y"])
        crater_z.append(z[iy, ix] + 8)
        crater_text.append(
            f"Crater Hazard<br>"
            f"Object ID: {c['id']}<br>"
            f"Radius: {c['radius']:.1f} m<br>"
            f"Depth: {c['depth']:.1f} m"
        )

    fig.add_trace(
        go.Scatter3d(
            x=crater_x,
            y=crater_y,
            z=crater_z,
            mode="markers",
            marker=dict(size=4, symbol="circle", color="#ff4d4d"),
            text=crater_text,
            hovertemplate="%{text}<extra></extra>",
            name="Crater Hazards",
        )
    )

    boulder_x, boulder_y, boulder_z, boulder_text = [], [], [], []
    for b in boulders:
        iy, ix = nearest_grid_index(x, y, b["x"], b["y"])
        boulder_x.append(b["x"])
        boulder_y.append(b["y"])
        boulder_z.append(z[iy, ix] + 10)
        boulder_text.append(
            f"Boulder Hazard<br>"
            f"Object ID: {b['id']}<br>"
            f"Radius: {b['radius']:.1f} m<br>"
            f"Height: {b['height']:.1f} m"
        )

    fig.add_trace(
        go.Scatter3d(
            x=boulder_x,
            y=boulder_y,
            z=boulder_z,
            mode="markers",
            marker=dict(size=3, symbol="diamond", color="#ffb000"),
            text=boulder_text,
            hovertemplate="%{text}<extra></extra>",
            name="Boulder Hazards",
        )
    )

    iy, ix = nearest_grid_index(x, y, best_x, best_y)
    fig.add_trace(
        go.Scatter3d(
            x=[best_x],
            y=[best_y],
            z=[z[iy, ix] + 30],
            mode="markers+text",
            marker=dict(size=9, symbol="diamond", color="#5cff8d"),
            text=["BEST LZ"],
            textposition="top center",
            hovertemplate=(
                "Recommended Landing Zone<br>"
                f"X: {best_x:.1f} m<br>"
                f"Y: {best_y:.1f} m"
                "<extra></extra>"
            ),
            name="Recommended LZ",
        )
    )

    fig.update_layout(
        height=610,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6f1ff"),
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="East-West, m", gridcolor="#20324d", showbackground=False),
            yaxis=dict(title="North-South, m", gridcolor="#20324d", showbackground=False),
            zaxis=dict(title="Relative Elevation, m", gridcolor="#20324d", showbackground=False),
            camera=dict(eye=dict(x=1.6, y=-1.7, z=0.85)),
            aspectratio=dict(x=1.35, y=1.35, z=0.35),
        ),
        legend=dict(
            bgcolor="rgba(8,14,24,0.70)",
            bordercolor="#20324d",
            borderwidth=1,
            orientation="h",
            y=0.01,
        ),
    )
    return fig


def build_heatmap(title, x, y, values, colorscale, best_x=None, best_y=None, value_fmt=".2f"):
    fig = go.Figure(
        go.Heatmap(
            x=x,
            y=y,
            z=values,
            colorscale=colorscale,
            colorbar=dict(thickness=10),
            hovertemplate=(
                f"{title}<br>"
                "X: %{x:.1f} m<br>"
                "Y: %{y:.1f} m<br>"
                f"Value: %{{z:{value_fmt}}}"
                "<extra></extra>"
            ),
            name=title,
        )
    )

    if best_x is not None:
        fig.add_trace(
            go.Scatter(
                x=[best_x],
                y=[best_y],
                mode="markers+text",
                marker=dict(size=14, symbol="x", color="#ffffff"),
                text=["LZ"],
                textposition="top center",
                hovertemplate="Recommended LZ<extra></extra>",
                name="Recommended LZ",
            )
        )

    fig.update_layout(
        title=title,
        height=305,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6f1ff"),
        xaxis=dict(gridcolor="#20324d"),
        yaxis=dict(gridcolor="#20324d", scaleanchor="x", scaleratio=1),
    )
    return fig


def run_assessment(params):
    body_name = params["target_body"]
    body = BODY_PRESETS[body_name]

    x, y, xx, yy, z, craters, boulders = generate_planetary_terrain(
        int(params["seed"]),
        int(params["grid_size"]),
        float(params["roughness"]),
        int(params["crater_count"]),
        int(params["boulder_count"]),
        body_name,
    )

    spacing = abs(x[1] - x[0])
    slope = slope_degrees(z, spacing)

    solar_score, shadow_map, incidence_map = illumination_model(
        x,
        y,
        z,
        spacing,
        float(params["sun_azimuth"]),
        float(params["sun_elevation"]),
        body["solar_flux"],
    )

    comm_score, comm_blocked = line_of_sight_score(
        x,
        y,
        z,
        xx,
        yy,
        float(params["relay_azimuth"]),
        float(params["relay_elevation"]),
        body_name,
    )

    suitability, hazard_score, slope_score, thermal_score = landing_suitability(
        xx,
        yy,
        z,
        slope,
        craters,
        boulders,
        solar_score,
        comm_score,
        body_name,
        float(params["solar_weight"]),
        float(params["comm_weight"]),
        float(params["hazard_weight"]),
    )

    best_idx = np.unravel_index(np.argmax(suitability), suitability.shape)
    best_y = float(yy[best_idx])
    best_x = float(xx[best_idx])
    best_score = float(suitability[best_idx])
    best_slope = float(slope[best_idx])
    best_solar = float(solar_score[best_idx])
    best_comm = float(comm_score[best_idx])
    best_hazard = float(hazard_score[best_idx])
    best_thermal = float(thermal_score[best_idx])

    cost_map = hazard_cost_map(suitability, slope, shadow_map, comm_blocked, body_name)
    path_points = rover_path_hazard_aware(
        x,
        y,
        best_x,
        best_y,
        float(params["route_radius"]),
        int(params["route_legs"]),
        cost_map,
    )

    total_path_m = path_distance(path_points)
    mobility_factor = clamp(body["gravity"] / 1.62, 0.18, 2.4)
    effective_speed = max(float(params["rover_speed"]) / (0.75 + 0.25 * mobility_factor), 0.01)
    survey_time_hr = total_path_m / effective_speed / 3600

    crater_risks = sum(
        1 for c in craters if math.dist((best_x, best_y), (c["x"], c["y"])) < c["radius"] * 2.2 + body["safe_hazard_buffer_m"]
    )
    boulder_risks = sum(
        1 for b in boulders if math.dist((best_x, best_y), (b["x"], b["y"])) < b["radius"] * 3.0 + body["safe_hazard_buffer_m"]
    )

    total_hazards = crater_risks + boulder_risks
    survey_coverage = clamp((math.pi * float(params["route_radius"]) ** 2) / (1000 * 1000) * 100, 0, 100)
    risk, risk_class = risk_band(best_score)

    return {
        "x": x,
        "y": y,
        "xx": xx,
        "yy": yy,
        "z": z,
        "craters": craters,
        "boulders": boulders,
        "slope": slope,
        "suitability": suitability,
        "solar_score": solar_score,
        "comm_score": comm_score,
        "hazard_score": hazard_score,
        "slope_score": slope_score,
        "thermal_score": thermal_score,
        "shadow_map": shadow_map,
        "incidence_map": incidence_map,
        "comm_blocked": comm_blocked,
        "cost_map": cost_map,
        "best_x": best_x,
        "best_y": best_y,
        "best_score": best_score,
        "best_slope": best_slope,
        "best_solar": best_solar,
        "best_comm": best_comm,
        "best_hazard": best_hazard,
        "best_thermal": best_thermal,
        "path_points": path_points,
        "total_path_m": total_path_m,
        "effective_speed": effective_speed,
        "survey_time_hr": survey_time_hr,
        "total_hazards": total_hazards,
        "survey_coverage": survey_coverage,
        "risk": risk,
        "risk_class": risk_class,
    }



# -----------------------------
# Optional LLM mission reasoning layer
# -----------------------------
def build_llm_hazard_packet(active_params, active_body, results, summary=None):
    """Create a compact, non-sensitive engineering packet for LLM assessment."""
    packet = {
        "mission_id": active_params.get("mission_id"),
        "target_body": active_params.get("target_body"),
        "mission_profile": active_params.get("mission_profile"),
        "body_model": {
            "gravity_mps2": active_body["gravity"],
            "solar_flux_w_m2": active_body["solar_flux"],
            "atmosphere": active_body["atmosphere"],
            "comm_mode": active_body["comm_mode"],
            "slope_soft_limit_deg": active_body["slope_soft_limit"],
            "slope_hard_limit_deg": active_body["slope_hard_limit"],
            "safe_hazard_buffer_m": active_body["safe_hazard_buffer_m"],
            "terrain_note": active_body["terrain_note"],
        },
        "geometry_inputs": {
            "sun_azimuth_deg": active_params.get("sun_azimuth"),
            "sun_elevation_deg": active_params.get("sun_elevation"),
            "relay_azimuth_deg": active_params.get("relay_azimuth"),
            "relay_elevation_deg": active_params.get("relay_elevation"),
        },
        "recommended_landing_zone": {
            "x_m": round(float(results["best_x"]), 2),
            "y_m": round(float(results["best_y"]), 2),
            "score_0_100": round(float(results["best_score"]), 1),
            "slope_deg": round(float(results["best_slope"]), 2),
            "illumination_index_0_1": round(float(results["best_solar"]), 3),
            "communication_index_0_1": round(float(results["best_comm"]), 3),
            "hazard_clearance_index_0_1": round(float(results["best_hazard"]), 3),
            "thermal_index_0_1": round(float(results["best_thermal"]), 3),
        },
        "map_statistics": {
            "craters_modeled": len(results["craters"]),
            "boulders_modeled": len(results["boulders"]),
            "local_buffered_hazards_near_lz": int(results["total_hazards"]),
            "terrain_shadow_fraction": round(float(np.mean(results["shadow_map"])), 3),
            "comm_blocked_fraction": round(float(np.mean(results["comm_blocked"])), 3),
        },
        "rover_plan": {
            "survey_path_km": round(float(results["total_path_m"] / 1000), 3),
            "effective_speed_mps": round(float(results["effective_speed"]), 3),
            "estimated_survey_time_hr": round(float(results["survey_time_hr"]), 3),
            "survey_coverage_percent": round(float(results["survey_coverage"]), 2),
        },
        "engine_status": "Physics engine remains authoritative. LLM output is advisory reasoning only.",
    }
    if summary:
        packet["baseline_recommendations"] = summary.get("recommendations", [])
    return packet


def deterministic_hazard_brief(active_params, active_body, results):
    """Fallback advisory text when no LLM API key is available."""
    concerns = []
    if results["best_slope"] > active_body["slope_soft_limit"]:
        concerns.append("LZ slope is above the soft limit and should be treated as a mobility and landing-stability concern.")
    if results["best_solar"] < 0.35:
        concerns.append("Illumination is weak at the selected point, so solar power and thermal margins need additional validation.")
    if results["best_comm"] < 0.35:
        concerns.append("Communication line-of-sight is marginal and relay placement should be prioritized before surface operations.")
    if results["total_hazards"] > 0:
        concerns.append("Buffered crater or boulder hazards remain near the LZ, so final descent imagery should verify clearance.")
    if not concerns:
        concerns.append("No dominant single-point hazard was identified by the current physics-informed scoring model.")

    return "\n".join([
        "**LLM advisory unavailable. Deterministic autonomy brief generated instead.**",
        "",
        "**Hazard assessment:** " + " ".join(concerns),
        "",
        "**Avoidance strategy:** Maintain the hazard-aware rover route, verify local slopes with descent or rover imagery, and avoid committing habitat placement until illumination and relay geometry are confirmed over the relevant mission window.",
        "",
        "**AI claim boundary:** This fallback uses rule-based autonomy logic, not generative reasoning.",
    ])


def call_llm_hazard_advisor(packet, model_name, temperature=0.2):
    """Call OpenAI as an optional mission reasoning layer. Returns advisory markdown."""
    client = get_openai_client()

    system_text = (
        "You are an aerospace mission autonomy analyst. Review the provided planetary landing-zone "
        "assessment packet. Do not invent sensor data, flight certification, or real terrain validation. "
        "Treat the numerical physics engine as authoritative. Provide concise advisory reasoning for hazard "
        "avoidance, sensor tasking, rover path risk, and site verification. Use professional aerospace language."
    )
    user_text = (
        "Analyze this assessment packet and return exactly four sections: "
        "1) Top Hazards, 2) Avoidance Actions, 3) Sensor Tasking, 4) Confidence and Limitations.\n\n"
        + json.dumps(packet, indent=2)
    )

    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
        temperature=temperature,
    )
    return response.output_text

# -----------------------------
# Sidebar mission controls
# -----------------------------
with st.sidebar:
    with st.form("mission_controls_form", clear_on_submit=False):
        st.markdown("## 🌙 Mission Controls")

        mission_id = st.text_input("Mission ID", "LZAP-2026-001")
        target_body = st.selectbox(
            "Target Body",
            ["Moon", "Mars", "Phobos", "Europa", "Custom airless body"],
            index=0,
        )
        mission_profile = st.selectbox(
            "Mission Profile",
            [
                "Landing Zone + Habitat Preparation",
                "Landing Zone Survey Only",
                "Habitat Site Planning Only",
                "Communications Relay Survey",
            ],
            index=0,
        )

        body = BODY_PRESETS[target_body]

        st.markdown("---")
        st.markdown("### Terrain Model")
        seed = st.number_input("Simulation Seed", min_value=1, max_value=9999, value=42, step=1)
        grid_size = st.slider("Map Resolution", 40, 110, 72, 2)
        roughness = st.slider("Surface Roughness", 0.5, 8.0, 3.0, 0.1)
        crater_count = st.slider("Crater Count", 3, 35, 12)
        boulder_count = st.slider("Boulder Count", 5, 80, 28)

        st.markdown("---")
        st.markdown("### Illumination and Communications")
        sun_azimuth = st.slider("Sun Azimuth, deg", 0, 359, 125)
        sun_elevation = st.slider("Sun Elevation, deg", 1, 80, 18)
        relay_azimuth = st.slider("Relay or Earth Azimuth, deg", 0, 359, 210)
        relay_elevation = st.slider("Relay or Earth Elevation, deg", 1, 85, 24)

        st.caption(
            f"{target_body}: gravity {body['gravity']} m/s², solar flux about {body['solar_flux']} W/m², "
            f"comm mode: {body['comm_mode']}."
        )

        st.markdown("---")
        st.markdown("### Scoring Weights")
        hazard_weight = st.slider("Hazard Avoidance Weight", 0.30, 0.70, 0.50, 0.01)
        solar_weight = st.slider("Illumination Weight", 0.05, 0.30, 0.14, 0.01)
        comm_weight = st.slider("Communication Weight", 0.05, 0.30, 0.10, 0.01)

        st.markdown("---")
        st.markdown("### Rover Planning")
        rover_speed = st.slider("Nominal Rover Speed, m/s", 0.05, 1.50, 0.35, 0.05)
        route_radius = st.slider("Survey Radius, m", 80, 420, 260, 10)
        route_legs = st.slider("Survey Waypoints", 4, 16, 9)

        st.markdown("---")
        st.markdown("### Optional LLM Advisory")
        use_llm = st.checkbox("Enable LLM hazard advisor", value=False)
        llm_model = st.selectbox("LLM Model", ["gpt-5.5", "gpt-5.5-mini", "gpt-4.1-mini"], index=1)
        st.caption("Requires OPENAI_API_KEY in Streamlit secrets or environment variables. The physics engine remains authoritative.")

        submitted = st.form_submit_button("Run Autonomous Assessment", use_container_width=True)

    params = {
        "mission_id": mission_id,
        "target_body": target_body,
        "mission_profile": mission_profile,
        "seed": seed,
        "grid_size": grid_size,
        "roughness": roughness,
        "crater_count": crater_count,
        "boulder_count": boulder_count,
        "sun_azimuth": sun_azimuth,
        "sun_elevation": sun_elevation,
        "relay_azimuth": relay_azimuth,
        "relay_elevation": relay_elevation,
        "hazard_weight": hazard_weight,
        "solar_weight": solar_weight,
        "comm_weight": comm_weight,
        "rover_speed": rover_speed,
        "route_radius": route_radius,
        "route_legs": route_legs,
        "use_llm": use_llm,
        "llm_model": llm_model,
    }

    if submitted:
        st.session_state["assessment_results"] = run_assessment(params)
        st.session_state["assessment_params"] = params.copy()
        st.session_state["assessment_timestamp"] = datetime.now(timezone.utc).isoformat()


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="main-title">PLANETARY AUTONOMOUS LANDING ZONE ASSESSMENT</div>
    <div class="subtitle">Physics-informed survey, hazard scoring, illumination analysis, and habitat planning</div>
    """,
    unsafe_allow_html=True,
)


if "assessment_results" not in st.session_state:
    st.info("Configure mission parameters in the sidebar, then click **Run Autonomous Assessment**.")
    st.stop()


results = st.session_state["assessment_results"]
active_params = st.session_state["assessment_params"]
active_body = BODY_PRESETS[active_params["target_body"]]

x = results["x"]
y = results["y"]
z = results["z"]
craters = results["craters"]
boulders = results["boulders"]
slope = results["slope"]
suitability = results["suitability"]
solar_score = results["solar_score"]
comm_score = results["comm_score"]
hazard_score = results["hazard_score"]
shadow_map = results["shadow_map"]
comm_blocked = results["comm_blocked"]
best_x = results["best_x"]
best_y = results["best_y"]
best_score = results["best_score"]
best_slope = results["best_slope"]
best_solar = results["best_solar"]
best_comm = results["best_comm"]
best_hazard = results["best_hazard"]
best_thermal = results["best_thermal"]
path_points = results["path_points"]
total_path_m = results["total_path_m"]
effective_speed = results["effective_speed"]
survey_time_hr = results["survey_time_hr"]
total_hazards = results["total_hazards"]
survey_coverage = results["survey_coverage"]
risk = results["risk"]
risk_class = results["risk_class"]


# -----------------------------
# Top mission metrics
# -----------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Landing Zone Score", f"{best_score:.0f}/100", risk)
m2.metric("Slope at LZ", f"{best_slope:.1f}°", f"Soft limit {active_body['slope_soft_limit']:.0f}°")
m3.metric("Hazards Near LZ", f"{total_hazards}", "Buffered count")
m4.metric("Illumination Index", f"{best_solar * 100:.0f}%", "Sun plus shadow")
m5.metric("Comm Index", f"{best_comm * 100:.0f}%", "LOS masked")


# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns([3.6, 1.25], gap="large")

with left:
    st.markdown(
        '<div class="section-card"><div class="card-title">3D Terrain Map and Landing Zone Analysis</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        build_terrain_figure(x, y, z, suitability, craters, boulders, best_x, best_y),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-card"><div class="card-title">Mission Overview</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <span class="metric-label">Mission ID</span><br>
        <span class="mono">{active_params["mission_id"]}</span><br><br>
        <span class="metric-label">Target Body</span><br>
        <b>{active_params["target_body"]}</b><br><br>
        <span class="metric-label">Profile</span><br>
        <b>{active_params["mission_profile"]}</b><br><br>
        <span class="metric-label">Body Model</span><br>
        <span class="mono">g = {active_body["gravity"]} m/s²</span><br>
        <span class="mono">Flux = {active_body["solar_flux"]:.0f} W/m²</span><br><br>
        <span class="metric-label">Assessment</span><br>
        <span class="{risk_class}">{risk} RISK</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">Autonomous Hazard Scoring</div>', unsafe_allow_html=True)
    st.write(f"🔴 Craters modeled: **{len(craters)}**")
    st.write(f"🟠 Boulder hazards modeled: **{len(boulders)}**")
    st.write(f"🟡 Buffered hazards near LZ: **{total_hazards}**")
    st.write(f"🟢 Hazard clearance score: **{best_hazard * 100:.0f}%**")
    st.write(f"🔵 Terrain shadow fraction: **{np.mean(shadow_map) * 100:.0f}%**")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Secondary analysis panels
# -----------------------------
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown('<div class="small-card"><div class="card-title">Slope Hazard Map</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_heatmap("Slope, degrees", x, y, slope, "Inferno", best_x, best_y, ".1f"),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="small-card"><div class="card-title">Physics-Based Illumination</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_heatmap("Illumination Score", x, y, solar_score, "YlOrBr", best_x, best_y, ".2f"),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="small-card"><div class="card-title">Line-of-Sight Communication</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_heatmap("Relay Link Score", x, y, comm_score, "Blues", best_x, best_y, ".2f"),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Habitat and rover path
# -----------------------------
h1, h2 = st.columns([1.55, 1.45], gap="large")

with h1:
    st.markdown(
        '<div class="section-card"><div class="card-title">Habitat Planning Recommendation</div>',
        unsafe_allow_html=True,
    )

    hab_offset = max(115, active_body["safe_hazard_buffer_m"] * 2.5)
    habitat_x = clamp(best_x + hab_offset, -480, 480)
    habitat_y = clamp(best_y + hab_offset * 0.45, -480, 480)
    power_x = clamp(best_x - 120, -480, 480)
    power_y = clamp(best_y + 85, -480, 480)
    comm_x = clamp(best_x + 180, -480, 480)
    comm_y = clamp(best_y - 75, -480, 480)

    plan_df = pd.DataFrame(
        [
            ["Primary Landing Zone", best_x, best_y, "Low aggregate risk from slope, hazards, illumination, and comm masking"],
            ["Habitat Zone", habitat_x, habitat_y, "Offset from touchdown, dust, plume interaction, and landing dispersion"],
            ["Power Zone", power_x, power_y, "Selected for relative illumination and terrain separation"],
            ["Comms Relay", comm_x, comm_y, "Placed to improve local line of sight and relay geometry"],
        ],
        columns=["Element", "X m", "Y m", "Rationale"],
    )

    st.dataframe(plan_df, use_container_width=True, hide_index=True)

    st.markdown(
        f"""
        **Engineering interpretation:** the selected site balances hazard clearance, slope, illumination,
        communication visibility, and thermal risk for **{active_params["target_body"]}**. This is a
        physics-informed conceptual assessment, not a certified flight design or terrain-relative navigation product.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with h2:
    st.markdown('<div class="section-card"><div class="card-title">Mobility-Aware Rover Path Planning</div>', unsafe_allow_html=True)

    px = [p[0] for p in path_points]
    py = [p[1] for p in path_points]

    path_fig = go.Figure()
    path_fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=suitability,
            colorscale="Viridis",
            colorbar=dict(thickness=10, title="LZ"),
            hovertemplate="LZ Score: %{z:.1f}/100<extra></extra>",
            name="Landing Suitability",
        )
    )
    path_fig.add_trace(
        go.Scatter(
            x=px,
            y=py,
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=6),
            name="Hazard-Aware Rover Path",
            hovertemplate="Rover Path<br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
        )
    )

    path_fig.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6f1ff"),
        xaxis=dict(gridcolor="#20324d"),
        yaxis=dict(gridcolor="#20324d", scaleanchor="x", scaleratio=1),
    )

    st.plotly_chart(path_fig, use_container_width=True)
    st.write(f"Survey path length: **{total_path_m / 1000:.2f} km**")
    st.write(f"Effective rover speed: **{effective_speed:.2f} m/s**")
    st.write(f"Estimated survey time: **{survey_time_hr:.2f} hr**")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Recommendations and export
# -----------------------------
r1, r2 = st.columns([1.6, 1.0], gap="large")

recommendations = [
    "Select the identified landing zone only after orbital or descent imagery confirms local hazard clearance.",
    "Keep habitat assets outside the touchdown zone to reduce plume, dust, ejecta, and landing-dispersion risk.",
    "Validate illumination against local time, slope aspect, terrain shadowing, and mission duration.",
    "Use LiDAR, stereo EO imagery, thermal sensing, and radar or ground-penetrating radar for final site verification.",
    "Place relay assets where terrain masking is minimized and link geometry remains available during surface operations.",
]

with r1:
    st.markdown('<div class="section-card"><div class="card-title">Mission Recommendations</div>', unsafe_allow_html=True)
    for rec in recommendations:
        st.success(rec)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">LLM Hazard Advisor</div>', unsafe_allow_html=True)

    if active_params.get("use_llm"):
        packet = build_llm_hazard_packet(active_params, active_body, results)
        try:
            with st.spinner("Generating LLM hazard advisory..."):
                llm_advisory = call_llm_hazard_advisor(packet, active_params.get("llm_model", "gpt-5.5-mini"))
            st.info(llm_advisory)
        except Exception as exc:
            st.warning(deterministic_hazard_brief(active_params, active_body, results))
            st.caption(f"LLM advisory failed or is not configured: {exc}")
    else:
        st.caption("Disabled. Enable this to add a generative mission-reasoning layer on top of the physics model.")

    with st.expander("Model limitations and aerospace interpretation"):
        st.write(
            """
            This app now uses body-specific gravity, solar flux, first-order Sun vector illumination,
            terrain shadowing, line-of-sight communication masking, slope scoring, hazard buffering,
            and mobility-aware path planning. It remains a conceptual trade-space tool. It does not replace
            high-fidelity ephemeris analysis, LOLA or LROC terrain products, finite-element lander plume analysis,
            flight software verification, or certified terrain-relative navigation.
            """
        )
    st.markdown("</div>", unsafe_allow_html=True)

with r2:
    st.markdown('<div class="section-card"><div class="card-title">Mission Summary and Export</div>', unsafe_allow_html=True)

    summary = {
        "mission_id": active_params["mission_id"],
        "target_body": active_params["target_body"],
        "mission_profile": active_params["mission_profile"],
        "timestamp_utc": st.session_state.get("assessment_timestamp", datetime.now(timezone.utc).isoformat()),
        "body_model": {
            "gravity_mps2": active_body["gravity"],
            "solar_flux_w_m2": active_body["solar_flux"],
            "atmosphere": active_body["atmosphere"],
            "comm_mode": active_body["comm_mode"],
            "slope_soft_limit_deg": active_body["slope_soft_limit"],
            "slope_hard_limit_deg": active_body["slope_hard_limit"],
        },
        "geometry_inputs": {
            "sun_azimuth_deg": active_params["sun_azimuth"],
            "sun_elevation_deg": active_params["sun_elevation"],
            "relay_azimuth_deg": active_params["relay_azimuth"],
            "relay_elevation_deg": active_params["relay_elevation"],
        },
        "recommended_landing_zone": {"x_m": best_x, "y_m": best_y, "score": best_score},
        "slope_deg": best_slope,
        "illumination_index": best_solar,
        "communication_index": best_comm,
        "hazard_clearance_index": best_hazard,
        "thermal_index": best_thermal,
        "local_hazards": total_hazards,
        "craters_modeled": len(craters),
        "boulders_modeled": len(boulders),
        "terrain_shadow_fraction": float(np.mean(shadow_map)),
        "comm_blocked_fraction": float(np.mean(comm_blocked)),
        "survey_path_km": total_path_m / 1000,
        "effective_rover_speed_mps": effective_speed,
        "estimated_survey_time_hr": survey_time_hr,
        "survey_coverage_percent": survey_coverage,
        "recommendations": recommendations,
        "model_status": "Physics-informed conceptual trade-space tool, not flight-certified analysis",
        "llm_advisory_enabled": bool(active_params.get("use_llm")),
        "llm_model": active_params.get("llm_model"),
        "ai_claim_boundary": "Hazard detection is physics-based. Optional LLM output is advisory mission reasoning, not certified autonomy.",
    }

    st.json(summary, expanded=False)

    st.download_button(
        "Export Mission Report JSON",
        data=json.dumps(summary, indent=2),
        file_name=f"{active_params['mission_id']}_mission_report.json",
        mime="application/json",
        use_container_width=True,
    )

    csv_df = pd.DataFrame(
        [
            {
                "mission_id": active_params["mission_id"],
                "target_body": active_params["target_body"],
                "mission_profile": active_params["mission_profile"],
                "timestamp_utc": summary["timestamp_utc"],
                "gravity_mps2": active_body["gravity"],
                "solar_flux_w_m2": active_body["solar_flux"],
                "sun_azimuth_deg": active_params["sun_azimuth"],
                "sun_elevation_deg": active_params["sun_elevation"],
                "relay_azimuth_deg": active_params["relay_azimuth"],
                "relay_elevation_deg": active_params["relay_elevation"],
                "lz_x_m": best_x,
                "lz_y_m": best_y,
                "lz_score": best_score,
                "slope_deg": best_slope,
                "illumination_index": best_solar,
                "communication_index": best_comm,
                "hazard_clearance_index": best_hazard,
                "local_hazards": total_hazards,
                "craters_modeled": len(craters),
                "boulders_modeled": len(boulders),
                "terrain_shadow_fraction": float(np.mean(shadow_map)),
                "comm_blocked_fraction": float(np.mean(comm_blocked)),
                "survey_path_km": total_path_m / 1000,
                "effective_rover_speed_mps": effective_speed,
                "estimated_survey_time_hr": survey_time_hr,
                "survey_coverage_percent": survey_coverage,
            }
        ]
    )

    st.download_button(
        "Export Mission Summary CSV",
        data=csv_df.to_csv(index=False),
        file_name=f"{active_params['mission_id']}_mission_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Footer
# -----------------------------
st.markdown(
    """
    <br>
    <div class="muted" style="text-align:center; letter-spacing:0.20em; text-transform:uppercase;">
        Autonomy | Illumination | Hazard Scoring | Mobility | Systems Engineering
    </div>
    """,
    unsafe_allow_html=True,
)
