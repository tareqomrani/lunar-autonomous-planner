import math
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Lunar Autonomous Landing Zone Assessment",
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
        font-size: 2.65rem;
        font-weight: 900;
        letter-spacing: 0.05em;
        line-height: 1.05;
        margin-bottom: 0.15rem;
        color: #f4f8ff;
        text-shadow: 0 0 22px rgba(77,179,255,0.25);
    }

    .subtitle {
        color: var(--blue);
        font-size: 1.02rem;
        letter-spacing: 0.18em;
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
# Utility functions
# -----------------------------
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def clamp_array(arr, low, high):
    return np.maximum(low, np.minimum(high, arr))


def generate_lunar_terrain(seed: int, grid_size: int, roughness: float, crater_count: int, boulder_count: int):
    rng = np.random.default_rng(seed)
    x = np.linspace(-500, 500, grid_size)
    y = np.linspace(-500, 500, grid_size)
    xx, yy = np.meshgrid(x, y)

    z = (
        18 * np.sin(xx / 155)
        + 12 * np.cos(yy / 128)
        + 7 * np.sin((xx + yy) / 95)
        + rng.normal(0, roughness, size=xx.shape)
    )

    craters = []
    for i in range(crater_count):
        cx, cy = rng.uniform(-420, 420, 2)
        radius = rng.uniform(28, 95)
        depth = rng.uniform(8, 32)
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
    for i in range(boulder_count):
        bx, by = rng.uniform(-450, 450, 2)
        radius = rng.uniform(8, 30)
        height = rng.uniform(4, 18)
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


def radial_penalty(xx, yy, items, kind):
    penalty = np.zeros_like(xx, dtype=float)
    for item in items:
        if kind == "crater":
            sigma = item["radius"] * 1.35
            weight = clamp(item["depth"] / 32, 0.2, 1.0)
        else:
            sigma = item["radius"] * 2.5
            weight = clamp(item["height"] / 18, 0.2, 1.0)

        dist = np.sqrt((xx - item["x"]) ** 2 + (yy - item["y"]) ** 2)
        penalty += weight * np.exp(-(dist / sigma) ** 2)

    return clamp_array(penalty, 0, 1)


def landing_suitability(xx, yy, z, slope, craters, boulders, solar_weight, comm_weight, hazard_weight):
    slope_score = clamp_array(1 - slope / 18, 0, 1)
    crater_penalty = radial_penalty(xx, yy, craters, "crater")
    boulder_penalty = radial_penalty(xx, yy, boulders, "boulder")
    hazard_score = clamp_array(1 - (0.58 * crater_penalty + 0.42 * boulder_penalty), 0, 1)

    z_norm = (z - z.min()) / max(1e-6, (z.max() - z.min()))
    solar_score = clamp_array(0.45 + 0.55 * z_norm + 0.08 * np.sin(xx / 170), 0, 1)
    comm_score = clamp_array(0.60 + 0.25 * np.cos((xx - yy) / 260) + 0.15 * z_norm, 0, 1)

    score = (
        hazard_weight * hazard_score
        + 0.30 * slope_score
        + solar_weight * solar_score
        + comm_weight * comm_score
    )
    score = score / (hazard_weight + 0.30 + solar_weight + comm_weight)

    return clamp_array(score * 100, 0, 100), solar_score, comm_score, hazard_score


def nearest_grid_index(x, y, target_x, target_y):
    ix = int(np.argmin(np.abs(x - target_x)))
    iy = int(np.argmin(np.abs(y - target_y)))
    return iy, ix


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
            zaxis=dict(title="Elevation, m", gridcolor="#20324d", showbackground=False),
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


def build_heatmap(title, x, y, values, colorscale, best_x=None, best_y=None):
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
                "Value: %{z:.2f}"
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


def rover_path(best_x, best_y, radius, legs):
    theta = np.linspace(0, 2 * np.pi, legs, endpoint=False)
    points = []

    for i, t in enumerate(theta):
        r = radius * (0.65 + 0.35 * ((i % 3) / 2))
        px = best_x + r * np.cos(t)
        py = best_y + r * np.sin(t)
        points.append((float(clamp(px, -480, 480)), float(clamp(py, -480, 480))))

    points.insert(0, (float(best_x), float(best_y)))
    points.append((float(best_x), float(best_y)))
    return points


def path_distance(points):
    return sum(math.dist(points[i], points[i + 1]) for i in range(len(points) - 1))


def risk_band(score):
    if score >= 85:
        return "LOW", "good"
    if score >= 70:
        return "MODERATE", "warn"
    return "HIGH", "bad"


def run_assessment(params):
    x, y, xx, yy, z, craters, boulders = generate_lunar_terrain(
        int(params["seed"]),
        int(params["grid_size"]),
        float(params["roughness"]),
        int(params["crater_count"]),
        int(params["boulder_count"]),
    )

    spacing = abs(x[1] - x[0])
    slope = slope_degrees(z, spacing)

    suitability, solar_score, comm_score, hazard_score = landing_suitability(
        xx,
        yy,
        z,
        slope,
        craters,
        boulders,
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

    path_points = rover_path(best_x, best_y, params["route_radius"], params["route_legs"])
    total_path_m = path_distance(path_points)
    survey_time_hr = total_path_m / max(params["rover_speed"], 0.01) / 3600

    crater_risks = sum(
        1 for c in craters if math.dist((best_x, best_y), (c["x"], c["y"])) < c["radius"] * 2.2
    )
    boulder_risks = sum(
        1 for b in boulders if math.dist((best_x, best_y), (b["x"], b["y"])) < b["radius"] * 3.0
    )

    total_hazards = crater_risks + boulder_risks
    survey_coverage = clamp((math.pi * params["route_radius"] ** 2) / (1000 * 1000) * 100, 0, 100)
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
        "best_x": best_x,
        "best_y": best_y,
        "best_score": best_score,
        "best_slope": best_slope,
        "best_solar": best_solar,
        "best_comm": best_comm,
        "best_hazard": best_hazard,
        "path_points": path_points,
        "total_path_m": total_path_m,
        "survey_time_hr": survey_time_hr,
        "total_hazards": total_hazards,
        "survey_coverage": survey_coverage,
        "risk": risk,
        "risk_class": risk_class,
    }


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

        st.markdown("---")
        st.markdown("### Terrain Model")
        seed = st.number_input("Simulation Seed", min_value=1, max_value=9999, value=42, step=1)
        grid_size = st.slider("Map Resolution", 40, 110, 72, 2)
        roughness = st.slider("Regolith Roughness", 0.5, 8.0, 3.0, 0.1)
        crater_count = st.slider("Crater Density", 3, 35, 12)
        boulder_count = st.slider("Boulder Density", 5, 80, 28)

        st.markdown("---")
        st.markdown("### Scoring Weights")
        hazard_weight = st.slider("Hazard Avoidance Weight", 0.30, 0.70, 0.48, 0.01)
        solar_weight = st.slider("Solar Exposure Weight", 0.05, 0.30, 0.14, 0.01)
        comm_weight = st.slider("Communication Coverage Weight", 0.05, 0.30, 0.08, 0.01)

        st.markdown("---")
        st.markdown("### Rover Planning")
        rover_speed = st.slider("Rover Survey Speed, m/s", 0.05, 1.50, 0.35, 0.05)
        route_radius = st.slider("Survey Radius, m", 80, 420, 260, 10)
        route_legs = st.slider("Survey Waypoints", 4, 16, 9)

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
        "hazard_weight": hazard_weight,
        "solar_weight": solar_weight,
        "comm_weight": comm_weight,
        "rover_speed": rover_speed,
        "route_radius": route_radius,
        "route_legs": route_legs,
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
    <div class="main-title">LUNAR AUTONOMOUS LANDING ZONE ASSESSMENT</div>
    <div class="subtitle">AI-powered survey, hazard detection, and habitat planning for future lunar operations</div>
    """,
    unsafe_allow_html=True,
)


if "assessment_results" not in st.session_state:
    st.info("Configure mission parameters in the sidebar, then click **Run Autonomous Assessment**.")
    st.stop()


results = st.session_state["assessment_results"]
active_params = st.session_state["assessment_params"]

x = results["x"]
y = results["y"]
z = results["z"]
craters = results["craters"]
boulders = results["boulders"]
slope = results["slope"]
suitability = results["suitability"]
solar_score = results["solar_score"]
comm_score = results["comm_score"]
best_x = results["best_x"]
best_y = results["best_y"]
best_score = results["best_score"]
best_slope = results["best_slope"]
best_solar = results["best_solar"]
best_comm = results["best_comm"]
best_hazard = results["best_hazard"]
path_points = results["path_points"]
total_path_m = results["total_path_m"]
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
m2.metric("Slope at LZ", f"{best_slope:.1f}°", "Target < 5°")
m3.metric("Hazards Near LZ", f"{total_hazards}", "Crater + boulder")
m4.metric("Solar Index", f"{best_solar * 100:.0f}%", "Power viability")
m5.metric("Comm Index", f"{best_comm * 100:.0f}%", "Relay visibility")


# -----------------------------
# Main layout
# -----------------------------
left, right = st.columns([3.6, 1.25], gap="large")

with left:
    st.markdown(
        '<div class="section-card"><div class="card-title">3D Terrain Map & Landing Zone Analysis</div>',
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
        <span class="metric-label">System Status</span><br>
        <span class="good">NOMINAL</span><br><br>
        <span class="metric-label">Assessment</span><br>
        <span class="{risk_class}">{risk} RISK</span>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="card-title">AI Hazard Detection</div>', unsafe_allow_html=True)
    st.write(f"🔴 Craters detected: **{len(craters)}**")
    st.write(f"🟠 Boulder fields detected: **{len(boulders)}**")
    st.write(f"🟡 Local hazards near LZ: **{total_hazards}**")
    st.write(f"🟢 Regolith stability estimate: **{best_hazard * 100:.0f}%**")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Secondary analysis panels
# -----------------------------
c1, c2, c3 = st.columns(3, gap="large")

with c1:
    st.markdown('<div class="small-card"><div class="card-title">Slope Hazard Map</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_heatmap("Slope, degrees", x, y, slope, "Inferno", best_x, best_y),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="small-card"><div class="card-title">Solar Exposure Analysis</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_heatmap("Relative Solar Exposure", x, y, solar_score, "YlOrBr", best_x, best_y),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with c3:
    st.markdown('<div class="small-card"><div class="card-title">Communication Coverage</div>', unsafe_allow_html=True)
    st.plotly_chart(
        build_heatmap("Relay Link Quality", x, y, comm_score, "Blues", best_x, best_y),
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

    hab_offset = 115
    habitat_x = clamp(best_x + hab_offset, -480, 480)
    habitat_y = clamp(best_y + hab_offset * 0.45, -480, 480)
    power_x = clamp(best_x - 120, -480, 480)
    power_y = clamp(best_y + 85, -480, 480)
    comm_x = clamp(best_x + 180, -480, 480)
    comm_y = clamp(best_y - 75, -480, 480)

    plan_df = pd.DataFrame(
        [
            ["Primary Landing Zone", best_x, best_y, "Low slope, strong hazard clearance"],
            ["Habitat Zone", habitat_x, habitat_y, "Offset from descent plume and local hazards"],
            ["Solar Power Zone", power_x, power_y, "Higher exposure and clear line of sight"],
            ["Comms Relay", comm_x, comm_y, "Improves Earth/relay geometry"],
        ],
        columns=["Element", "X m", "Y m", "Rationale"],
    )

    st.dataframe(plan_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        **Engineering interpretation:** the recommended site balances terrain safety, slope, solar power availability,
        and communication geometry. A habitat should not be placed directly at the touchdown point because plume effects,
        dust lofting, and landing dispersions create avoidable risk.
        """
    )
    st.markdown("</div>", unsafe_allow_html=True)

with h2:
    st.markdown('<div class="section-card"><div class="card-title">Rover Path Planning</div>', unsafe_allow_html=True)

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
            marker=dict(size=8),
            name="Rover Survey Path",
            hovertemplate="Rover Waypoint<br>X: %{x:.1f} m<br>Y: %{y:.1f} m<extra></extra>",
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
    st.write(f"Estimated survey time: **{survey_time_hr:.2f} hr** at **{active_params['rover_speed']:.2f} m/s**")
    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Recommendations and export
# -----------------------------
r1, r2 = st.columns([1.6, 1.0], gap="large")

recommendations = [
    "Select the identified landing zone as the primary candidate if orbital imagery confirms local hazard clearance.",
    "Place the habitat outside the touchdown zone to reduce dust, plume, and descent dispersion risk.",
    "Deploy a solar array on the higher-exposure ridge and validate shadowing across the local lunar day.",
    "Use LiDAR, stereo EO imagery, thermal imaging, and ground-penetrating radar for final site verification.",
    "Establish a relay node before crew arrival to improve command, telemetry, and contingency operations.",
]

with r1:
    st.markdown('<div class="section-card"><div class="card-title">AI Mission Recommendations</div>', unsafe_allow_html=True)
    for rec in recommendations:
        st.success(rec)
    st.markdown("</div>", unsafe_allow_html=True)

with r2:
    st.markdown('<div class="section-card"><div class="card-title">Mission Summary & Export</div>', unsafe_allow_html=True)

    summary = {
        "mission_id": active_params["mission_id"],
        "target_body": active_params["target_body"],
        "mission_profile": active_params["mission_profile"],
        "timestamp_utc": st.session_state.get("assessment_timestamp", datetime.now(timezone.utc).isoformat()),
        "recommended_landing_zone": {"x_m": best_x, "y_m": best_y, "score": best_score},
        "slope_deg": best_slope,
        "solar_index": best_solar,
        "communication_index": best_comm,
        "local_hazards": total_hazards,
        "craters_detected": len(craters),
        "boulders_detected": len(boulders),
        "survey_path_km": total_path_m / 1000,
        "estimated_survey_time_hr": survey_time_hr,
        "survey_coverage_percent": survey_coverage,
        "recommendations": recommendations,
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
                "lz_x_m": best_x,
                "lz_y_m": best_y,
                "lz_score": best_score,
                "slope_deg": best_slope,
                "solar_index": best_solar,
                "communication_index": best_comm,
                "local_hazards": total_hazards,
                "craters_detected": len(craters),
                "boulders_detected": len(boulders),
                "survey_path_km": total_path_m / 1000,
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
        Autonomy | Intelligence | Exploration | Systems Engineering
    </div>
    """,
    unsafe_allow_html=True,
)
