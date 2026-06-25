import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go

def create_vehicle_3d(slope_deg, roll_deg):

    pitch = np.radians(slope_deg)
    roll  = np.radians(roll_deg)

    L, W, H = 1.5, 1.0, 0.5

    corners = np.array([
        [-L/2, -W/2, -H/2],
        [ L/2, -W/2, -H/2],
        [ L/2,  W/2, -H/2],
        [-L/2,  W/2, -H/2],
        [-L/2, -W/2,  H/2],
        [ L/2, -W/2,  H/2],
        [ L/2,  W/2,  H/2],
        [-L/2,  W/2,  H/2],
    ])

    R_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0,              1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_roll = np.array([
        [1, 0,               0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    R = R_pitch @ R_roll
    rotated = corners @ R.T

    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]

    lines = []
    for e in edges:
        lines.append(go.Scatter3d(
            x=[rotated[e[0]][0], rotated[e[1]][0]],
            y=[rotated[e[0]][1], rotated[e[1]][1]],
            z=[rotated[e[0]][2], rotated[e[1]][2]],
            mode='lines',
            line=dict(width=4)
        ))

    fig = go.Figure(data=lines)

    # Find lowest point of the vehicle
    z_min = np.min(rotated[:, 2])

    # Define ground plane corners
    plane_size = 3
    ground = np.array([
        [-plane_size, -plane_size, z_min],
        [ plane_size, -plane_size, z_min],
        [ plane_size,  plane_size, z_min],
        [-plane_size,  plane_size, z_min]
    ])

    # Create ground surface
    ground_surface = go.Mesh3d(
        x=ground[:, 0],
        y=ground[:, 1],
        z=ground[:, 2],
        color='lightgray',
        opacity=0.5
    )

    # Add to figure
    fig = go.Figure(data=lines + [ground_surface])

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-2,2]),
            yaxis=dict(range=[-2,2]),
            zaxis=dict(range=[-2,2]),

            # Regular view camera is distorted, so change aspect mode
            aspectmode='cube',

            camera=dict(
                projection=dict(type="orthographic"),
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )

    

    return fig

DT = 0.01

@st.cache_data
def load_data():
    df = pd.read_csv("jan7combine.csv")
    df = df.sort_values("TimeBucket").reset_index(drop=True)
    return df

df = load_data()

st.title("IMU Gravity & Vehicle Visualizer")

if "playing" not in st.session_state:
    st.session_state.playing = False

if "t_index" not in st.session_state:
    st.session_state.t_index = 0

p1, p2, p3 = st.columns(3)

with p1:
    if st.button("▶ Play"):
        st.session_state.playing = True

with p2:
    if st.button("⏸ Pause"):
        st.session_state.playing = False

with p3:
    speed = st.slider("Playback Speed (frames/sec)", 1, 30, 10)

step_size = st.slider(
    "Step Size (Elapsed time between frame jumps) [seconds]",
    1, 10000, 2000
)

step_time = step_size * DT
st.write(f"Each frame skips ≈ {step_time:.2f} seconds")

t_index = st.slider(
    f"Time (seconds) — current: {st.session_state.t_index * DT:.2f}s",
    0,
    len(df) - 1,
    st.session_state.t_index
)

st.session_state.t_index = t_index

window_size = st.slider(
    "Averaging Window (rows)",
    5,
    100,
    20
)

st.subheader("Baseline Gravity")

g1, g2, g3 = st.columns(3)
with g1: g0_x = st.number_input("X", value=-6.38)
with g2: g0_y = st.number_input("Y", value=-7.38)
with g3: g0_z = st.number_input("Z", value=2.38)

g_ref = np.array([g0_x, g0_y, g0_z])
z_axis = g_ref / np.linalg.norm(g_ref)

st.subheader("Forward Direction")

f1, f2, f3 = st.columns(3)
with f1: f_x = st.number_input("Forward X", value=1.0)
with f2: f_y = st.number_input("Forward Y", value=0.0)
with f3: f_z = st.number_input("Forward Z", value=0.0)

f_input = np.array([f_x, f_y, f_z])
f_unit = f_input / np.linalg.norm(f_input)

start = max(0, t_index - window_size)
end = min(len(df), t_index + window_size)
window = df.iloc[start:end]

g = window[['AccelX','AccelY','AccelZ']].mean().values
g_unit = g / np.linalg.norm(g)

raw = df.iloc[t_index][['AccelX','AccelY','AccelZ']].values
raw_unit = raw / np.linalg.norm(raw)

y_axis = np.cross(z_axis, f_unit)
y_axis = y_axis / np.linalg.norm(y_axis)

x_axis = np.cross(y_axis, z_axis)
x_axis = x_axis / np.linalg.norm(x_axis)

g_vehicle = np.array([
    np.dot(g_unit, x_axis),
    np.dot(g_unit, y_axis),
    np.dot(g_unit, z_axis)
])

slope_deg = np.degrees(np.arcsin(g_vehicle[0]))
roll_deg  = np.degrees(np.arcsin(g_vehicle[1]))

tilt_deg = np.degrees(
    np.arccos(np.clip(np.dot(g_unit, z_axis), -1.0, 1.0))
)

col1, col2, col3 = st.columns([1.2, 1.2, 1])

with col1:
    st.subheader("3D Orientation")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(0,0,0,*g_unit,color='blue')
    ax.quiver(0,0,0,*z_axis,color='green')
    ax.quiver(0,0,0,*x_axis,color='black')
    ax.quiver(0,0,0,*y_axis,color='purple')

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])

    st.pyplot(fig)

with col2:

    st.subheader("Slope (Rear → Front)")

    fig2, ax2 = plt.subplots()

    length = 1.6
    height = 0.4

    theta = np.radians(np.clip(slope_deg, -45, 45))

    corners = np.array([
        [-length/2, -height/2],
        [ length/2, -height/2],
        [ length/2,  height/2],
        [-length/2,  height/2]
    ])

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    ax2.add_patch(plt.Polygon(corners @ R.T, fill=False, linewidth=2))
    ax2.set_aspect('equal')
    ax2.set_xlim([-2,2])
    ax2.set_ylim([-1.5,1.5])
    ax2.axhline(0)
    ax2.axvline(0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(-length/2 - 0.3, 0, "Rear")
    ax2.text(length/2 + 0.2, 0, "Front")
    st.pyplot(fig2)
    st.subheader("Roll (Left -> Right)")

    fig3, ax3 = plt.subplots()

    width = 1.6
    height = 0.4

    phi = np.radians(np.clip(roll_deg, -45, 45))

    corners_roll = np.array([
        [-width/2, -height/2],
        [ width/2, -height/2],
        [ width/2,  height/2],
        [-width/2,  height/2]
    ])

    R_roll = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi),  np.cos(phi)]
    ])

    rotated_roll = corners_roll @ R_roll.T
    ax3.add_patch(plt.Polygon(rotated_roll, fill=False, linewidth=2))

    # labels
    ax3.text(-width/2 - 0.3, 0, "Left")
    ax3.text(width/2 + 0.2, 0, "Right")

    ax3.set_xlim([-2,2])
    ax3.set_ylim([-1.5,1.5])
    ax3.set_aspect('equal')

    ax3.axhline(0)
    ax3.axvline(0)
    ax3.set_xticks([])
    ax3.set_yticks([])

    st.pyplot(fig3)

with col3:

    st.subheader("Outputs")

    st.markdown(f"### Slope: {slope_deg:.2f}°")
    st.markdown(f"### Roll: {roll_deg:.2f}°")
    st.markdown(f"### Tilt: {tilt_deg:.2f}°")

# 3D Vehicle Visualization
st.subheader("3D Vehicle Model")

fig_vehicle = create_vehicle_3d(slope_deg, roll_deg)
st.plotly_chart(fig_vehicle, use_container_width=True)


# Playback loop
if st.session_state.playing:
    time.sleep(1.0 / speed)

    if st.session_state.t_index < len(df) - step_size:
        st.session_state.t_index += step_size
    else:
        st.session_state.t_index = 0

    st.rerun()