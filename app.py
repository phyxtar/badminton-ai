import streamlit as st
import pandas as pd
import time
import os
from analysis import analyze_video, validate_badminton_video, detect_player_count
from utils import save_uploaded_file

st.set_page_config(page_title="Badminton AI", page_icon="🏸", layout="wide", initial_sidebar_state="collapsed")

USERS = {"admin": "admin123", "coach": "coach123", "player": "player123"}

for k, v in [("logged_in", False), ("username", ""), ("expert_result", None), ("expert_name", "")]:
    if k not in st.session_state: st.session_state[k] = v

params = st.query_params
if not st.session_state.logged_in and "user" in params:
    u = params["user"]
    if u in USERS:
        st.session_state.logged_in = True
        st.session_state.username = u

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { box-sizing: border-box; font-family: 'Inter', sans-serif; }
.stApp {
    background: radial-gradient(ellipse at 10% 40%, rgba(59,130,246,0.12) 0%, transparent 50%),
                radial-gradient(ellipse at 90% 10%, rgba(99,102,241,0.10) 0%, transparent 50%),
                linear-gradient(160deg, #05090f 0%, #0a1120 50%, #060c18 100%) !important;
    color: #e2e8f0;
}
.block-container { padding: 1.5rem 2rem !important; max-width: 1060px !important; margin: 0 auto !important; }
header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { visibility: hidden !important; height: 0 !important; }
[data-testid="stSidebar"] { display: none !important; }

.login-card {
    width: 360px; background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 20px;
    padding: 36px 32px 28px; backdrop-filter: blur(20px);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.06);
    text-align: center;
}
.login-icon { font-size: 42px; margin-bottom: 8px; }
.login-title { font-size: 20px; font-weight: 700; color: #fff; margin-bottom: 3px; }
.login-sub { font-size: 12px; color: #4b5563; }
.demo-hint { font-size: 10px; color: #2d3748; margin-top: 14px; line-height: 1.9; }

.topbar { display:flex; align-items:center; justify-content:space-between; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.05); margin-bottom:16px; }
.tb-left { display:flex; align-items:center; gap:10px; }
.tb-icon { font-size:20px; }
.tb-title { font-size:15px; font-weight:700; color:#f1f5f9; }
.tb-sub { font-size:10px; color:#374151; margin-top:1px; }
.user-badge { background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.22); border-radius:20px; padding:4px 12px; font-size:11px; color:#7dd3fc; }

.stTextInput > div > div > input { background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.09) !important; color:white !important; border-radius:9px !important; font-size:13px !important; padding:9px 13px !important; }
.stTextInput > div > div > input:focus { border-color:#3b82f6 !important; box-shadow:0 0 0 2px rgba(59,130,246,0.14) !important; }
.stTextInput label { font-size:11px !important; color:#6b7280 !important; font-weight:500 !important; }

.stButton > button { background:linear-gradient(90deg,#1d4ed8,#4338ca) !important; color:white !important; border:none !important; border-radius:9px !important; font-size:12px !important; font-weight:600 !important; padding:9px 18px !important; width:100% !important; }
.stButton > button:hover { opacity:0.86 !important; transform:translateY(-1px) !important; }

[data-testid="metric-container"] { background:rgba(255,255,255,0.02) !important; border:1px solid rgba(255,255,255,0.06) !important; border-radius:11px !important; padding:12px 14px !important; transition:all 0.2s !important; }
[data-testid="metric-container"]:hover { border-color:rgba(59,130,246,0.28) !important; background:rgba(59,130,246,0.04) !important; }
[data-testid="stMetricLabel"] p { font-size:10px !important; color:#6b7280 !important; font-weight:500 !important; text-transform:uppercase; letter-spacing:0.4px; }
[data-testid="stMetricValue"] { font-size:18px !important; color:#f1f5f9 !important; font-weight:700 !important; }
[data-testid="stMetricDelta"] { font-size:10px !important; }

.stTabs [data-baseweb="tab-list"] { background:rgba(255,255,255,0.02) !important; border-radius:9px !important; padding:3px !important; border:1px solid rgba(255,255,255,0.05) !important; }
.stTabs [data-baseweb="tab"] { border-radius:7px !important; font-size:12px !important; color:#6b7280 !important; padding:6px 14px !important; font-weight:500 !important; }
.stTabs [aria-selected="true"] { background:rgba(59,130,246,0.16) !important; color:#93c5fd !important; font-weight:600 !important; }

[data-testid="stFileUploader"] section { background:rgba(255,255,255,0.015) !important; border:1px dashed rgba(255,255,255,0.08) !important; border-radius:10px !important; }
[data-testid="stFileUploader"] label { font-size:12px !important; color:#6b7280 !important; }

video { border-radius:10px !important; max-height:200px !important; width:100% !important; object-fit:cover !important; }
.stAlert { border-radius:9px !important; font-size:12px !important; padding:8px 12px !important; }
hr { border-color:rgba(255,255,255,0.04) !important; margin:8px 0 !important; }
.sec { font-size:10px; font-weight:600; color:#64748b; margin:12px 0 6px; text-transform:uppercase; letter-spacing:0.8px; }
.cmp-banner { background:rgba(59,130,246,0.06); border:1px solid rgba(59,130,246,0.18); border-radius:8px; padding:8px 12px; font-size:12px; color:#7dd3fc; margin-bottom:12px; }
.stProgress > div > div > div { background:linear-gradient(90deg,#1d4ed8,#4338ca) !important; border-radius:8px !important; }
.stCaptionContainer p { font-size:11px !important; color:#6b7280 !important; }

.badge-single { display:inline-block; background:rgba(16,185,129,0.1); border:1px solid rgba(16,185,129,0.25); color:#6ee7b7; border-radius:20px; padding:3px 12px; font-size:11px; font-weight:600; }
.badge-double { display:inline-block; background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.25); color:#fcd34d; border-radius:20px; padding:3px 12px; font-size:11px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

SKELETON_SVG = """
<svg width="110" height="190" viewBox="0 0 110 190" style="display:block;margin:10px auto">
  <circle cx="55" cy="95" r="18" style="fill:none;stroke:rgba(59,130,246,0.3);animation:ringExpand 2s ease-out infinite"/>
  <circle cx="55" cy="95" r="18" style="fill:none;stroke:rgba(59,130,246,0.3);animation:ringExpand 2s ease-out 0.7s infinite"/>
  <line x1="55" y1="52" x2="55" y2="110" style="stroke:#3b82f6;stroke-width:2.5;stroke-linecap:round;animation:limbGlow 1.5s ease-in-out infinite alternate"/>
  <line x1="55" y1="62" x2="28" y2="86" style="stroke:#3b82f6;stroke-width:2.5;stroke-linecap:round;animation:limbGlow 1.5s ease-in-out 0.1s infinite alternate"/>
  <line x1="28" y1="86" x2="12" y2="112" style="stroke:#3b82f6;stroke-width:2;stroke-linecap:round;animation:limbGlow 1.5s ease-in-out 0.2s infinite alternate"/>
  <line x1="55" y1="62" x2="84" y2="76" style="stroke:#3b82f6;stroke-width:2.5;stroke-linecap:round;animation:limbGlow 1.5s ease-in-out 0.15s infinite alternate"/>
  <line x1="84" y1="76" x2="100" y2="50" style="stroke:#3b82f6;stroke-width:2;stroke-linecap:round;animation:limbGlow 1.5s ease-in-out 0.25s infinite alternate"/>
  <line x1="55" y1="110" x2="40" y2="148" style="stroke:#3b82f6;stroke-width:2.5;stroke-linecap:round;animation:limbGlow 1.5s ease-in-out 0.3s infinite alternate"/>
  <line x1="40" y1="148" x2="34" y2="178" style="stroke:#3b82f6;stroke-width:2;stroke-linecap:round;animation:limbGlow 1.5s ease-in-out 0.4s infinite alternate"/>
  <line x1="55" y1="110" x2="70" y2="148" style="stroke:#3b82f6;stroke-width:2.5;stroke-linecap:round;animation:limbGlow 1.5s ease-in-out 0.35s infinite alternate"/>
  <line x1="70" y1="148" x2="76" y2="178" style="stroke:#3b82f6;stroke-width:2;stroke-linecap:round;animation:limbGlow 1.5s ease-in-out 0.45s infinite alternate"/>
  <ellipse cx="104" cy="41" rx="9" ry="12" style="fill:none;stroke:#6366f1;stroke-width:1.5;animation:limbGlow 1.5s ease-in-out 0.5s infinite alternate"/>
  <line x1="100" y1="50" x2="104" y2="53" style="stroke:#6366f1;stroke-width:1.5;animation:limbGlow 1.5s ease-in-out 0.5s infinite alternate"/>
  <circle cx="55" cy="40" r="7" style="fill:none;stroke:#3b82f6;stroke-width:2;animation:glow 1.5s ease-in-out infinite alternate"/>
  <circle cx="55" cy="52" r="3.5" style="fill:rgba(59,130,246,0.3);stroke:#3b82f6;stroke-width:1.5;animation:glow 1.5s 0.1s infinite alternate"/>
  <circle cx="28" cy="86" r="3" style="fill:rgba(59,130,246,0.3);stroke:#3b82f6;stroke-width:1.5;animation:glow 1.5s 0.2s infinite alternate"/>
  <circle cx="84" cy="76" r="3" style="fill:rgba(59,130,246,0.3);stroke:#3b82f6;stroke-width:1.5;animation:glow 1.5s 0.15s infinite alternate"/>
  <circle cx="12" cy="112" r="2.5" style="fill:rgba(99,102,241,0.4);stroke:#6366f1;stroke-width:1.5;animation:glow 1.5s 0.3s infinite alternate"/>
  <circle cx="100" cy="50" r="2.5" style="fill:rgba(99,102,241,0.4);stroke:#6366f1;stroke-width:1.5;animation:glow 1.5s 0.25s infinite alternate"/>
  <circle cx="55" cy="110" r="3.5" style="fill:rgba(59,130,246,0.3);stroke:#3b82f6;stroke-width:1.5;animation:glow 1.5s 0.35s infinite alternate"/>
  <circle cx="40" cy="148" r="3" style="fill:rgba(59,130,246,0.3);stroke:#3b82f6;stroke-width:1.5;animation:glow 1.5s 0.4s infinite alternate"/>
  <circle cx="70" cy="148" r="3" style="fill:rgba(59,130,246,0.3);stroke:#3b82f6;stroke-width:1.5;animation:glow 1.5s 0.45s infinite alternate"/>
  <circle cx="34" cy="178" r="2.5" style="fill:rgba(59,130,246,0.3);stroke:#3b82f6;stroke-width:1.5;animation:glow 1.5s 0.5s infinite alternate"/>
  <circle cx="76" cy="178" r="2.5" style="fill:rgba(59,130,246,0.3);stroke:#3b82f6;stroke-width:1.5;animation:glow 1.5s 0.55s infinite alternate"/>
  <style>
    @keyframes glow { from{stroke:rgba(59,130,246,0.4)} to{stroke:rgba(99,102,241,1)} }
    @keyframes limbGlow { from{stroke:rgba(59,130,246,0.3)} to{stroke:rgba(139,92,246,0.9)} }
    @keyframes ringExpand { 0%{transform:scale(0);opacity:0.8} 100%{transform:scale(3);opacity:0} }
  </style>
</svg>
"""

def scan_html(msg):
    return (
        "<div style='background:rgba(5,9,15,0.92);border:1px solid rgba(59,130,246,0.15);"
        "border-radius:16px;padding:24px 20px;text-align:center;position:relative;overflow:hidden;margin:10px 0'>"
        "<div style='position:absolute;left:0;top:0;width:100%;height:2px;"
        "background:linear-gradient(90deg,transparent,#3b82f6,#6366f1,transparent);"
        "animation:scanMove 1.8s ease-in-out infinite'></div>"
        "<style>@keyframes scanMove{0%{top:0%;opacity:0}10%{opacity:1}90%{opacity:1}100%{top:100%;opacity:0}}</style>"
        f"<p style='font-size:12px;font-weight:600;color:#93c5fd;margin:0 0 4px'>🔬 {msg}</p>"
        + SKELETON_SVG +
        "<p style='font-size:11px;color:#374151;margin:8px 0 0'>AI motion analysis in progress...</p>"
        "</div>"
    )


def login_page():
    st.markdown("<div style='height:5vh'></div>", unsafe_allow_html=True)
    _, col, _ = st.columns([1, 1, 1])
    with col:
        st.markdown("""
        <div class='login-card'>
            <div class='login-icon'>🏸</div>
            <div class='login-title'>Badminton-Analysis</div>
            <div class='login-sub'>Sign in to continue</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter username", key="lu")
        password = st.text_input("Password", type="password", placeholder="Enter password", key="lp")
        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
        if st.button("🔐  Sign In", key="signin"):
            if username in USERS and USERS[username] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.query_params["user"] = username
                st.rerun()
            else:
                st.error("❌ Wrong credentials!")
        # st.markdown("<div class='demo-hint'>admin/admin123 · coach/coach123 · player/player123</div>", unsafe_allow_html=True)


def analyzer_page():
    tc1, tc2 = st.columns([7, 1])
    with tc1:
        st.markdown(f"""
        <div class='topbar'>
            <div class='tb-left'>
                <span class='tb-icon'>🏸</span>
                <div>
                    <div class='tb-title'>Badminton AI Analyzer</div>
                    <div class='tb-sub'>AI · Computer Vision · Motion Tracking</div>
                </div>
            </div>
            <span class='user-badge'>👤 {st.session_state.username}</span>
        </div>
        """, unsafe_allow_html=True)
    with tc2:
        st.markdown("<div style='padding-top:10px'>", unsafe_allow_html=True)
        if st.button("Logout"):
            for k in ["logged_in","username","expert_result","expert_name"]:
                del st.session_state[k]
            st.query_params.clear()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊  Analyze Player", "🎓  Expert Reference"])

    with tab2:
        c1, c2 = st.columns([1.2, 1])
        with c1:
            st.markdown("<div class='sec'>Set Expert Reference</div>", unsafe_allow_html=True)
            exp_name = st.text_input("Expert Name", placeholder="e.g. PV Sindhu, Lin Dan", key="en")
            expert_file = st.file_uploader("Upload Expert Video", type=["mp4","mov","avi"], key="eu")
            if expert_file:
                ep = save_uploaded_file(expert_file)
                st.video(ep)
                if st.button("🧠  Validate & Set Reference"):
                    with st.spinner("Validating..."):
                        is_valid, msg = validate_badminton_video(ep)
                    if not is_valid:
                        st.error(f"❌ {msg}")
                    else:
                        ph = st.empty()
                        ph.markdown(scan_html("Analyzing Expert..."), unsafe_allow_html=True)
                        res = analyze_video(ep)
                        ph.empty()
                        st.session_state.expert_result = res
                        st.session_state.expert_name = exp_name or "Expert"
                        st.success("✅ Reference set!")
        with c2:
            if st.session_state.expert_result:
                st.markdown("<div class='sec'>Active Reference</div>", unsafe_allow_html=True)
                r = st.session_state.expert_result
                st.markdown(f"<div style='margin-bottom:10px'><span class='badge-single'>✅ {st.session_state.expert_name}</span></div>", unsafe_allow_html=True)
                a, b = st.columns(2)
                a.metric("Movement", f"{r['movement_score']}%")
                b.metric("Grade", r['grade'])
                c3, d = st.columns(2)
                c3.metric("Fitness", r['fitness'])
                d.metric("Stamina", r['stamina'])
                if st.button("🗑️  Clear"):
                    st.session_state.expert_result = None
                    st.session_state.expert_name = ""
                    st.rerun()
            else:
                st.caption("No expert reference set yet.")

    with tab1:
        er = st.session_state.expert_result
        if er:
            st.markdown(f"<div class='cmp-banner'>🎯 Comparing against: <b>{st.session_state.expert_name}</b></div>", unsafe_allow_html=True)

        up_col, _ = st.columns([1.5, 1])
        with up_col:
            player_file = st.file_uploader("Upload Player Video", type=["mp4","mov","avi"], key="pu")

        if player_file:
            vpath = save_uploaded_file(player_file)
            vid_col, scan_col = st.columns([1.2, 1])

            with vid_col:
                st.video(vpath)

            with scan_col:
                vph = st.empty()
                vph.info("🔍 Validating video...")
                is_valid, val_msg = validate_badminton_video(vpath)
                if not is_valid:
                    vph.empty()
                    st.error(f"❌ {val_msg}")
                    st.warning("Please upload a badminton court video with players visible.")
                    st.stop()
                vph.success("✅ Valid badminton video!")
                time.sleep(0.3)
                vph.empty()

                dph = st.empty()
                dph.info("👥 Detecting players...")
                player_type, player_count = detect_player_count(vpath)
                dph.empty()

                if player_type == "Doubles":
                    st.markdown("<div style='margin-bottom:8px'><span class='badge-double'>👥 Doubles Match</span></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='margin-bottom:8px'><span class='badge-single'>🧍 Singles Match</span></div>", unsafe_allow_html=True)

                sph = st.empty()
                sph.markdown(scan_html("Scanning Player Motion..."), unsafe_allow_html=True)
                res = analyze_video(vpath)
                sph.empty()

                st.markdown("<div class='sec'>Quick Stats</div>", unsafe_allow_html=True)
                q1, q2 = st.columns(2)
                q1.metric("Grade", res['grade'])
                q2.metric("Shots", str(res['total_shots']))
                q3, q4 = st.columns(2)
                q3.metric("Distance", f"{res['distance_covered']}m")
                q4.metric("Avg Speed", f"{res['avg_speed']} m/s")

            st.divider()
            st.markdown("<div class='sec'>📊 Performance Dashboard</div>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            if er:
                c1.metric("Movement", f"{res['movement_score']}%", f"{res['movement_score']-er['movement_score']:+d}%")
                c2.metric("Coverage", f"{res['court_coverage']}%", f"{res['court_coverage']-er['court_coverage']:+d}%")
            else:
                c1.metric("Movement", f"{res['movement_score']}%")
                c2.metric("Coverage", f"{res['court_coverage']}%")
            c3.metric("Stamina", res['stamina'])
            c4.metric("Footwork", res['footwork'])

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Activity", f"{res['activity_percent']}%")
            c6.metric("Zone", res['dominant_zone'])
            c7.metric("Attack", str(res['attack_score']))
            c8.metric("Defense", str(res['defense_score']))

            st.divider()

            # ── Heatmap & Trajectory ──
            st.markdown("<div class='sec'>🔥 Court Analysis</div>", unsafe_allow_html=True)
            hm_col, tr_col = st.columns(2)
            with hm_col:
                st.caption("Player Position Heatmap")
                if res.get('heatmap_path') and os.path.exists(res['heatmap_path']):
                    st.image(res['heatmap_path'])
                else:
                    st.info("Heatmap data insufficient")
            with tr_col:
                st.caption("Movement Trajectory")
                if res.get('trajectory_path') and os.path.exists(res['trajectory_path']):
                    st.image(res['trajectory_path'])
                else:
                    st.info("Trajectory data insufficient")

            st.divider()

            # ── Shot Analysis ──
            st.markdown("<div class='sec'>🎯 Shot Analysis</div>", unsafe_allow_html=True)
            sh1, sh2, sh3, sh4 = st.columns(4)
            sh1.metric("Total Shots", str(res['total_shots']))
            sh2.metric("Front Court", str(res.get('shot_zones', {}).get('Front Court', 0)))
            sh3.metric("Mid Court", str(res.get('shot_zones', {}).get('Mid Court', 0)))
            sh4.metric("Rear Court", str(res.get('shot_zones', {}).get('Rear Court', 0)))
            if res.get('shot_chart_path') and os.path.exists(res['shot_chart_path']):
                st.image(res['shot_chart_path'])

            st.divider()

            # ── Movement Timeline ──
            st.markdown("<div class='sec'>📈 Movement Timeline</div>", unsafe_allow_html=True)
            if res.get('timeline_path') and os.path.exists(res['timeline_path']):
                st.image(res['timeline_path'])
            else:
                st.caption("Timeline chart not available")

            st.divider()

            # ── Motion Breakdown ──
            st.markdown("<div class='sec'>📐 Motion Breakdown</div>", unsafe_allow_html=True)
            ml, mr = st.columns([3, 1])
            with ml:
                st.progress(min(res['movement_score'], 100))
                x1, x2, x3 = st.columns(3)
                x1.caption(f"Distance: **{res['distance_covered']}m**")
                x2.caption(f"Dir Changes: **{res['direction_changes']}**")
                x3.caption(f"Max Speed: **{res['max_speed']} m/s**")
                x1.caption(f"Duration: **{res.get('duration', 0)}s**")
                x2.caption(f"Type: **{player_type}**")
                x3.caption(f"Players: **{player_count}**")
            with mr:
                st.markdown("<div class='sec'>⚠️ Weak Side</div>", unsafe_allow_html=True)
                if res['weak_side'] == "Balanced":
                    st.success("Balanced ✅")
                else:
                    st.error(res['weak_side'])

            st.divider()

            # ── Court Position Balance ──
            st.markdown("<div class='sec'>📐 Court Position Balance</div>", unsafe_allow_html=True)
            bl1, bl2 = st.columns(2)
            with bl1:
                st.caption(f"⬅️ Left {res['left_pct']}%  |  Right {res['right_pct']}% ➡️")
                st.progress(min(res['left_pct'], 100))
            with bl2:
                st.caption(f"⬆️ Front {res['front_pct']}%  |  Rear {res['rear_pct']}% ⬇️")
                st.progress(min(res['front_pct'], 100))

            if er:
                st.divider()
                st.markdown(f"<div class='sec'>🆚 vs {st.session_state.expert_name}</div>", unsafe_allow_html=True)
                df = pd.DataFrame({
                    "Metric": ["Movement", "Activity", "Coverage", "Attack", "Defense", "Footwork", "Grade"],
                    "You": [f"{res['movement_score']}%", f"{res['activity_percent']}%", f"{res['court_coverage']}%",
                            str(res['attack_score']), str(res['defense_score']), res['footwork'], res['grade']],
                    st.session_state.expert_name: [f"{er['movement_score']}%", f"{er['activity_percent']}%",
                            f"{er['court_coverage']}%", str(er['attack_score']), str(er['defense_score']), er['footwork'], er['grade']],
                    "Gap": [f"{res['movement_score']-er['movement_score']:+d}%",
                            f"{res['activity_percent']-er['activity_percent']:+d}%",
                            f"{res['court_coverage']-er['court_coverage']:+d}%",
                            f"{res['attack_score']-er['attack_score']:+d}",
                            f"{res['defense_score']-er['defense_score']:+d}", "—", "—"]
                })
                st.dataframe(df, width='stretch', hide_index=True)

            st.divider()
            st.markdown("<div class='sec'>🤖 AI Coach Feedback</div>", unsafe_allow_html=True)
            if er:
                if res['movement_score'] < er['movement_score']:
                    st.warning(f"📉 Movement {er['movement_score']-res['movement_score']}% below {st.session_state.expert_name}. Focus on speed drills.")
                if res['activity_percent'] < er['activity_percent']:
                    st.warning(f"🏃 {st.session_state.expert_name} is {er['activity_percent']-res['activity_percent']}% more active. Build endurance.")
                if res['court_coverage'] < er['court_coverage']:
                    st.warning(f"📐 Court coverage lower. Practice all 4 corners.")
            for s in res['suggestions']:
                st.info(s)

            st.markdown("<br><center style='color:#1f2937;font-size:10px'>Powered by AI Motion Tracking · Computer Vision · Sports Analytics</center>", unsafe_allow_html=True)


if not st.session_state.logged_in:
    login_page()
else:
    analyzer_page()