import streamlit as st
import pandas as pd
from analysis import analyze_video
from utils import save_uploaded_file

# PAGE CONFIG
st.set_page_config(
    page_title="Badminton AI",
    page_icon="🏸",
    layout="wide"
)

# CUSTOM CSS
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background: linear-gradient(to bottom right, #0b0f19, #111827);
    color: white;
}

/* PAGE */
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

/* REMOVE DEFAULT */
header, footer {
    visibility: hidden;
}

/* TITLE */
.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-bottom: 5px;
}

/* SUBTITLE */
.sub-title {
    text-align: center;
    color: #9ca3af;
    font-size: 16px;
    margin-bottom: 30px;
}

/* UPLOAD BOX */
.upload-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    margin-bottom: 25px;
}

/* METRIC CARD */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 18px;
    border-radius: 18px;
    backdrop-filter: blur(8px);
    transition: 0.3s;
}

/* HOVER EFFECT */
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    border: 1px solid #3b82f6;
}

/* METRIC LABEL */
[data-testid="stMetricLabel"] {
    font-size: 13px !important;
    color: #9ca3af !important;
}

/* METRIC VALUE */
[data-testid="stMetricValue"] {
    font-size: 28px !important;
    color: white !important;
    font-weight: bold;
}

/* VIDEO */
.stVideo {
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

/* SECTION HEADINGS */
.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 15px;
    color: white;
}

/* ALERTS */
.stAlert {
    border-radius: 14px;
    font-size: 14px;
}

/* CHART */
canvas {
    border-radius: 15px;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] {
    background: transparent;
}

/* DIVIDER */
hr {
    margin-top: 18px;
    margin-bottom: 18px;
    border-color: rgba(255,255,255,0.08);
}

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown(
    "<div class='main-title'>🏸 Badminton AI Analyzer</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='sub-title'>AI Powered Player Performance & Movement Analysis System</div>",
    unsafe_allow_html=True
)

# UPLOAD SECTION
st.markdown("<div class='upload-box'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Badminton Video",
    type=["mp4", "mov", "avi"]
)

st.markdown("</div>", unsafe_allow_html=True)

# VIDEO ANALYSIS
if uploaded_file:

    video_path = save_uploaded_file(uploaded_file)

    st.video(video_path)

    with st.spinner("Analyzing AI Performance..."):
        result = analyze_video(video_path)

    st.divider()

    # DASHBOARD
    st.markdown(
        "<div class='section-title'>📊 Performance Dashboard</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Movement Score",
        f"{result['movement_score']}%"
    )

    col2.metric(
        "Fitness",
        result['fitness']
    )

    col3.metric(
        "Performance Grade",
        result['grade']
    )

    col4.metric(
        "Stamina",
        result['stamina']
    )

    st.divider()

    # SECOND ROW
    col5, col6, col7, col8 = st.columns(4)

    col5.metric(
        "Jump Count",
        result['jump_count']
    )

    col6.metric(
        "Smash Activity",
        result['smash_count']
    )

    col7.metric(
        "Court Coverage",
        f"{result['court_coverage']}%"
    )

    col8.metric(
        "Footwork Speed",
        result['footwork']
    )

    st.divider()

    # CHART + INFO
    left, right = st.columns([2,1])

    with left:

        st.markdown(
            "<div class='section-title'>📈 Attack vs Defense</div>",
            unsafe_allow_html=True
        )

        chart_data = pd.DataFrame({
            "Category": ["Attack", "Defense"],
            "Score": [
                result['attack_score'],
                result['defense_score']
            ]
        })

        st.bar_chart(
            chart_data.set_index("Category")
        )

    with right:

        st.markdown(
            "<div class='section-title'>⚠ Weak Position</div>",
            unsafe_allow_html=True
        )

        st.error(result['weak_side'])

        st.markdown(
            "<div class='section-title'>🎯 Dominant Zone</div>",
            unsafe_allow_html=True
        )

        st.success(result['dominant_zone'])

    st.divider()

    # AI FEEDBACK
    st.markdown(
        "<div class='section-title'>🤖 AI Coach Feedback</div>",
        unsafe_allow_html=True
    )

    for s in result['suggestions']:
        st.warning(s)

    st.divider()

    # FOOTER
    st.markdown(
        """
        <center style='color:gray;font-size:13px'>
        Powered by AI Pose Detection • Computer Vision • Sports Analytics
        </center>
        """,
        unsafe_allow_html=True
    )