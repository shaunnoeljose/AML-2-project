# app_local.py  — local-only Streamlit UI for SocraDesign

import os
import json
import uuid
import streamlit as st
import time

from socradesign_logic import (
    build_graph,
    get_initial_state,
    initialize_firestore,
    get_progress_indicator,
    save_state_to_firestore,
    save_analytics_snapshot,
    load_state_from_firestore,
    DoubleDiamondPhase,
    QUESTIONS_PER_PHASE,
)

# ---------------------------------------------------------
# 0. Local Firestore service account wiring
# ---------------------------------------------------------
# Adjust these if your file name/path is different
SERVICE_ACCOUNT_CANDIDATES = [
    "firestore-service-account.json",
    "firestore-service-account",
]

def _ensure_firestore_env_from_file():
    """For local dev: read service-account JSON file and push into env var."""
    if os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON"):
        # Already set (e.g., via shell), don't override
        return

    for candidate in SERVICE_ACCOUNT_CANDIDATES:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                # Basic sanity check: looks like JSON
                json.loads(content)
                os.environ["FIREBASE_SERVICE_ACCOUNT_JSON"] = content
                return
            except Exception as e:
                st.error(f"Found {candidate} but failed to load as JSON: {e}")
                return

    # If we reach here, nothing found
    st.warning(
        "Could not find a local Firestore service account file.\n\n"
        "Expected one of:\n"
        "- firestore-service-account.json\n"
        "- firestore-service-account\n\n"
        "Please place your Firebase service account JSON in this folder with one of these names."
    )

_ensure_firestore_env_from_file()

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="SocraDesign — Double Diamond Socratic Tutor",
    page_icon="🐊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# UF Color Scheme & Custom CSS (FIXED LAYOUT + FULL MOBILE SUPPORT)
# ---------------------------------------------------------
UF_ORANGE = "#FA4616"
UF_BLUE = "#0021A5"
UF_LIGHT_BG = "#F2F4FF"
TEXT_COLOR = "#1f2937"

st.markdown(f"""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        color: {TEXT_COLOR};
    }}
    
    /* --- 1. REMOVE TOP WHITE SPACE --- */
    /* Force top padding to be minimal so content starts higher */
    .main .block-container {{
        padding-top: 1rem !important; 
        padding-bottom: 2rem;
        max-width: 1000px;
    }}
    
    /* Hide the top colored decoration line */
    [data-testid="stDecoration"] {{
        display: none;
    }}

    /* Hide the top-right 'hamburger' menu (optional, keeps it clean) */
    [data-testid="stToolbar"] {{
        visibility: hidden;
    }}

    /* Ensure the header background is transparent, but the container exists 
       so the 'Open Sidebar' arrow remains clickable */
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}

    /* --- SIDEBAR --- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {UF_BLUE} 0%, #00156b 100%);
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdown"], 
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div {{
        color: white !important;
    }}
    
    [data-testid="stSidebar"] .stTextInput input {{
        color: {TEXT_COLOR};
    }}
    
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.2);
    }}

    /* --- HEADER --- */
    .main-header {{
        background: white;
        padding: 1.5rem 0;
        border-bottom: 2px solid {UF_ORANGE};
        margin-bottom: 1rem;
    }}
    
    .main-header h1 {{
        color: {UF_BLUE};
        font-weight: 800;
        font-size: 2.2rem;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    
    .main-header p {{
        color: #6b7280;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }}
    
    /* --- PILLS (DECLUTTERED) --- */
    .phase-pills {{
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 0.5rem;
        justify-content: flex-start;
        padding: 4px;
    }}
    
    .phase-pill {{
        padding: 6px 16px;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        display: flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s ease;
    }}
    
    .phase-pill-active {{ 
        background: {UF_ORANGE} !important; 
        color: white !important; 
        border: 1px solid {UF_ORANGE};
        box-shadow: 0 4px 10px rgba(250, 70, 22, 0.3);
        transform: translateY(-1px);
    }}
    
    .phase-pill-completed {{ 
        background: {UF_BLUE} !important; 
        color: white !important; 
        border: 1px solid {UF_BLUE};
        opacity: 0.9;
    }}
    
    .phase-pill-pending {{ 
        background: #F3F4F6 !important; 
        color: #9CA3AF !important; 
        border: 1px solid #E5E7EB;
    }}
    
    /* --- CHAT AREA (THE FIX: REDUCED HEIGHT) --- */
    .chat-container {{
        background: transparent;
        border-radius: 16px;
        padding-right: 10px;
        
        /* 40% height ensures header + chat + input fits on one screen */
        height: 40vh; 
        
        overflow-y: auto;
        margin-bottom: 0.5rem;
        display: flex;
        flex-direction: column;
        gap: 1rem;
        scroll-behavior: smooth;
        position: relative;
    }}
    
    /* Custom scrollbar */
    .chat-container::-webkit-scrollbar {{ width: 8px; }}
    .chat-container::-webkit-scrollbar-track {{ background: #f1f1f1; border-radius: 4px; }}
    .chat-container::-webkit-scrollbar-thumb {{ background: {UF_BLUE}; border-radius: 4px; }}
    
    /* AGENT BUBBLE */
    .message-agent {{
        align-self: flex-start;
        background: white;
        border: 1px solid #e5e7eb;
        border-left: 5px solid {UF_BLUE};
        color: {TEXT_COLOR};
        padding: 1rem 1.5rem;
        border-radius: 4px 18px 18px 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        max-width: 85%;
        line-height: 1.6;
    }}
    
    /* USER BUBBLE */
    .message-user {{
        align-self: flex-end;
        background-color: {UF_ORANGE};
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 18px 4px 18px 18px;
        box-shadow: 0 4px 8px rgba(250, 70, 22, 0.2);
        max-width: 80%;
        line-height: 1.6;
    }}
    
    .message-user div {{ color: white !important; }}
    
    /* --- SCROLL BUTTON --- */
    .scroll-to-bottom {{
        position: fixed;
        bottom: 100px;
        right: 30px;
        width: 48px;
        height: 48px;
        background: {UF_ORANGE};
        color: white;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(250, 70, 22, 0.4);
        display: none;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        transition: all 0.3s ease;
        z-index: 1000;
    }}
    
    .scroll-to-bottom.visible {{
        display: flex;
        animation: fadeInUp 0.3s ease;
    }}
    
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    /* --- BUTTONS --- */
    .stButton > button {{
        border: 1px solid {UF_BLUE};
        color: {UF_BLUE};
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }}
    
    .stButton > button:hover {{
        background: {UF_BLUE};
        color: white;
    }}
    
    /* --- PHASE CARD & TIPS --- */
    .phase-card, .tip-card {{
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }}
    
    .phase-title {{ color: #9ca3af; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem; }}
    .phase-name {{ color: {UF_BLUE}; font-size: 1.5rem; font-weight: 700; }}
    
    .tip-card h4 {{ color: {UF_ORANGE}; font-size: 0.85rem; font-weight: 600; margin: 0 0 0.5rem 0; }}
    .tip-card p {{ color: #666; font-size: 0.85rem; margin: 0; line-height: 1.5; }}
    
    /* --- PROGRESS BAR --- */
    .progress-container {{
        background: {UF_LIGHT_BG};
        border-radius: 100px;
        height: 8px;
        margin-top: 1rem;
        overflow: hidden;
    }}
    
    .progress-fill {{
        background: {UF_ORANGE};
        height: 100%;
        border-radius: 100px;
        transition: width 0.5s ease;
    }}
    
    /* Input Box styling */
    .stChatInput textarea:focus {{
        border-color: {UF_ORANGE} !important;
        box-shadow: 0 0 0 1px {UF_ORANGE} !important;
    }}
    
    /* Completion card */
    .completion-card {{
        background: linear-gradient(135deg, {UF_BLUE} 0%, #1a3a8f 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }}
    
    .completion-card h2 {{ color: {UF_ORANGE}; margin-bottom: 1rem; }}
    
    /* Hide default streamlit elements */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* =============================================
       MOBILE RESPONSIVENESS (FULLY PRESERVED)
       ============================================= */
    
    /* Tablets and smaller desktops */
    @media screen and (max-width: 1024px) {{
        .main .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
        
        .main-header {{
            padding: 1.25rem 1.5rem;
        }}
        
        .main-header h1 {{
            font-size: 1.5rem;
        }}
        
        .phase-pills {{
            gap: 6px;
        }}
        
        .phase-pill {{
            padding: 3px 10px;
            font-size: 0.7rem;
        }}
    }}
    
    /* Mobile devices */
    @media screen and (max-width: 768px) {{
        .main .block-container {{
            padding-top: 1rem;
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }}
        
        .main-header {{
            padding: 1rem;
            border-radius: 12px;
            margin-bottom: 1rem;
        }}
        
        .main-header h1 {{
            font-size: 1.3rem;
            gap: 8px;
        }}
        
        .main-header p {{
            font-size: 0.8rem;
        }}
        
        /* Stack phase pills in 2 rows on mobile */
        .phase-pills {{
            justify-content: center;
        }}
        
        .phase-pill {{
            padding: 4px 8px;
            font-size: 0.65rem;
        }}
        
        /* Chat container adjustments */
        .chat-container {{
            min-height: 300px;
            max-height: 45vh; /* Keep mobile height flexible */
            padding: 0.75rem;
        }}
        
        .message-agent, .message-user {{
            padding: 0.75rem 1rem;
        }}
        
        .message-label {{
            font-size: 0.7rem;
        }}
        
        /* Scroll button mobile */
        .scroll-to-bottom {{
            width: 44px;
            height: 44px;
            bottom: 90px;
            right: 15px;
            font-size: 1.3rem;
        }}
        
        /* Motivation banner mobile */
        .motivation-banner {{
            font-size: 0.85rem;
            padding: 0.6rem 0.75rem;
        }}
    }}
    
    /* Small mobile devices */
    @media screen and (max-width: 480px) {{
        .main-header h1 {{
            font-size: 1.1rem;
        }}
        
        .main-header p {{
            font-size: 0.75rem;
        }}
        
        /* Compact phase pills */
        .phase-pill {{
            padding: 3px 6px;
            font-size: 0.6rem;
        }}
        
        .phase-pill span {{
            display: none;
        }}
        
        .chat-container {{
            max-height: 40vh;
            min-height: 250px;
        }}
        
        .message-agent, .message-user {{
            padding: 0.6rem 0.8rem;
            font-size: 0.9rem;
        }}
        
        .phase-card, .tip-card {{
            padding: 0.6rem 0.8rem;
        }}
        
        .phase-title {{ font-size: 0.75rem; }}
        .phase-name {{ font-size: 1rem; }}
    }}
    
    /* Ensure sidebar looks good on mobile */
    @media screen and (max-width: 768px) {{
        [data-testid="stSidebar"] {{
            min-width: 250px !important;
        }}
        
        [data-testid="stSidebar"] h2 {{
            font-size: 1.2rem !important;
        }}
        
        [data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
        }}
    }}
    
    /* Touch-friendly improvements */
    @media (hover: none) and (pointer: coarse) {{
        .scroll-to-bottom {{
            width: 52px;
            height: 52px;
        }}
        
        .stButton > button {{
            min-height: 44px;
            padding: 0.6rem 1rem;
        }}
        
        [data-testid="stChatInput"] {{
            min-height: 50px;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
PHASE_ORDER = [
    "USER_PERSONA_SETUP",
    "AUDIENCE_PERSONA_SETUP",
    "DISCOVER",
    "DEFINE",
    "DEVELOP",
    "DELIVER",
    "COMPLETED",
]

PHASE_DISPLAY = {
    "USER_PERSONA_SETUP": ("👤", "About You"),
    "AUDIENCE_PERSONA_SETUP": ("🎯", "Your Audience"),
    "DISCOVER": ("🔍", "Discover"),
    "DEFINE": ("📌", "Define"),
    "DEVELOP": ("💡", "Develop"),
    "DELIVER": ("🚀", "Deliver"),
    "COMPLETED": ("✅", "Complete"),
}

PHASE_TIPS = {
    "USER_PERSONA_SETUP": "Share openly about yourself! The more context you provide, the better I can tailor our session to your needs.",
    "AUDIENCE_PERSONA_SETUP": "Think deeply about who you're designing for. Consider their daily challenges, motivations, and what success looks like for them.",
    "DISCOVER": "This is divergent thinking time! There are no wrong answers. Explore freely and challenge your assumptions.",
    "DEFINE": "Now we converge. What patterns do you see? What's the core problem that needs solving?",
    "DEVELOP": "Let your creativity flow! Wild ideas welcome. What if there were no constraints?",
    "DELIVER": "Time to get practical. Which idea has the most potential? How would you actually build it?",
}

MOTIVATIONAL_MESSAGES = [
    "🔥 You're doing great! Keep that momentum going!",
    "💪 Every answer brings you closer to clarity!",
    "🌟 Your insights are shaping something meaningful!",
    "🎯 Stay curious — the best ideas come from deep thinking!",
    "🚀 You're making excellent progress!",
    "✨ Great thinking! Let's keep exploring!",
]

def get_phase_index(phase_name: str) -> int:
    try:
        return PHASE_ORDER.index(phase_name)
    except ValueError:
        return 0

def render_phase_pills(current_phase: str) -> str:
    current_idx = get_phase_index(current_phase)
    pills_html = '<div class="phase-pills">'
    for i, phase in enumerate(PHASE_ORDER[:-1]):  # exclude COMPLETED pill
        emoji, name = PHASE_DISPLAY[phase]
        if i < current_idx:
            pills_html += f'<span class="phase-pill phase-pill-completed">{emoji} {name}</span>'
        elif i == current_idx:
            pills_html += f'<span class="phase-pill phase-pill-active">{emoji} {name}</span>'
        else:
            pills_html += f'<span class="phase-pill phase-pill-pending">{emoji} {name}</span>'
    pills_html += "</div>"
    return pills_html

def render_progress_bar(current_phase: str, question_count: int) -> str:
    phase_idx = get_phase_index(current_phase)
    total_phases = len(PHASE_ORDER) - 1  # exclude COMPLETED
    phase_progress = phase_idx / total_phases
    question_progress = (question_count / QUESTIONS_PER_PHASE) / total_phases
    total_progress = min((phase_progress + question_progress) * 100, 100)
    return f"""
    <div class="progress-container">
        <div class="progress-fill" style="width: {total_progress}%"></div>
    </div>
    """

def render_phase_card(state) -> str:
    phase = state["current_phase"]
    question_count = state["question_count_in_phase"]
    emoji, name = PHASE_DISPLAY.get(phase, ("📊", phase))
    return f"""
    <div class="phase-card">
        <div class="phase-title">Current Phase</div>
        <div class="phase-name">{emoji} {name}</div>
        <div style="color: #666; font-size: 0.85rem; margin-top: 0.25rem;">
            Question {question_count + 1} of {QUESTIONS_PER_PHASE}
        </div>
        {render_progress_bar(phase, question_count)}
    </div>
    """

def render_tip_card(phase: str) -> str:
    tip = PHASE_TIPS.get(phase, "Take your time and think deeply about each question.")
    return f"""
    <div class="tip-card">
        <h4>💡 Pro Tip</h4>
        <p>{tip}</p>
    </div>
    """

def get_random_motivation() -> str:
    import random
    return random.choice(MOTIVATIONAL_MESSAGES)

# ---------------------------------------------------------
# 1. Initialize Firestore + LangGraph once per session
# ---------------------------------------------------------
if "db" not in st.session_state:
    db = initialize_firestore()
    if db is None:
        st.error(
            "🔥 Could not connect to Firestore.\n\n"
            "For local use, ensure firestore-service-account.json is present "
            "and valid, or set FIREBASE_SERVICE_ACCOUNT_JSON manually."
        )
        st.stop()
    st.session_state.db = db

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "show_motivation" not in st.session_state:
    st.session_state.show_motivation = False

# ---------------------------------------------------------
# 2. Session management (sidebar)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: white; margin: 0;">🐊 SocraDesign</h2>
        <p style="color: rgba(255,255,255,0.7); font-size: 0.8rem; margin-top: 0.5rem;">
            Local Dev Mode
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">📁 Session Management</p>',
        unsafe_allow_html=True,
    )

    current_id = st.session_state.get("session_id", "")
    if current_id:
        st.markdown(
            f'<p style="color: rgba(255,255,255,0.8); font-size: 0.8rem;">'
            f'Current: <code style="background: rgba(255,255,255,0.1); '
            f'padding: 2px 6px; border-radius: 4px;">{current_id}</code></p>',
            unsafe_allow_html=True,
        )

    manual_id = st.text_input(
        "Resume a session",
        value="",
        placeholder="Paste session ID here…",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns(2)
    with col1:
        new_clicked = st.button("➕ New", use_container_width=True)
    with col2:
        load_clicked = st.button("📂 Load", use_container_width=True)

    st.markdown("---")

    if "agent_state" in st.session_state:
        analytics = st.session_state.agent_state.get("analytics", {})
        st.markdown(
            '<p style="color: white; font-weight: 600;">📊 Quick Stats</p>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Turns", analytics.get("total_turns", 0))
        with c2:
            st.metric("Phase Qs", analytics.get("phase_q_count", 0))

    st.markdown("---")
    st.markdown(
        '<p style="color: white; font-weight: 600;">❓ Need Help?</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    <p style="color: rgba(255,255,255,0.7); font-size: 0.8rem; line-height: 1.5;">
    This tool guides you through the <strong>Double Diamond</strong> design thinking process using Socratic questioning.
    </p>
    <p style="color: rgba(255,255,255,0.7); font-size: 0.8rem; line-height: 1.5;">
    Type <code style="background: rgba(255,255,255,0.1); padding: 2px 4px; border-radius: 3px;">quit</code> to end early.
    </p>
    """,
        unsafe_allow_html=True,
    )

def _create_new_session():
    new_id = f"local-{uuid.uuid4().hex[:8]}"
    state = get_initial_state(new_id)
    save_state_to_firestore(st.session_state.db, new_id, state)
    save_analytics_snapshot(st.session_state.db, state)
    st.session_state.session_id = new_id
    st.session_state.agent_state = state
    st.session_state.show_motivation = False

def _load_session_by_id(sid: str):
    state = load_state_from_firestore(st.session_state.db, sid)
    if state is None:
        st.warning(f"No existing session found for ID `{sid}`. Creating a new one.")
        state = get_initial_state(sid)
        save_state_to_firestore(st.session_state.db, sid, state)
        save_analytics_snapshot(st.session_state.db, state)
    st.session_state.session_id = sid
    st.session_state.agent_state = state

if new_clicked:
    _create_new_session()
    st.rerun()

if load_clicked:
    if manual_id.strip():
        _load_session_by_id(manual_id.strip())
        st.rerun()
    else:
        st.sidebar.warning("Enter a session ID or click New")

if "agent_state" not in st.session_state:
    _create_new_session()

agent_state = st.session_state.agent_state
graph = st.session_state.graph
db = st.session_state.db

# ---------------------------------------------------------
# 3. Ensure first Socratic question
# ---------------------------------------------------------
if not agent_state["conversation_history"]:
    result_state = graph.invoke(agent_state, {})
    agent_state.update(result_state)
    save_state_to_firestore(db, agent_state["session_id"], agent_state)
    save_analytics_snapshot(db, agent_state)
    st.session_state.agent_state = agent_state

# ---------------------------------------------------------
# 4. Main Header
# ---------------------------------------------------------
st.markdown(
    """
<div class="main-header">
    <h1>🐊 SocraDesign</h1>
    <p>Your AI-powered guide through the Double Diamond design thinking process (Local Mode)</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 5. Phase Progress Pills
# ---------------------------------------------------------
st.markdown(render_phase_pills(agent_state["current_phase"]), unsafe_allow_html=True)

# ---------------------------------------------------------
# 6. Main Content
# ---------------------------------------------------------
if agent_state["current_phase"] == DoubleDiamondPhase.COMPLETED.name:
    st.markdown(
        """
    <div class="completion-card">
        <h2>🎉 Congratulations!</h2>
        <p>You've completed your design thinking journey!</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 Your Final Report")
    st.markdown(agent_state.get("final_summary", "Session completed."))

    if st.button("🔄 Start a New Design Journey", use_container_width=True):
        _create_new_session()
        st.rerun()
else:
    col_main, col_side = st.columns([2.5, 1])

    with col_main:
        if st.session_state.show_motivation:
            st.markdown(
                f'<div class="motivation-banner">{get_random_motivation()}</div>',
                unsafe_allow_html=True,
            )
            st.session_state.show_motivation = False

        st.markdown("### Conversation")

        chat_messages_html = ""
        for turn in agent_state["conversation_history"]:
            role = turn["role"]
            # Escape user text so they can't break your HTML
            text = (
                turn["text"]
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br>")
            )

            if role == "user":
                chat_messages_html += f"""
<div class="message-user">
    <div>{text}</div>
</div>
"""
            else:
                chat_messages_html += f"""
<div class="message-agent">
    <div style="font-weight:600; font-size:0.75rem; color:{UF_BLUE}; margin-bottom:4px;">🐊 SocraDesign</div>
    <div>{text}</div>
</div>
"""

        chat_messages_html += '<div id="chat-bottom"></div>'

        st.markdown(
            f"""
<div style="position: relative;">
    <div class="chat-container" id="chat-container">
        {chat_messages_html}
    </div>

<button class="scroll-to-bottom" id="scrollToBottomBtn" onclick="scrollToBottom()">
↓
</button>
</div>

<script>
(function() {{
    var chatContainer = document.getElementById('chat-container');
    var scrollBtn = document.getElementById('scrollToBottomBtn');

    window.scrollToBottom = function() {{
        if (chatContainer) {{
            chatContainer.scrollTo({{
                top: chatContainer.scrollHeight,
                behavior: 'smooth'
            }});
        }}
    }};

    if (chatContainer) {{
        setTimeout(function() {{ chatContainer.scrollTop = chatContainer.scrollHeight; }}, 100);
    }}

    if (chatContainer && scrollBtn) {{
        chatContainer.addEventListener('scroll', function() {{
            var threshold = 50;
            var position = chatContainer.scrollTop + chatContainer.offsetHeight;
            var height = chatContainer.scrollHeight;

            if (height - position > threshold) {{
                                scrollBtn.classList.add('visible');
                            }} else {{
                                scrollBtn.classList.remove('visible');
                            }}
                        }});
                    }}
                }})();
</script>
""",
            unsafe_allow_html=True,
        )

        user_msg = st.chat_input("Type your response here… (or 'quit' to end)")
        if user_msg:
            agent_state["last_user_response"] = user_msg
            result_state = graph.invoke(agent_state, {})
            agent_state.update(result_state)
            save_state_to_firestore(db, agent_state["session_id"], agent_state)
            save_analytics_snapshot(db, agent_state)
            st.session_state.agent_state = agent_state
            st.session_state.show_motivation = True
            st.rerun()

    with col_side:
        st.markdown(render_phase_card(agent_state), unsafe_allow_html=True)
        st.markdown(
            render_tip_card(agent_state["current_phase"]), unsafe_allow_html=True
        )

        if agent_state.get("user_persona") and agent_state["user_persona"] != "unknown":
            st.markdown(
                f"""
            <div style="background: {UF_LIGHT_BLUE}; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
                <div style="font-size: 0.75rem; color: {UF_BLUE}; font-weight: 600; text-transform: uppercase;">👤 You</div>
                <div style="font-size: 0.85rem; color: #333; margin-top: 0.25rem;">{agent_state["user_persona"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        if (
            agent_state.get("audience_persona")
            and agent_state["audience_persona"] != "unknown"
        ):
            st.markdown(
                f"""
            <div style="background: {UF_LIGHT_ORANGE}; padding: 1rem; border-radius: 8px; margin-top: 0.5rem;">
                <div style="font-size: 0.75rem; color: {UF_ORANGE}; font-weight: 600; text-transform: uppercase;">🎯 Your Audience</div>
                <div style="font-size: 0.85rem; color: #333; margin-top: 0.25rem;">{agent_state["audience_persona"]}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )

# ---------------------------------------------------------
# 7. Debug Panel
# ---------------------------------------------------------
st.markdown("---")
with st.expander("🔧 Debug Panel & Analytics", expanded=False):
    analytics = agent_state.get("analytics", {})
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**📋 Session Info**")
        st.json(
            {
                "session_id": agent_state.get("session_id", ""),
                "current_phase": agent_state.get("current_phase", ""),
                "question_count": agent_state.get("question_count_in_phase", 0),
                "user_persona": agent_state.get("user_persona", "unknown"),
                "audience_persona": agent_state.get("audience_persona", "unknown"),
            }
        )

    with c2:
        st.markdown("**🎯 Last Turn**")
        st.json(
            {
                "last_question": agent_state.get("last_agent_question", "")[:100]
                + "...",
                "last_response": agent_state.get("last_user_response", "")[:100],
                "evaluation": agent_state.get("last_evaluation", ""),
                "tone": agent_state.get("current_user_tone", ""),
            }
        )

    with c3:
        st.markdown("**📊 Analytics**")
        st.json(
            {
                "total_turns": analytics.get("total_turns", 0),
                "weak_count": analytics.get("weak_count", 0),
                "confused_count": analytics.get("confused_count", 0),
                "probing_q": analytics.get("probing_q_count", 0),
                "rephrasing_q": analytics.get("rephrasing_q_count", 0),
                "avg_latency": round(analytics.get("avg_user_latency_sec", 0), 2),
            }
        )

    st.markdown("---")
    ct1, ct2, ct3 = st.columns(3)

    with ct1:
        st.markdown("**🎭 Tone Distribution**")
        tone_tracker = analytics.get("tone_tracker", [])
        if tone_tracker:
            tone_counts = {t: tone_tracker.count(t) for t in set(tone_tracker)}
            st.bar_chart(tone_counts)
        else:
            st.info("No tone data yet")

    with ct2:
        st.markdown("**✅ Evaluation Distribution**")
        eval_tracker = analytics.get("evaluation_tracker", [])
        if eval_tracker:
            eval_counts = {e: eval_tracker.count(e) for e in set(eval_tracker)}
            st.bar_chart(eval_counts)
        else:
            st.info("No evaluation data yet")

    with ct3:
        st.markdown("**⏱️ Phase Times (seconds)**")
        phase_times = analytics.get("phase_time_seconds", {})
        if phase_times:
            st.bar_chart(phase_times)
        else:
            st.info("No timing data yet")

    st.caption(
        "ℹ️ The local SLM vs Gemini evaluation choice is logged in server logs. "
        "Low-confidence predictions fall back to Gemini."
    )

# ---------------------------------------------------------
# 8. Footer
# ---------------------------------------------------------
st.markdown(
    f"""
<div style="text-align: center; padding: 2rem 0 1rem 0; color: #999; font-size: 0.8rem;">
    <p>Built with 🧡 for local experimentation</p>
    <p style="font-size: 0.7rem;">Powered by Gemini AI • LangGraph • Firestore</p>
</div>
""",
    unsafe_allow_html=True,
)
