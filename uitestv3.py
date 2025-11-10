import streamlit as st
import uuid
import sys
import os

# Important: We are importing all our functions and classes from the other file
try:
    from prod_test import (
        build_graph, 
        AgentState, 
        DoubleDiamondPhase,
        get_session_summary,
        get_progress_indicator,
        logger
    )
except ImportError:
    st.error("Error: Could not find 'agent.py'. Make sure it's in the same directory.")
    st.stop()
except SystemExit as e:
    # This catches the sys.exit(1) from agent.py if the API key is missing
    st.error(f"Error loading agent.py. Is your GEMINI_API_KEY in the .env file? Details: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading agent.py: {e}")
    st.stop()


st.set_page_config(page_title="Socratic Agent", page_icon="🤖")
st.title("🤖 Socratic Ideation Agent")
st.markdown(
    "Following the Double Diamond design process, this agent will guide you from a vague idea to a defined solution."
)

# -----------------------------
# 1) Build (and cache) the graph
# -----------------------------
@st.cache_resource
def get_socratic_app():
    """
    Builds and caches the LangGraph app.
    """
    logger.info("Building Socratic graph...")
    # build_graph() is imported from agent.py
    app = build_graph()
    logger.info("Socratic graph built successfully.")
    return app

try:
    app = get_socratic_app()
except Exception as e:
    st.error(f"Error building the graph: {e}")
    st.stop()

# -------------------------------------
# 2) Initialize Streamlit session state
# -------------------------------------
if "agent_state" not in st.session_state:
    st.session_state.agent_state = AgentState(
        session_id=str(uuid.uuid4()),
        user_persona="unknown",
        audience_persona="unknown",
        current_user_tone="neutral",
        conversation_history=[],
        last_agent_question="",
        last_user_response="",
        last_evaluation="GOOD",
        current_phase=DoubleDiamondPhase.USER_PERSONA_SETUP.name,
        question_count_in_phase=0,
        final_summary="",
        key_insights=[],
        phase_transition_message=""
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# 3) Sidebar UI
# -----------------------------
with st.sidebar:
    st.header("Session")
    st.text_input("Session ID", st.session_state.agent_state['session_id'], disabled=True)

    # Show live summary pulled from our new helper
    summary = get_session_summary(st.session_state.agent_state)
    
    st.markdown(f"**Phase:** {summary['phase']}")
    if not summary['is_completed']:
        st.progress(summary["progress_pct"] / 100)
        st.caption(f"Progress: {summary['progress_text']}")
    else:
        st.success("Session Complete!")

    with st.expander("Session Details"):
        st.markdown(f"**User Persona:** {summary.get('user_persona', 'Unknown')}")
        st.markdown(f"**Audience Persona:** {summary.get('audience_persona', 'Unknown')}")
        st.markdown(f"**Key Insights:** {summary.get('key_insights_count', 0)}")

    # Reset button
    if st.button("🔁 Reset session"):
        st.session_state.clear()
        st.rerun()

# -------------------------------------
# 4) Helper to run graph
# -------------------------------------
def run_graph_turn(payload: AgentState) -> AgentState:
    """Invokes the graph and returns the final state."""
    return app.invoke(payload)

# --------------------------------
# 5) Render existing chat messages
# --------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# --------------------------------
# 6) Main App Logic
# --------------------------------

# Get first question on first run
if not st.session_state.messages:
    with st.spinner("Agent is thinking..."):
        current_state = st.session_state.agent_state
        final_state = run_graph_turn(current_state)
        
        st.session_state.agent_state = final_state
        
        # Add welcome messages
        st.session_state.messages.append(
           {"role": "assistant", "content": "Hello! I'm a Socratic agent designed to help you with ideation. We'll start by understanding you and your audience, then move through the Double Diamond."}
        )

        first_question_raw = final_state.get("last_agent_question", "Hello?")
        progress_string = get_progress_indicator(final_state)
        formatted_question = f"**{progress_string.strip('[]')}**\n\n{first_question_raw}"
        
        st.session_state.messages.append({"role": "assistant", "content": formatted_question})
        
        st.rerun()

# Handle user responses
if prompt := st.chat_input("Your response…"):
    # Show the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare payload
    current_state = st.session_state.agent_state
    current_state["last_user_response"] = prompt
    
    with st.spinner("Agent is thinking…"):
        # Run the graph
        final_state = run_graph_turn(current_state)
        
        # Save the new state
        st.session_state.agent_state = final_state

        agent_response_raw = final_state.get("last_agent_question", "...")

        if final_state.get("current_phase") == DoubleDiamondPhase.COMPLETED.name:
            formatted_response = agent_response_raw
        else:
            # Otherwise, add the progress indicator
            progress_string = get_progress_indicator(final_state)
            formatted_response = f"**{progress_string.strip('[]')}**\n\n{agent_response_raw}"

        st.session_state.messages.append({"role": "assistant", "content": formatted_response})
        
        # Rerun to show new message and update sidebar
        st.rerun()

# Disable chat input if completed
if summary['is_completed']:
    st.chat_input(disabled=True, placeholder="Session complete! Thanks for chatting.")