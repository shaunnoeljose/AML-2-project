import os
import time
import requests
import random
import logging
import uuid
from enum import Enum, auto
from typing import List, TypedDict, Literal, Any, Optional, Dict

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
load_dotenv()

import sys

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 1. Define Phases (Unchanged) ---

class DoubleDiamondPhase(Enum):
    """Defines the phases of the Double Diamond and setup."""
    USER_PERSONA_SETUP = auto()
    AUDIENCE_PERSONA_SETUP = auto()
    DISCOVER = auto()
    DEFINE = auto()
    DEVELOP = auto()
    DELIVER = auto()
    COMPLETED = auto()

# Phase transition mapping
PHASE_TRANSITIONS = {
    DoubleDiamondPhase.USER_PERSONA_SETUP: DoubleDiamondPhase.AUDIENCE_PERSONA_SETUP,
    DoubleDiamondPhase.AUDIENCE_PERSONA_SETUP: DoubleDiamondPhase.DISCOVER,
    DoubleDiamondPhase.DISCOVER: DoubleDiamondPhase.DEFINE,
    DoubleDiamondPhase.DEFINE: DoubleDiamondPhase.DEVELOP,
    DoubleDiamondPhase.DEVELOP: DoubleDiamondPhase.DELIVER,
    DoubleDiamondPhase.DELIVER: DoubleDiamondPhase.COMPLETED,
}

# --- 2. Define Graph State ---

class AgentState(TypedDict):
    """
    Defines the state of our agent. This is passed between nodes.
    """
    session_id: str
    current_phase: str
    question_count_in_phase: int
    user_persona: str
    audience_persona: str
    current_user_tone: str
    conversation_history: List[tuple[str, str]]
    last_agent_question: str
    last_user_response: str
    last_evaluation: Literal["GOOD", "WEAK", "CONFUSED", "ERROR"]
    final_summary: str

# --- 3. Configuration & Constants ---

QUESTIONS_PER_PHASE = 3
SELF_CORRECTION_ATTEMPTS = 2
MAX_HISTORY_TURNS = 10 
SYSTEM_PROMPTS = {
    "USER_PERSONA_SETUP": (
        "You are a Socratic agent. Your very first goal is to understand who the user is so you can tailor the tone. "
        "Ask them about their role (e.g., student, marketer, researcher) or what they hope to achieve. "
        "This helps set the context. "
        "NEVER give answers or suggestions. Only ask one probing question at a time. Keep it concise and welcoming."
    ),
    "AUDIENCE_PERSONA_SETUP": (
        "You are a Socratic agent. The user has explained who they are. Now, you must guide them to define their *target audience*. "
        "This is a critical step before 'Discover'. "
        "Ask questions to help them build a clear persona of *who* they are solving this problem for. "
        "Ask about the audience's needs, behaviors, and pain points. Only ask one probing question at a time."
    ),
    "DISCOVER": (
        "You are a Socratic agent guiding a user through the 'DISCOVER' phase. "
        "The user has defined their audience. Now, help them explore the *problem space* for that audience. "
        "Focus on divergence: explore user needs, challenge assumptions, and uncover root causes. "
        "NEVER give answers. Only ask one probing question at a time."
    ),
    "DEFINE": (
        "You are a Socratic agent guiding a user through the 'DEFINE' phase. "
        "The user has finished exploring the problem. Your role is now to ask questions that help them CONVERGE. "
        "Help them synthesize their findings into a single, clear problem statement. "
        "Ask questions that reframe the problem or prioritize issues. "
        "NEVER give answers or suggestions. Only ask one probing question at a time."
    ),
    "DEVELOP": (
        "You are a Socratic agent guiding a user through the 'DEVELOP' phase. "
        "The user has a clear problem statement. Your role is to help them DIVERGE again and brainstorm solutions. "
        "Ask 'What if...' or 'How about...' questions to spark 'wild' and creative ideas. "
        "NEVER give answers or your own solutions. Only ask one probing question at a time."
    ),
    "DELIVER": (
        "You are a Socratic agent guiding a user through the 'DELIVER' phase. "
        "The user has many creative solutions. Your role is to help them CONVERGE on a single, actionable solution. "
        "Ask questions about feasibility, impact, and prototyping. "
        "NEVER give answers or opinions. Only ask one probing question at a time."
    ),
    "PROBING": (
        "You are a Socratic agent. The user just gave a weak or avoidant response. "
        "Your goal is to ask a gentle, encouraging follow-up question to help them elaborate. "
        "Don't be accusatory. For example, if they said 'I don't know', ask 'What part feels unclear?'. "
        "Keep your question concise and ask only one question."
    ),
    "REPHRASING": (
        "You are a Socratic agent. The user is confused by your last question. "
        "Your goal is to rephrase your original question. "
        "Make it simpler, clearer, or approach it from a different angle. "
        "Apologize briefly, e.g., 'My mistake, let me try that another way.' "
        "Ask only one question."
    )
}


# --- 4. Define LLM and Schemas ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    error_message = "FATAL ERROR: 'GEMINI_API_KEY' not found in your .env file or environment."
    logger.error(error_message)
    print(f"\n{error_message}\n")
    print("Please make sure your .env file is in the same directory as your script.")
    sys.exit(1)

llm = None
try:
    # We explicitly pass the key to the google_api_key parameter
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", # Using the model you confirmed works
        temperature=0.7, 
        google_api_key=api_key
    )
    
    logger.info("ChatGoogleGenerAI initialized successfully.")

except Exception as e:
    # This error message will work in the CLI/notebook
    error_message = f"FATAL ERROR: Failed to initialize Gemini, even with API key. Error: {e}"
    logger.error(error_message)
    print(f"\n{error_message}\n")
    sys.exit(1)

# --- Schemas for structured output ---

class Evaluation(BaseModel):
    """The evaluation of the user's response and their tone."""
    evaluation: Literal["GOOD", "WEAK", "CONFUSED"] = Field(description="The quality of the user's response.")
    tone: Literal["analytical", "curious", "frustrated", "playful", "neutral"] = Field(description="The user's dominant emotional tone.")

class QuestionSelfCorrection(BaseModel):
    """Evaluation of the agent's own generated question."""
    evaluation: Literal["GOOD", "BAD"] = Field(description="'GOOD' if open-ended and Socratic. 'BAD' if closed-ended or leading.")

class Persona(BaseModel):
    """A concise, one-sentence description of a persona."""
    description: str = Field(description="A one-sentence summary of the persona.")

class RoadmapStep(BaseModel):
    """A single, actionable step in a high-level roadmap."""
    step: str = Field(description="A short, high-level step title (e.g., 'Define MVP').")
    description: str = Field(description="A brief one-sentence description of what this step involves.")

class FinalReport(BaseModel):
    """A final report containing a summary and a next-steps roadmap."""
    summary: str = Field(description="A concise one-paragraph summary of the key problem and the chosen solution.")
    roadmap: List[RoadmapStep] = Field(description="A 3-5 step high-level roadmap to help the user get started.")


# --- 5. Helper functions ---

def _invoke_llm_with_retry(chain: Any, *args, **kwargs) -> Any:
    """Wraps any LangChain .invoke() call with exponential backoff."""
    max_retries = 5
    backoff_time = 1 
    for attempt in range(max_retries):
        try:
            return chain.invoke(*args, **kwargs)
        except requests.exceptions.HTTPError as http_err:
            logger.warning(f"HTTP error occurred: {http_err} - Status Code: {http_err.response.status_code}")
            if 400 <= http_err.response.status_code < 500:
                logger.error("API access issue (4xx). Aborting.")
                return None
        except Exception as e:
            logger.warning(f"An error occurred during LLM invoke: {e}")
        
        if attempt < max_retries - 1:
            sleep_time = backoff_time + random.uniform(0.1, 0.5)
            logger.info(f"Retrying in {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
            backoff_time *= 2
        else:
            logger.error("Max retries reached for LLM invoke.")
            return None

def _build_dynamic_prompt(state: AgentState, base_prompt: str) -> SystemMessage:
    """Prepends the base system prompt with persona and tone context."""
    prompt_parts = [base_prompt]
    if state["user_persona"] != "unknown":
        prompt_parts.insert(0, f"CONTEXT: You are speaking to a *user* who is '{state['user_persona']}'.")
    if state["audience_persona"] != "unknown":
        prompt_parts.insert(1, f"They are designing for an *audience* described as '{state['audience_persona']}'.")
    
    tone_instruction = {
        "frustrated": "The user's current tone is *frustrated*. Be extra encouraging and simple in your question.",
        "playful": "The user's current tone is *playful*. You can be slightly more creative or light-hearted.",
        "analytical": "The user's current tone is *analytical*. Match this by asking a logical, structured question.",
        "curious": "The user's current tone is *curious*. Encourage this by asking a deeply probing question.",
        "neutral": "The user's tone is *neutral*. Maintain your standard helpful, Socratic tone."
    }.get(state["current_user_tone"], "The user's tone is *neutral*.")
    
    prompt_parts.insert(2, f"TONE: {tone_instruction}")
    return SystemMessage(content="\n".join(prompt_parts))

def _get_history_for_api(history: List[tuple[str, str]]):
    """Formats history for the LLM, pruning if necessary."""
    messages = []
    for role, text in history[-(MAX_HISTORY_TURNS * 2):]:
        if role == "user":
            messages.append(HumanMessage(content=text))
        elif role == "model":
            messages.append(AIMessage(content=text))
    return messages

def get_progress_indicator(state: AgentState) -> str:
    """ Creates a markdown progress string for the UI. """
    phase = state["current_phase"]
    count = state["question_count_in_phase"]
    total = QUESTIONS_PER_PHASE

    phase_emoji = {
        "USER_PERSONA_SETUP": "👤",
        "AUDIENCE_PERSONA_SETUP": "🎯",
        "DISCOVER": "🔍",
        "DEFINE": "📌",
        "DEVELOP": "💡",
        "DELIVER": "🚀",
        "COMPLETED": "✅"
    }
    emoji = phase_emoji.get(phase, "📊")
    phase_name = phase.replace("_", " ").title()
    # Format for plain text console
    return f"[ {emoji} {phase_name} | Question {count + 1}/{total} ]"

# --- 6. Define Graph Nodes ---

def start_session(state: AgentState) -> AgentState:
    """
    Node to initialize or reload the session.
    """
    logger.info(f"Graph starting in phase: {state['current_phase']}")
    return {}

def generate_question(state: AgentState) -> AgentState:
    """
    Generates the next question based on the current phase and evaluation.
    """
    phase = state["current_phase"]
    evaluation = state["last_evaluation"]
    logger.info(f"Generating question for phase '{phase}' (Last eval: {evaluation})")

    if phase == DoubleDiamondPhase.COMPLETED.name:
        return {"last_agent_question": "Session is complete."}

    if evaluation == "WEAK":
        base_prompt = SYSTEM_PROMPTS["PROBING"]
    elif evaluation == "CONFUSED":
        base_prompt = SYSTEM_PROMPTS["REPHRASING"]
    else: # GOOD or ERROR
        base_prompt = SYSTEM_PROMPTS.get(phase, SYSTEM_PROMPTS["DISCOVER"])

    system_prompt_msg = _build_dynamic_prompt(state, base_prompt)
    history_messages = _get_history_for_api(state["conversation_history"])
    
    messages_for_api = [system_prompt_msg] + history_messages
    
    if not history_messages:
        messages_for_api.append(HumanMessage(content="Start the conversation by asking your first question based on your system prompt."))
    
    question = ""
    for _ in range(SELF_CORRECTION_ATTEMPTS):
        
        response = _invoke_llm_with_retry(llm, messages_for_api)

        if response is None:
            break 
        
        gen_question = response.content.strip().strip('"')

        eval_chain = (
            ChatPromptTemplate.from_messages([
                SystemMessage(content="Evaluate this Socratic question. 'GOOD' if open-ended/non-leading. 'BAD' if closed-ended/leading."),
                HumanMessage(content=f"Evaluate: {gen_question}")
            ]) | llm.with_structured_output(QuestionSelfCorrection)
        )
        correction = _invoke_llm_with_retry(eval_chain, {})
        
        if correction and correction.evaluation == "GOOD":
            question = gen_question
            break
        else:
            logger.warning("Agent self-correction: Regenerating question")
    
    if not question:
        question = "Can you tell me more?"
    
    logger.info(f"Agent asks: {question}")
    
    return {
        "last_agent_question": question,
        "conversation_history": state["conversation_history"] + [("model", question)]
    }

def process_user_response(state: AgentState) -> AgentState:
    """
    Node to process the user's response from the state and add it to history.
    """
    response = state["last_user_response"]
    logger.info(f"Processing user response: {response}")

    quit_tokens = {"quit", "exit", "stop", "end", "finish", "done"}
    if response.lower() in quit_tokens:
        logger.info("User requested to finish. Advancing to summary.")
        return {
            "current_phase": DoubleDiamondPhase.DELIVER.name,
            "question_count_in_phase": QUESTIONS_PER_PHASE,
            "conversation_history": state["conversation_history"] + [("user", response)],
            "last_evaluation": "GOOD"
        }
        
    return {
        "last_user_response": response,
        "conversation_history": state["conversation_history"] + [("user", response)]
    }

def evaluate_response(state: AgentState) -> AgentState:
    """
    Node to evaluate the user's response and detect tone.
    """
    logger.info("Evaluating user response.")
    eval_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content="Evaluate the user's response to a Socratic question. "
                    "1. Quality: 'GOOD' (thoughtful), 'WEAK' (short, 'I don't know'), 'CONFUSED' (off-topic). "
                    "2. Tone: 'analytical', 'curious', 'frustrated', 'playful', 'neutral'."
        ),
        HumanMessage(content=f"Question: {state['last_agent_question']}\n\nResponse: {state['last_user_response']}")
    ])
    eval_chain = eval_prompt | llm.with_structured_output(Evaluation)
    
    try:
        result = _invoke_llm_with_retry(eval_chain, {})
        if result is None:
            raise Exception("API call failed after retries.")

        logger.info(f"Evaluation: {result.evaluation}, Tone: {result.tone}")
        return {
            "last_evaluation": result.evaluation,
            "current_user_tone": result.tone
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return {"last_evaluation": "ERROR", "current_user_tone": "neutral"}

def advance_phase(state: AgentState) -> AgentState:
    """
    Node to advance the phase and handle persona distillation or final report.
    """
    current_phase = DoubleDiamondPhase[state["current_phase"]]
    next_phase = current_phase
    user_persona = state["user_persona"]
    audience_persona = state["audience_persona"]
    final_summary = ""
    
    history_str = "\n".join([f"{r}: {t}" for r, t in state["conversation_history"]])

    distill_chain = (
        ChatPromptTemplate.from_template(
            "Based on this history, provide a concise, one-sentence summary for the request.\n\n"
            "History:\n{history}\n\nRequest: {prompt}"
        ) | llm.with_structured_output(Persona)
    )

    if current_phase == DoubleDiamondPhase.USER_PERSONA_SETUP:
        logger.info("Advancing phase: Distilling User Persona")
        result = _invoke_llm_with_retry(distill_chain, {"history": history_str, "prompt": "Summarize the *user*."})
        if result:
            user_persona = result.description
            logger.info(f"User Persona set to: {user_persona}")
        next_phase = PHASE_TRANSITIONS[current_phase]

    elif current_phase == DoubleDiamondPhase.AUDIENCE_PERSONA_SETUP:
        logger.info("Advancing phase: Distilling Audience Persona")
        result = _invoke_llm_with_retry(distill_chain, {"history": history_str, "prompt": "Summarize the *target audience*."})
        if result:
            audience_persona = result.description
            logger.info(f"Audience Persona set to: {audience_persona}")
        next_phase = PHASE_TRANSITIONS[current_phase]
        
    elif current_phase == DoubleDiamondPhase.DELIVER:
        logger.info("Advancing phase: Generating Final Report")
        next_phase = PHASE_TRANSITIONS[current_phase]
        
        report_prompt = (
            "You are an expert design strategist. Based on the entire conversation, provide a final report. "
            "Include: 1. A concise summary of the key problem and chosen solution. "
            "2. A 3-5 step high-level 'Next Steps' roadmap."
            "\n\nHistory:\n{history}"
        )
        report_chain = ChatPromptTemplate.from_template(report_prompt) | llm.with_structured_output(FinalReport)
        final_report = _invoke_llm_with_retry(report_chain, {"history": history_str})
        
        if final_report:
            # Format for Streamlit's markdown
            summary = f"### 🎉 Session Summary\n\n{final_report.summary}\n\n"
            if final_report.roadmap and isinstance(final_report.roadmap, list):
                summary += "### 🗺️ Recommended Roadmap\n\n"
                for i, step in enumerate(final_report.roadmap, 1):
                    summary += f"{i}. **{step.step}**: {step.description}\n"
            final_summary = summary
        else:
            final_summary = "### ⚠️ Error\n\nCould not generate final summary."
        
        return {
            "current_phase": next_phase.name,
            "question_count_in_phase": 0,
            "user_persona": user_persona,
            "audience_persona": audience_persona,
            "current_user_tone": "neutral",
            "final_summary": final_summary,
            "last_agent_question": final_summary # This puts the summary in the chat
        }

    else:
        # Just advance the phase
        logger.info(f"Advancing from {current_phase.name}...")
        next_phase = PHASE_TRANSITIONS.get(current_phase, DoubleDiamondPhase.COMPLETED)

    return {
        "current_phase": next_phase.name,
        "question_count_in_phase": 0,
        "user_persona": user_persona,
        "audience_persona": audience_persona,
        "current_user_tone": "neutral",
        "final_summary": final_summary
    }

def update_question_count(state: AgentState):
    """Simple node to update the question count before looping."""
    new_count = state["question_count_in_phase"] + 1
    logger.info(f"Incrementing question count to {new_count}")
    return {"question_count_in_phase": new_count}

# --- 7. Define Graph Edges (Routing Logic) ---

def _start_router(state: AgentState) -> str:
    """Decide where to go after start_session."""
    if state.get("last_user_response"):
        logger.info("Routing: has_user_response -> process_user_response")
        return "process_user_response"
    logger.info("Routing: no_user_response -> generate_question")
    return "generate_question"

def decide_next_step(state: AgentState):
    """Conditional edge to route after evaluation."""
    evaluation = state["last_evaluation"]
    
    if evaluation == "WEAK" or evaluation == "CONFUSED":
        logger.info("Routing: eval_weak_or_confused -> generate_question")
        return "generate_question"
    
    count = state["question_count_in_phase"] + 1
    
    if count >= QUESTIONS_PER_PHASE:
        logger.info("Routing: eval_good_and_phase_complete -> advance_phase")
        return "advance_phase"
    else:
        logger.info("Routing: eval_good_and_continue_phase -> update_question_count")
        return "continue_phase"

def check_if_done(state: AgentState):
    """Conditional edge to check if the process is complete."""
    if state["current_phase"] == DoubleDiamondPhase.COMPLETED.name:
        logger.info("Routing: phase_completed -> END")
        return END
    logger.info("Routing: phase_not_completed -> generate_question")
    return "generate_question"

# --- 8. Build and Compile the Graph ---

def build_graph():
    """
    Builds the Socratic agent graph.
    """
    builder = StateGraph(AgentState)

    builder.add_node("start_session", start_session)
    builder.add_node("generate_question", generate_question)
    builder.add_node("process_user_response", process_user_response)
    builder.add_node("evaluate_response", evaluate_response)
    builder.add_node("advance_phase", advance_phase)
    builder.add_node("update_question_count", update_question_count)

    builder.set_entry_point("start_session")

    builder.add_conditional_edges(
        "start_session",
        _start_router,
        {
            "generate_question": "generate_question",
            "process_user_response": "process_user_response"
        }
    )
    
    builder.add_edge("generate_question", END)
    builder.add_edge("process_user_response", "evaluate_response")
    
    builder.add_conditional_edges(
        "evaluate_response",
        decide_next_step,
        {
            "generate_question": "generate_question",
            "advance_phase": "advance_phase",
            "continue_phase": "update_question_count"
        }
    )
    
    builder.add_edge("update_question_count", "generate_question")
    
    builder.add_conditional_edges(
        "advance_phase",
        check_if_done,
        {
            "generate_question": "generate_question",
            END: END
        }
    )

    logger.info("Compiling LangGraph...")
    return builder.compile()

# --- 9. NEW: Helper for Streamlit Sidebar ---

def get_session_summary(state: AgentState) -> dict:
    """
    Pulls summary data from the AgentState dict for the UI.
    This replaces the database-dependent version.
    """
    phase_raw = state.get("current_phase", "UNKNOWN")
    phase_name = phase_raw.replace("_", " ").title()
    count = state.get("question_count_in_phase", 0)
    total = QUESTIONS_PER_PHASE
    
    # Calculate progress
    pct = 0
    if phase_raw != "COMPLETED":
         pct = max(0, min(100, int((min(count, total) / max(total, 1)) * 100)))

    return {
        "phase": phase_name,
        "progress_pct": pct,
        "progress_text": f"{count}/{total} Questions",
        "user_persona": state.get("user_persona", "Unknown"),
        "audience_persona": state.get("audience_persona", "Unknown"),
        "key_insights_count": len(state.get("key_insights", [])),
        "is_completed": (phase_raw == "COMPLETED")
    }
# --- 9. Command-Line Test Runner (REMOVED) ---
# (main_demo function and __name__ == "__main__" block are removed)