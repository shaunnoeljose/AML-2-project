#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import requests
import random
import logging
import uuid
import json
from enum import Enum, auto
from typing import List, TypedDict, Literal, Any, Optional, Dict
import hashlib
import joblib

# --- Firebase Imports ---
import firebase_admin
from firebase_admin import credentials, firestore

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv
load_dotenv()

import sys

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. Double Diamond Phases
# =====================================================================

class DoubleDiamondPhase(Enum):
    """Defines the phases of the Double Diamond and setup."""
    USER_PERSONA_SETUP = auto()
    AUDIENCE_PERSONA_SETUP = auto()
    DISCOVER = auto()
    DEFINE = auto()
    DEVELOP = auto()
    DELIVER = auto()
    COMPLETED = auto()


PHASE_TRANSITIONS = {
    DoubleDiamondPhase.USER_PERSONA_SETUP: DoubleDiamondPhase.AUDIENCE_PERSONA_SETUP,
    DoubleDiamondPhase.AUDIENCE_PERSONA_SETUP: DoubleDiamondPhase.DISCOVER,
    DoubleDiamondPhase.DISCOVER: DoubleDiamondPhase.DEFINE,
    DoubleDiamondPhase.DEFINE: DoubleDiamondPhase.DEVELOP,
    DoubleDiamondPhase.DEVELOP: DoubleDiamondPhase.DELIVER,
    DoubleDiamondPhase.DELIVER: DoubleDiamondPhase.COMPLETED,
}

# =====================================================================
# 2. Graph State
# =====================================================================

class AnalyticsState(TypedDict, total=False):
    """Holds analytics data."""
    total_turns: int
    tone_tracker: List[str]
    evaluation_tracker: List[str]
    phase_time_seconds: Dict[str, float]
    last_turn_ts: float                 # epoch seconds when agent asked the last question
    avg_user_latency_sec: float         # running average
    avg_user_response_chars: float      # running average length
    probing_q_count: int
    rephrasing_q_count: int
    phase_q_count: int                  # DISCOVER/DEFINE/DEVELOP/DELIVER questions
    weak_count: int
    confused_count: int
    feedback_helpful: int
    feedback_unhelpful: int


class ConversationTurn(TypedDict):
    role: Literal["user", "model"]
    text: str


class AgentState(TypedDict, total=False):
    """
    Defines the state of our agent. This is passed between nodes.
    """
    session_id: str
    current_phase: str
    question_count_in_phase: int
    user_persona: str
    audience_persona: str
    current_user_tone: str
    conversation_history: List[ConversationTurn]
    last_agent_question: str
    last_user_response: str
    last_evaluation: Literal["GOOD", "WEAK", "CONFUSED", "ERROR"]
    final_summary: str
    start_time_in_phase: float
    analytics: AnalyticsState
    feedback_log: List[Dict[str, str]]
    resume_summary: Optional[str]
    last_question_id: str
    rated_question_ids: List[str]

# =====================================================================
# 3. Configuration & Constants
# =====================================================================

QUESTIONS_PER_PHASE = 3
SELF_CORRECTION_ATTEMPTS = 2
MAX_HISTORY_TURNS = 10

SYSTEM_PROMPTS = {
    "USER_PERSONA_SETUP": (
        "You are a Socratic agent. Your very first goal is to understand who the user is "
        "so you can tailor the tone. Ask them about their role (e.g., student, marketer, "
        "researcher) or what they hope to achieve. This helps set the context. "
        "NEVER give answers or suggestions. Only ask one probing question at a time. "
        "Keep it concise and welcoming."
    ),
    "AUDIENCE_PERSONA_SETUP": (
        "You are a Socratic agent. The user has explained who they are. Now, you must guide "
        "them to define their *target audience*. This is a critical step before 'Discover'. "
        "Ask questions to help them build a clear persona of *who* they are solving this "
        "problem for. Ask about the audience's needs, behaviors, and pain points. "
        "Only ask one probing question at a time."
    ),
    "DISCOVER": (
        "You are a Socratic agent guiding a user through the 'DISCOVER' phase. "
        "The user has defined their audience. Now, help them explore the *problem space* "
        "for that audience. Focus on divergence: explore user needs, challenge assumptions, "
        "and uncover root causes. NEVER give answers. Only ask one probing question at a time."
    ),
    "DEFINE": (
        "You are a Socratic agent guiding a user through the 'DEFINE' phase. "
        "The user has finished exploring the problem. Your role is now to ask questions "
        "that help them CONVERGE. Help them synthesize their findings into a single, clear "
        "problem statement. Ask questions that reframe the problem or prioritize issues. "
        "NEVER give answers or suggestions. Only ask one probing question at a time."
    ),
    "DEVELOP": (
        "You are a Socratic agent guiding a user through the 'DEVELOP' phase. "
        "The user has a clear problem statement. Your role is to help them DIVERGE again "
        "and brainstorm solutions. Ask 'What if...' or 'How about...' questions to spark "
        "'wild' and creative ideas. NEVER give answers or your own solutions. "
        "Only ask one probing question at a time."
    ),
    "DELIVER": (
        "You are a Socratic agent guiding a user through the 'DELIVER' phase. "
        "The user has many creative solutions. Your role is to help them CONVERGE on a "
        "single, actionable solution. Ask questions about feasibility, impact, and "
        "prototyping. NEVER give answers or opinions. Only ask one probing question at a time."
    ),
    "PROBING": (
        "You are a Socratic agent. The user just gave a weak or avoidant response. "
        "Your goal is to ask a gentle, encouraging follow-up question to help them elaborate. "
        "Don't be accusatory. For example, if they said 'I don't know', ask "
        "'What part feels unclear?'. Keep your question concise and ask only one question."
    ),
    "REPHRASING": (
        "You are a Socratic agent. The user is confused by your last question. "
        "Your goal is to rephrase your original question. Make it simpler, clearer, or "
        "approach it from a different angle. Apologize briefly, e.g., "
        "'My mistake, let me try that another way.' Ask only one question."
    ),
}

# =====================================================================
# 4. LLM + Local Evaluator
# =====================================================================

USE_LOCAL_EVAL = os.getenv("USE_LOCAL_EVAL", "0") == "1"

# Confidence-aware threshold for SLM
SLM_CONF_THRESHOLD = float(os.getenv("SLM_CONF_THRESHOLD", "0.7"))

# Where to log disagreement examples for future retraining
SLM_DISAGREE_LOG = os.getenv("SLM_DISAGREE_LOG", "logs/slm_disagreements.jsonl")
SLM_DISAGREE_COLLECTION = os.getenv("SLM_DISAGREE_COLLECTION", "slm_disagreements")


class LocalEvaluator:
    """
    Wraps the trained SLM models (TF-IDF + LogisticRegression pipelines)
    produced by slm_train.py.

    Expects in repo:
      models/slm_eval/slm_evaluation.joblib
      models/slm_eval/slm_tone.joblib
    """

    def __init__(self, model_dir: str = "models/slm_eval"):
        eval_path = os.path.join(model_dir, "slm_evaluation.joblib")
        tone_path = os.path.join(model_dir, "slm_tone.joblib")

        if not os.path.exists(eval_path) or not os.path.exists(tone_path):
            raise FileNotFoundError(
                f"Could not find local eval models in {model_dir}. "
                f"Expected slm_evaluation.joblib and slm_tone.joblib."
            )

        self.eval_model = joblib.load(eval_path)
        self.tone_model = joblib.load(tone_path)

    @staticmethod
    def _pack(q: str, a: str) -> str:
        """
        Pack question+answer for the TF-IDF pipeline.
        Must match slm_train.py convention.
        """
        return f"{q or ''} [SEP] {a or ''}"

    def predict(self, question: str, response: str):
        """
        Simple label-only prediction (no confidences).
        """
        text = [self._pack(question, response)]
        eval_label = self.eval_model.predict(text)[0]
        tone_label = self.tone_model.predict(text)[0]
        return str(eval_label), str(tone_label)

    def predict_with_proba(self, question: str, response: str):
        """
        Prediction + per-label max probability.
        Returns:
            ( (eval_label, eval_conf), (tone_label, tone_conf) )
        """
        text = [self._pack(question, response)]

        eval_proba = self.eval_model.predict_proba(text)[0]
        tone_proba = self.tone_model.predict_proba(text)[0]

        eval_idx = int(eval_proba.argmax())
        tone_idx = int(tone_proba.argmax())

        eval_label = str(self.eval_model.classes_[eval_idx])
        tone_label = str(self.tone_model.classes_[tone_idx])

        eval_conf = float(eval_proba[eval_idx])
        tone_conf = float(tone_proba[tone_idx])

        return (eval_label, eval_conf), (tone_label, tone_conf)


_local_eval: Optional[LocalEvaluator] = None

if USE_LOCAL_EVAL:
    try:
        _local_eval = LocalEvaluator()
        logger.info("LocalEvaluator loaded. Using local evaluation.")
    except Exception as e:
        logger.warning(f"LocalEvaluator not available, falling back to API: {e}")
        _local_eval = None


def slm_score(question: str, response: str) -> Dict[str, Any]:
    """
    Public helper to score a (question, response) pair with the local SLM.

    Returns:
        {
          "evaluation": str,
          "tone": str,
          "evaluation_conf": float,
          "tone_conf": float
        }
    """
    if _local_eval is None:
        raise RuntimeError(
            "Local evaluator not loaded. "
            "Ensure USE_LOCAL_EVAL=1 and models/slm_eval/* joblibs exist."
        )

    (eval_label, eval_conf), (tone_label, tone_conf) = _local_eval.predict_with_proba(
        question, response
    )
    return {
        "evaluation": eval_label,
        "tone": tone_label,
        "evaluation_conf": eval_conf,
        "tone_conf": tone_conf,
    }


def _log_slm_disagreement(
    state: AgentState,
    local_eval: str,
    local_tone: str,
    local_eval_conf: float,
    local_tone_conf: float,
    api_eval: str,
    api_tone: str,
) -> None:
    """
    Log examples where SLM and Gemini disagree so you can use them
    for future retraining / calibration.
    Logs to:
      - JSONL file (SLM_DISAGREE_LOG)
      - Optionally Firestore collection SLM_DISAGREE_COLLECTION
        if SLM_FIRESTORE_LOG=1 and Firebase is initialized.
    """
    payload = {
        "ts": time.time(),
        "session_id": state.get("session_id", ""),
        "phase": state.get("current_phase", ""),
        "question": state.get("last_agent_question", ""),
        "response": state.get("last_user_response", ""),
        "local_eval": local_eval,
        "local_tone": local_tone,
        "local_eval_conf": local_eval_conf,
        "local_tone_conf": local_tone_conf,
        "api_eval": api_eval,
        "api_tone": api_tone,
    }

    # File logging
    try:
        if SLM_DISAGREE_LOG:
            os.makedirs(os.path.dirname(SLM_DISAGREE_LOG), exist_ok=True)
            with open(SLM_DISAGREE_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            logger.info("Logged SLM disagreement to file.")
    except Exception as e:
        logger.error(f"Failed to log SLM disagreement to file: {e}")

    # Optional Firestore logging
    try:
        if os.getenv("SLM_FIRESTORE_LOG", "0") == "1" and firebase_admin._apps:
            db = firestore.client()
            db.collection(SLM_DISAGREE_COLLECTION).add(payload)
            logger.info("Logged SLM disagreement to Firestore.")
    except Exception as e:
        logger.error(f"Failed to log SLM disagreement to Firestore: {e}")


# --- Gemini model ---

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logger.error("GEMINI_API_KEY not set in environment.")
    sys.exit(1)

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=api_key,
    )
    logger.info("ChatGoogleGenerativeAI initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
    sys.exit(1)

# =====================================================================
# 5. Schemas
# =====================================================================

class Evaluation(BaseModel):
    """The evaluation of the user's response and their tone."""
    evaluation: Literal["GOOD", "WEAK", "CONFUSED"] = Field(
        description="The quality of the user's response."
    )
    tone: Literal["analytical", "curious", "frustrated", "playful", "neutral"] = Field(
        description="The user's dominant emotional tone."
    )


class QuestionSelfCorrection(BaseModel):
    """Evaluation of the agent's own generated question."""
    evaluation: Literal["GOOD", "BAD"] = Field(
        description="'GOOD' if open-ended and Socratic. 'BAD' if closed-ended or leading."
    )


class Persona(BaseModel):
    """A concise, one-sentence description of a persona."""
    description: str = Field(description="A one-sentence summary of the persona.")


class RoadmapStep(BaseModel):
    """A single, actionable step in a high-level roadmap."""
    step: str = Field(description="A short, high-level step title (e.g., 'Define MVP').")
    description: str = Field(description="A brief one-sentence description of what this step involves.")


class FinalReport(BaseModel):
    """A final report containing a summary and a next-steps roadmap."""
    summary: str = Field(
        description="A concise one-paragraph summary of the key problem and the chosen solution."
    )
    roadmap: List[RoadmapStep] = Field(
        description="A 3-5 step high-level roadmap to help the user get started."
    )

# =====================================================================
# 6. Helper functions
# =====================================================================

def _invoke_llm_with_retry(chain: Any, *args, **kwargs) -> Any:
    """Wraps any LangChain .invoke() call with exponential backoff."""
    max_retries = 5
    backoff_time = 1
    for attempt in range(max_retries):
        try:
            return chain.invoke(*args, **kwargs)
        except requests.exceptions.HTTPError as http_err:
            logger.warning(
                f"HTTP error occurred: {http_err} - "
                f"Status Code: {http_err.response.status_code}"
            )
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
        prompt_parts.insert(
            0,
            f"CONTEXT: You are speaking to a *user* who is '{state['user_persona']}'.",
        )
    if state["audience_persona"] != "unknown":
        prompt_parts.insert(
            1,
            f"They are designing for an *audience* described as '{state['audience_persona']}'.",
        )

    tone_instruction = {
        "frustrated": "The user's current tone is *frustrated*. Be extra encouraging and simple in your question.",
        "playful": "The user's current tone is *playful*. You can be slightly more creative or light-hearted.",
        "analytical": "The user's current tone is *analytical*. Match this by asking a logical, structured question.",
        "curious": "The user's current tone is *curious*. Encourage this by asking a deeply probing question.",
        "neutral": "The user's tone is *neutral*. Maintain your standard helpful, Socratic tone.",
    }.get(state["current_user_tone"], "The user's tone is *neutral*.")

    prompt_parts.insert(2, f"TONE: {tone_instruction}")
    return SystemMessage(content="\n".join(prompt_parts))


def _get_history_for_api(history: List[ConversationTurn]):
    messages = []
    for msg in history[-(MAX_HISTORY_TURNS * 2):]:
        role = msg.get("role")
        text = msg.get("text", "")
        if role == "user":
            messages.append(HumanMessage(content=text))
        elif role == "model":
            messages.append(AIMessage(content=text))
    return messages


def get_progress_indicator(state: AgentState) -> str:
    """Creates a markdown progress string for the UI."""
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
        "COMPLETED": "✅",
    }
    emoji = phase_emoji.get(phase, "📊")
    phase_name = phase.replace("_", " ").title()
    return f"[ {emoji} {phase_name} | Question {count + 1}/{total} ]"


def _qid(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:10]

# =====================================================================
# 7. Graph Nodes
# =====================================================================

def start_session(state: AgentState) -> AgentState:
    """
    Node to initialize or reload the session.
    """
    logger.info(f"Graph starting in phase: {state['current_phase']}")
    if "analytics" not in state:
        state["analytics"] = {
            "total_turns": 0,
            "tone_tracker": [],
            "evaluation_tracker": [],
            "phase_time_seconds": {},
            "last_turn_ts": 0.0,
            "avg_user_latency_sec": 0.0,
            "avg_user_response_chars": 0.0,
            "probing_q_count": 0,
            "rephrasing_q_count": 0,
            "phase_q_count": 0,
            "weak_count": 0,
            "confused_count": 0,
            "feedback_helpful": 0,
            "feedback_unhelpful": 0,
        }
    if "feedback_log" not in state:
        state["feedback_log"] = []
    if "start_time_in_phase" not in state:
        state["start_time_in_phase"] = time.time()
    if "rated_question_ids" not in state:
        state["rated_question_ids"] = []
    if "last_question_id" not in state:
        state["last_question_id"] = ""

    return state


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
        atype = "probing"
    elif evaluation == "CONFUSED":
        base_prompt = SYSTEM_PROMPTS["REPHRASING"]
        atype = "rephrasing"
    else:  # GOOD or ERROR
        base_prompt = SYSTEM_PROMPTS.get(phase, SYSTEM_PROMPTS["DISCOVER"])
        atype = "phase"

    system_prompt_msg = _build_dynamic_prompt(state, base_prompt)
    history_messages = _get_history_for_api(state["conversation_history"])
    messages_for_api = [system_prompt_msg] + history_messages

    if not history_messages:
        messages_for_api.append(
            HumanMessage(
                content="Start the conversation by asking your first question based on your system prompt."
            )
        )

    question = ""
    for _ in range(SELF_CORRECTION_ATTEMPTS):
        response = _invoke_llm_with_retry(llm, messages_for_api)
        if response is None:
            break
        gen_question = response.content.strip().strip('"')

        eval_chain = (
            ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="Evaluate this Socratic question. 'GOOD' if open-ended/non-leading. "
                                "'BAD' if closed-ended/leading."
                    ),
                    HumanMessage(content=f"Evaluate: {gen_question}"),
                ]
            )
            | llm.with_structured_output(QuestionSelfCorrection)
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

    analytics = state["analytics"]
    if atype == "probing":
        analytics["probing_q_count"] += 1
    elif atype == "rephrasing":
        analytics["rephrasing_q_count"] += 1
    else:
        analytics["phase_q_count"] += 1

    analytics["last_turn_ts"] = time.time()
    qid = _qid(question)

    return {
        "last_agent_question": question,
        "last_question_id": qid,
        "conversation_history": state["conversation_history"]
        + [{"role": "model", "text": question}],
        "analytics": analytics,
    }


def process_user_response(state: AgentState) -> AgentState:
    """
    Node to process the user's response and update analytics.
    """
    response = state["last_user_response"]
    logger.info(f"Processing user response: {response}")

    analytics = state["analytics"]
    now = time.time()

    # Latency since last agent question
    if analytics.get("last_turn_ts", 0):
        dt = max(0.0, now - analytics["last_turn_ts"])
        n = max(1, analytics["total_turns"])
        analytics["avg_user_latency_sec"] = (
            (analytics["avg_user_latency_sec"] * (n - 1)) + dt
        ) / n

    # Running avg of response length
    rlen = len(response or "")
    n = max(1, analytics["total_turns"])
    analytics["avg_user_response_chars"] = (
        (analytics["avg_user_response_chars"] * (n - 1)) + rlen
    ) / n

    analytics["total_turns"] += 1

    quit_tokens = {"quit", "exit", "stop", "end", "finish", "done"}

    if (response or "").lower() in quit_tokens:
        logger.info("User requested to finish. Advancing to summary.")
        return {
            "current_phase": DoubleDiamondPhase.DELIVER.name,
            "question_count_in_phase": QUESTIONS_PER_PHASE,
            "conversation_history": state["conversation_history"]
            + [{"role": "user", "text": response}],
            "last_evaluation": "GOOD",
            "analytics": analytics,
        }

    return {
        "last_user_response": response,
        "conversation_history": state["conversation_history"]
        + [{"role": "user", "text": response}],
        "analytics": analytics,
    }


def evaluate_response(state: AgentState) -> AgentState:
    """
    Node to evaluate the user's response, detect tone, and update analytics.

    Logic:
      1. Try local SLM (if available) with predict_proba.
      2. If both evaluation & tone confidences >= SLM_CONF_THRESHOLD → trust SLM, no API call.
      3. Otherwise, call Gemini.
      4. If SLM was low-confidence and Gemini disagrees → log to file / Firestore.
    """
    logger.info("Evaluating user response.")
    analytics = state["analytics"]

    local_eval_label: Optional[str] = None
    local_tone_label: Optional[str] = None
    local_eval_conf: Optional[float] = None
    local_tone_conf: Optional[float] = None

    # --- 1. Local SLM path ---
    if _local_eval is not None:
        try:
            (eval_pair, tone_pair) = _local_eval.predict_with_proba(
                state["last_agent_question"], state["last_user_response"]
            )
            local_eval_label, local_eval_conf = eval_pair
            local_tone_label, local_tone_conf = tone_pair

            logger.info(
                f"[LocalEval] Eval={local_eval_label} (p={local_eval_conf:.3f}), "
                f"Tone={local_tone_label} (p={local_tone_conf:.3f})"
            )

            # If both predictions are confident enough, we trust the SLM and skip Gemini.
            if (
                local_eval_conf >= SLM_CONF_THRESHOLD
                and local_tone_conf >= SLM_CONF_THRESHOLD
            ):
                logger.info(
                    "Using LOCAL SLM evaluator "
                    f"(eval_conf={local_eval_conf:.3f}, tone_conf={local_tone_conf:.3f})"
                )
                logger.info("• Eval model: slm_evaluation.joblib")
                logger.info("• Tone model: slm_tone.joblib")

                q_pred = local_eval_label
                t_pred = local_tone_label

                analytics["tone_tracker"].append(str(t_pred))
                analytics["evaluation_tracker"].append(str(q_pred))
                if q_pred == "WEAK":
                    analytics["weak_count"] += 1
                elif q_pred == "CONFUSED":
                    analytics["confused_count"] += 1

                return {
                    "last_evaluation": str(q_pred),
                    "current_user_tone": str(t_pred),
                    "analytics": analytics,
                }
            else:
                logger.warning(
                    "Local SLM LOW CONFIDENCE — falling back to Gemini.\n"
                    f"    • eval_conf={local_eval_conf:.3f}\n"
                    f"    • tone_conf={local_tone_conf:.3f}\n"
                    f"    • Threshold={SLM_CONF_THRESHOLD}"
                )

        except Exception as e:
            logger.warning(f"Local evaluator failed, falling back to API: {e}")

    # --- 2. Gemini evaluation path ---
    eval_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Evaluate the user's response to a Socratic question. "
                    "1. Quality: 'GOOD' (thoughtful), 'WEAK' (short, 'I don't know'), "
                    "'CONFUSED' (off-topic). "
                    "2. Tone: 'analytical', 'curious', 'frustrated', 'playful', 'neutral'."
                )
            ),
            HumanMessage(
                content=(
                    f"Question: {state['last_agent_question']}\n\n"
                    f"Response: {state['last_user_response']}"
                )
            ),
        ]
    )
    eval_chain = eval_prompt | llm.with_structured_output(Evaluation)

    try:
        result = _invoke_llm_with_retry(eval_chain, {})
        if result is None:
            raise Exception("API call failed after retries.")

        logger.info("Using GEMINI API for evaluation")
        logger.info(f"• Gemini model: gemini-2.5-flash")
        logger.info(f"• Result Eval={result.evaluation}, Tone={result.tone}")

        analytics["tone_tracker"].append(result.tone)
        analytics["evaluation_tracker"].append(result.evaluation)

        if result.evaluation == "WEAK":
            analytics["weak_count"] += 1
        elif result.evaluation == "CONFUSED":
            analytics["confused_count"] += 1

        # --- 3. If we had local low-confidence predictions, log disagreement cases ---
        if local_eval_label is not None and local_tone_label is not None:
            if (
                local_eval_label != result.evaluation
                or local_tone_label != result.tone
            ):
                _log_slm_disagreement(
                    state,
                    local_eval_label,
                    local_tone_label,
                    local_eval_conf or 0.0,
                    local_tone_conf or 0.0,
                    result.evaluation,
                    result.tone,
                )

        return {
            "last_evaluation": result.evaluation,
            "current_user_tone": result.tone,
            "analytics": analytics,
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return {
            "last_evaluation": "ERROR",
            "current_user_tone": "neutral",
            "analytics": analytics,
        }


def advance_phase(state: AgentState) -> AgentState:
    """
    Node to advance the phase and handle persona distillation or final report.
    Also logs analytics for the completed phase.
    """
    current_phase = DoubleDiamondPhase[state["current_phase"]]
    next_phase = current_phase
    user_persona = state["user_persona"]
    audience_persona = state["audience_persona"]
    final_summary = ""
    analytics = state["analytics"]

    time_in_phase = time.time() - state["start_time_in_phase"]
    analytics["phase_time_seconds"][current_phase.name] = round(time_in_phase, 2)
    logger.info(f"Phase {current_phase.name} completed in {time_in_phase:.2f}s")

    history_str = "\n".join(
        [f"{turn['role']}: {turn['text']}" for turn in state["conversation_history"]]
    )

    distill_chain = (
        ChatPromptTemplate.from_template(
            "Based on this history, provide a concise, one-sentence summary for the request.\n\n"
            "History:\n{history}\n\nRequest: {prompt}"
        )
        | llm.with_structured_output(Persona)
    )

    if current_phase == DoubleDiamondPhase.USER_PERSONA_SETUP:
        logger.info("Advancing phase: Distilling User Persona")
        result = _invoke_llm_with_retry(
            distill_chain,
            {"history": history_str, "prompt": "Summarize the *user*."},
        )
        if result:
            user_persona = result.description
        next_phase = PHASE_TRANSITIONS[current_phase]

    elif current_phase == DoubleDiamondPhase.AUDIENCE_PERSONA_SETUP:
        logger.info("Advancing phase: Distilling Audience Persona")
        result = _invoke_llm_with_retry(
            distill_chain,
            {"history": history_str, "prompt": "Summarize the *target audience*."},
        )
        if result:
            audience_persona = result.description
        next_phase = PHASE_TRANSITIONS[current_phase]

    elif current_phase == DoubleDiamondPhase.DELIVER:
        logger.info("Advancing phase: Generating Final Report")
        next_phase = PHASE_TRANSITIONS[current_phase]

        report_prompt = (
            "You are an expert design strategist. Based on the entire conversation, "
            "provide a final report. Include: 1. A concise summary of the key problem "
            "and chosen solution. 2. A 3-5 step high-level 'Next Steps' roadmap."
            "\n\nHistory:\n{history}"
        )
        report_chain = (
            ChatPromptTemplate.from_template(report_prompt)
            | llm.with_structured_output(FinalReport)
        )
        final_report = _invoke_llm_with_retry(report_chain, {"history": history_str})

        if final_report:
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
            "last_agent_question": final_summary,
            "analytics": analytics,
            "start_time_in_phase": time.time(),
        }

    else:
        logger.info(f"Advancing from {current_phase.name}...")
        next_phase = PHASE_TRANSITIONS.get(current_phase, DoubleDiamondPhase.COMPLETED)

    return {
        "current_phase": next_phase.name,
        "question_count_in_phase": 0,
        "user_persona": user_persona,
        "audience_persona": audience_persona,
        "current_user_tone": "neutral",
        "final_summary": final_summary,
        "analytics": analytics,
        "start_time_in_phase": time.time(),
    }


def update_question_count(state: AgentState):
    """Simple node to update the question count before looping."""
    new_count = state["question_count_in_phase"] + 1
    logger.info(f"Incrementing question count to {new_count}")
    return {"question_count_in_phase": new_count}

# =====================================================================
# 8. Graph Edges & Builder
# =====================================================================

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
            "process_user_response": "process_user_response",
        },
    )

    builder.add_edge("generate_question", END)
    builder.add_edge("process_user_response", "evaluate_response")

    builder.add_conditional_edges(
        "evaluate_response",
        decide_next_step,
        {
            "generate_question": "generate_question",
            "advance_phase": "advance_phase",
            "continue_phase": "update_question_count",
        },
    )

    builder.add_edge("update_question_count", "generate_question")

    builder.add_conditional_edges(
        "advance_phase",
        check_if_done,
        {
            "generate_question": "generate_question",
            END: END,
        },
    )

    logger.info("Compiling LangGraph...")
    return builder.compile()

# =====================================================================
# 9. Firestore Persistence & Feedback Helpers
# =====================================================================

DB_COLLECTION = "socratic_sessions"
ANALYTICS_COLLECTION = "socratic_analytics"
FEEDBACK_EVENTS_COLLECTION = "socratic_feedback_events"


def _normalize_for_firestore(state: AgentState) -> AgentState:
    norm = dict(state)

    fixed_history: List[ConversationTurn] = []
    for item in state.get("conversation_history", []):
        if isinstance(item, dict) and "role" in item and "text" in item:
            fixed_history.append({"role": item["role"], "text": item["text"]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            role, text = item
            fixed_history.append({"role": str(role), "text": str(text)})
        else:
            fixed_history.append({"role": "model", "text": str(item)})
    norm["conversation_history"] = fixed_history

    norm["final_summary"] = norm.get("final_summary") or ""
    norm["resume_summary"] = norm.get("resume_summary") or ""
    norm["rated_question_ids"] = norm.get("rated_question_ids", [])
    norm["last_question_id"] = norm.get("last_question_id", "")

    norm.setdefault("analytics", {})
    a = norm["analytics"]
    a.setdefault("total_turns", 0)
    a.setdefault("tone_tracker", [])
    a.setdefault("evaluation_tracker", [])
    a.setdefault("phase_time_seconds", {})
    a.setdefault("last_turn_ts", 0.0)
    a.setdefault("avg_user_latency_sec", 0.0)
    a.setdefault("avg_user_response_chars", 0.0)
    a.setdefault("probing_q_count", 0)
    a.setdefault("rephrasing_q_count", 0)
    a.setdefault("phase_q_count", 0)
    a.setdefault("weak_count", 0)
    a.setdefault("confused_count", 0)
    a.setdefault("feedback_helpful", 0)
    a.setdefault("feedback_unhelpful", 0)

    phases = a.get("phase_time_seconds", {})
    a["phase_time_seconds"] = {
        str(k): float(v) if isinstance(v, (int, float)) else 0.0
        for k, v in phases.items()
    }

    return norm


# def initialize_firestore():
#     """
#     Initializes the Firebase Admin SDK for Hugging Face Spaces.

#     Expects FIREBASE_SERVICE_ACCOUNT_JSON in the environment,
#     containing the full service account JSON (not a file path).
#     """
#     json_str = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
#     if not json_str:
#         logger.error(
#             "FIREBASE_SERVICE_ACCOUNT_JSON is not set. "
#             "In your Hugging Face Space, create a secret with this name "
#             "and paste the full Firebase service account JSON."
#         )
#         return None

#     try:
#         cred_dict = json.loads(json_str)
#         cred = credentials.Certificate(cred_dict)
#     except Exception as e:
#         logger.error(f"Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: {e}")
#         return None

#     try:
#         if not firebase_admin._apps:
#             firebase_admin.initialize_app(cred)
#             logger.info("Firebase Admin SDK initialized from FIREBASE_SERVICE_ACCOUNT_JSON.")
#         return firestore.client()
#     except Exception as e:
#         logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
#         return None


# --- add at the top if not already there ---
import json
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore
import os
import logging

logger = logging.getLogger(__name__)


def initialize_firestore():
    """
    Local + cloud friendly Firestore init.

    Priority:
    1) If FIREBASE_SERVICE_ACCOUNT_JSON is set -> treat it as *JSON string* (HF Spaces).
    2) Otherwise, look for a local JSON file named 'firestore-service-account.json'
       in a few sensible locations:
         - same folder as socradesign_logic.py
         - current working directory (where you run `streamlit run ...`)
         - repo root (parent of this file's folder)
    """
    # 1) HF / cloud: env var with raw JSON
    env_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
    cred_dict = None

    if env_json:
        try:
            cred_dict = json.loads(env_json)
            logger.info("Initializing Firestore from FIREBASE_SERVICE_ACCOUNT_JSON env var")
        except Exception as e:
            logger.error(f"Env FIREBASE_SERVICE_ACCOUNT_JSON is set but not valid JSON: {e}")

    # 2) Local: look for a file
    if cred_dict is None:
        here = Path(__file__).resolve().parent
        cwd = Path.cwd()
        candidates = [
            here / "firestore-service-account.json",
            cwd / "firestore-service-account.json",
            here.parent / "firestore-service-account.json",
        ]

        json_path = None
        for p in candidates:
            if p.exists():
                json_path = p
                break

        if not json_path:
            logger.error(
                "Could not find firestore-service-account.json in any of these locations:\n"
                + "\n".join(str(p) for p in candidates)
            )
            return None

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                cred_dict = json.load(f)
            logger.info(f"Initializing Firestore from local file: {json_path}")
        except Exception as e:
            logger.error(f"Found {json_path} but failed to load/parse it as JSON: {e}")
            return None

    # 3) Initialize Firebase Admin SDK
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase Admin SDK initialized")
        return firestore.client()
    except Exception as e:
        logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        return None


def get_initial_state(session_id: str) -> AgentState:
    """
    Creates a brand new, empty state for a new session.
    """
    start_time = time.time()
    return {
        "session_id": session_id,
        "current_phase": DoubleDiamondPhase.USER_PERSONA_SETUP.name,
        "question_count_in_phase": 0,
        "user_persona": "unknown",
        "audience_persona": "unknown",
        "current_user_tone": "neutral",
        "conversation_history": [],
        "last_agent_question": "",
        "last_user_response": "",
        "last_evaluation": "GOOD",  # Start with GOOD to generate first Q
        "final_summary": "",
        "start_time_in_phase": start_time,
        "analytics": {
            "total_turns": 0,
            "tone_tracker": [],
            "evaluation_tracker": [],
            "phase_time_seconds": {},
            "last_turn_ts": 0.0,
            "avg_user_latency_sec": 0.0,
            "avg_user_response_chars": 0.0,
            "probing_q_count": 0,
            "rephrasing_q_count": 0,
            "phase_q_count": 0,
            "weak_count": 0,
            "confused_count": 0,
            "feedback_helpful": 0,
            "feedback_unhelpful": 0,
        },
        "feedback_log": [],
        "resume_summary": "",
        "last_question_id": "",
        "rated_question_ids": [],
    }


def save_state_to_firestore(db: Any, session_id: str, state: AgentState) -> None:
    """
    Serializes the AgentState to a Firestore document.
    """
    try:
        doc_ref = db.collection(DB_COLLECTION).document(session_id)
        doc_ref.set(_normalize_for_firestore(state))
        logger.info(f"Session state saved to Firestore (ID: {session_id})")
    except Exception as e:
        logger.error(f"Failed to save state to Firestore: {e}")


def load_state_from_firestore(db: Any, session_id: str) -> Optional[AgentState]:
    """
    Deserializes the AgentState from a Firestore document.
    """
    try:
        doc_ref = db.collection(DB_COLLECTION).document(session_id)
        doc = doc_ref.get()

        if doc.exists:
            loaded = doc.to_dict()
            logger.info(f"Session state loaded from Firestore (ID: {session_id})")
            return _normalize_for_firestore(loaded)
        else:
            logger.warning(f"No session found in Firestore for ID: {session_id}")
            return None
    except Exception as e:
        logger.error(f"Failed to load state from Firestore: {e}")
        return None


def save_analytics_snapshot(db: Any, state: AgentState) -> None:
    """Write a flat, dashboard-friendly analytics document."""
    try:
        s = _normalize_for_firestore(state)
        a = s["analytics"]
        doc = {
            "session_id": s["session_id"],
            "current_phase": s["current_phase"],
            "total_turns": a["total_turns"],
            "weak_count": a["weak_count"],
            "confused_count": a["confused_count"],
            "probing_q_count": a["probing_q_count"],
            "rephrasing_q_count": a["rephrasing_q_count"],
            "phase_q_count": a["phase_q_count"],
            "avg_user_latency_sec": a["avg_user_latency_sec"],
            "avg_user_response_chars": a["avg_user_response_chars"],
            "tone_distribution": {
                t: a["tone_tracker"].count(t) for t in set(a["tone_tracker"])
            },
            "evaluation_distribution": {
                e: a["evaluation_tracker"].count(e) for e in set(a["evaluation_tracker"])
            },
            "phase_time_seconds": a["phase_time_seconds"],
            "feedback_helpful": a["feedback_helpful"],
            "feedback_unhelpful": a["feedback_unhelpful"],
            "updated_at": time.time(),
        }
        db.collection(ANALYTICS_COLLECTION).document(s["session_id"]).set(doc)
        logger.info("Analytics snapshot saved.")
    except Exception as e:
        logger.error(f"Failed to save analytics snapshot: {e}")


def save_question_feedback_event(db: Any, state: AgentState, rating: str, comment: str = ""):
    """
    Store a single per-question feedback event (thumbs-up/down, etc.).
    """
    try:
        db.collection(FEEDBACK_EVENTS_COLLECTION).add(
            {
                "session_id": state.get("session_id", ""),
                "question_id": state.get("last_question_id", ""),
                "question": state.get("last_agent_question", ""),
                "rating": rating,
                "comment": comment,
                "phase": state.get("current_phase", ""),
                "created_at": time.time(),
            }
        )
        logger.info("Feedback event saved.")
    except Exception as e:
        logger.error(f"Failed to save feedback event: {e}")


def _generate_resume_summary(history: List[ConversationTurn]) -> Optional[str]:
    """
    Generate a short recap paragraph for a returning user.
    """
    if not history:
        return None

    history_str = "\n".join([f"{t['role']}: {t['text']}" for t in history])
    prompt = (
        "You are a helpful assistant. Summarize the following design-thinking "
        "conversation in 3-4 sentences so the user can quickly remember "
        "where they left off:\n\n{history}"
    )

    chain = ChatPromptTemplate.from_template(prompt) | llm
    result = _invoke_llm_with_retry(chain, {"history": history_str})
    if result is None:
        return None
    return result.content.strip()

# =====================================================================
# 10. CLI entry point (optional local testing)
# =====================================================================

if __name__ == "__main__":
    # This block is mainly for local debugging; on Hugging Face you will use app.py.
    db = initialize_firestore()

    if not db:
        print("\nFATAL: Could not connect to Firestore. Check FIREBASE_SERVICE_ACCOUNT_JSON.")
        sys.exit(1)

    logger.info("--- Socratic Agent CLI Test (Firestore) ---")

    session_id = input(
        "Enter a session ID to resume, or press Enter for a new one: "
    ).strip()
    if not session_id:
        session_id = f"cli-session-{uuid.uuid4().hex[:8]}"
        logger.info(f"Creating new session: {session_id}")

    agent_state = load_state_from_firestore(db, session_id)

    if agent_state:
        logger.info(f"Resuming session {session_id}")
        agent_state["last_user_response"] = ""

        resume_summary = agent_state.get("resume_summary")
        if not resume_summary and agent_state.get("conversation_history"):
            resume_summary = _generate_resume_summary(agent_state["conversation_history"])
            if resume_summary:
                agent_state["resume_summary"] = resume_summary
                save_state_to_firestore(db, session_id, agent_state)

        if resume_summary:
            print("\n--- 🚀 Welcome back! ---")
            print(f"Here's a quick recap of your last session:\n{resume_summary}")
            print("-------------------------\n")
    else:
        logger.info(f"Starting new session {session_id}")
        agent_state = get_initial_state(session_id)
        save_state_to_firestore(db, session_id, agent_state)
        save_analytics_snapshot(db, agent_state)

    socratic_graph = build_graph()

    if not agent_state["conversation_history"] or not agent_state["last_user_response"]:
        config: Dict[str, Any] = {}
        result_state = socratic_graph.invoke(agent_state, config)
        agent_state.update(result_state)
        print(f"\n{get_progress_indicator(agent_state)}")
        print(f"Agent: {agent_state['last_agent_question']}")
        save_state_to_firestore(db, session_id, agent_state)
        save_analytics_snapshot(db, agent_state)

    while agent_state["current_phase"] != DoubleDiamondPhase.COMPLETED.name:
        try:
            user_input = input("You: ")

            agent_state["last_user_response"] = user_input
            config = {}
            result_state = socratic_graph.invoke(agent_state, config)
            agent_state.update(result_state)

            if agent_state["current_phase"] != DoubleDiamondPhase.COMPLETED.name:
                print(f"\n{get_progress_indicator(agent_state)}")
                print(f"Agent: {agent_state['last_agent_question']}")

            save_state_to_firestore(db, session_id, agent_state)
            save_analytics_snapshot(db, agent_state)

        except KeyboardInterrupt:
            print(
                "\n\nSession paused. Run script again and enter the same ID to resume."
            )
            break
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            break

    if agent_state["current_phase"] == DoubleDiamondPhase.COMPLETED.name:
        logger.info("Session complete.")
        print("\nSession complete. Final summary:")
        print(agent_state.get("final_summary", ""))
