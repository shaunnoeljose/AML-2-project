# 🐊 SocraDesign: AI-Powered Socratic Ideation Agent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-FF4B4B.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-orange)](https://huggingface.co/spaces)

**SocraDesign** is an intelligent conversational agent designed to scaffold the creative ideation process. By strictly adhering to the **Double Diamond** design framework (Discover, Define, Develop, Deliver), it uses Socratic questioning to guide users from vague concepts to actionable, prototypable solutions.

Unlike standard chatbots, SocraDesign does not give answers. It asks the right questions to help *you* find them.

---

## 🚀 Key Features (v3.0 Extended)

This system has evolved from a stateless prototype to a robust, persistent application.

* **🧠 Hybrid Evaluation Engine:** Reduces API latency by ~50% using a local Small Language Model (SLM) to classify response quality ("Good", "Weak", "Confused"), falling back to Gemini only when confidence is low.
* **💾 Session Persistence:** Integrated with **Google Firestore** to save conversation history and user personas, allowing users to pause and resume sessions anytime.
* **📱 Mobile-First Experience:** A completely redesigned, responsive UI with touch-friendly navigation, gamified progress tracking ("Phase Pills"), and auto-scrolling.
* **📊 Live Analytics:** Tracks user engagement metrics (latency, response depth, confusion rate) in real-time for facilitators.

---

## 🛠️ Setup & Installation

Prerequisites

* Python 3.10+
* A Google Cloud Project with Firestore enabled.
* A Google Gemini API Key.

Clone the Repository
git clone [https://github.com/shaunjose/socra-design.git](https://github.com/shaunjose/socra-design.git)
cd socra-design

Install Dependencies

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


## 🏗️ System Architecture

The application follows a client-server architecture using **Streamlit** (Frontend) and **LangGraph** (Backend).

```mermaid
graph TD
    User[User Input] --> UI[Streamlit Interface]
    UI --> SLM{Local SLM}
    SLM -- "High Conf" --> Router
    SLM -- "Low Conf" --> API[Gemini API]
    API --> Router
    Router --> Logic{Decision Logic}
    Logic -- "Weak" --> Probe[Generate Probing Q]
    Logic -- "Good" --> Next[Generate Phase Q]
    Logic -- "Phase Done" --> Distill[Distill Persona]
    Distill --> Firestore[(Google Firestore)]
    Probe --> Firestore
    Next --> Firestore
    Firestore --> UI
