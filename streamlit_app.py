import time
import asyncio
import os
import streamlit as st
from openai import OpenAI
from typing import Optional
from agents import Agent, Runner, set_default_openai_key
from agents.tracing import trace

# --- VERSION CONFIG ---
RULE_VERSION = "65"

# © 2025 Tommy Smith. All Rights Reserved.
# NFHS Football Rules Assistant – 2025 Edition
# Unauthorized copying, modification, distribution, or use of this software is prohibited.

# --- ENVIRONMENT SETUP ---
os.environ["OPENAI_TRACING_ENABLED"] = "true"

# --- API KEYS ---
set_default_openai_key(st.secrets["openai"]["api_key"])
client = OpenAI(api_key=st.secrets["openai"]["api_key"])

# --- CONFIGURATION ---
CONFIG = {
    "RULE_PROMPT_ID": "pmpt_688eb6bb5d2c8195ae17efd5323009e0010626afbd178ad9",
    "RULE_VECTOR_STORE_ID": "vs_689558cb487c819196565f82ed51220f",   # existing rules store
    "CASEBOOK_VECTOR_STORE_ID": "vs_689f72f117c8819195716f04bc2ae546", # case book store
}

# --- DEBUG MODE CHECK (robust to list values in older Streamlit) ---
_qp_val = st.query_params.get("query", "")
if isinstance(_qp_val, list):
    _qp_val = _qp_val[0] if _qp_val else ""
debug_mode = isinstance(_qp_val, str) and _qp_val.lower() == "debug"

# --- PAGE SETUP ---
st.set_page_config(page_title="🏈 NFHS Football Rules Assistant – 2025 Edition", layout="wide")
st.title("🏈 NFHS Football Rules Assistant – 2025 Edition")

# --- UNIFIED STYLES ---
st.markdown("""
    <style>
    .visually-hidden-watermark {
        position: absolute;
        left: -10000px;
        top: auto;
        width: 1px;
        height: 1px;
        overflow: hidden;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        border: 2px solid var(--primary-color);
        border-radius: 6px;
        padding: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    @media (prefers-color-scheme: light) {
        .stTextInput > div > div > input,
        .stTextArea > div > textarea {
            background-color: #fefefe;
        }
    }
    @media (prefers-color-scheme: dark) {
        .stTextInput > div > div > input,
        .stTextArea > div > textarea {
            background-color: #1f1f1f;
        }
    }
    h1 {
        font-size: 1.6rem !important;
        margin-bottom: 0.5rem;
    }
    h2 {
        font-size: 1.2rem !important;
        margin-top: 0.8rem;
        margin-bottom: 0.4rem;
    }
    h3 {
        font-size: 1rem !important;
        margin-top: 0.6rem;
        margin-bottom: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)

# --- HIDDEN DIGITAL WATERMARK ---
st.markdown(
    """
    <div class="visually-hidden-watermark">
      © 2025 Tommy Smith — NFHS Football Rules Assistant. Proprietary content and methods.
    </div>
    """,
    unsafe_allow_html=True
)

# --- WATERMARK HELPER ---
def render_output_with_watermark(content: str) -> None:
    st.markdown(content, unsafe_allow_html=True)
    st.markdown(
        """
        <div style='margin-top:8px'>
          <sub>© 2025 Tommy Smith — NFHS Football Rules Assistant</sub><br>
          <sub>This GPT can make mistakes. Check important info.</sub>
        </div>
        """,
        unsafe_allow_html=True
    )

# --- RULE LOOKUP FUNCTION ---
def ask_rule_lookup(rule_id: str) -> str | None:
    try:
        vector_ids = [CONFIG.get("RULE_VECTOR_STORE_ID"), CONFIG.get("CASEBOOK_VECTOR_STORE_ID")]
        vector_ids = [v for v in vector_ids if v and isinstance(v, str) and v.strip()]

        res = client.responses.create(
            prompt={"id": CONFIG["RULE_PROMPT_ID"], "version": RULE_VERSION},
            input=[{"role": "user", "content": f"id:{rule_id}"}],
            tools=[{"type": "file_search", "vector_store_ids": vector_ids}],
            text={"format": {"type": "text"}},
            max_output_tokens=2048,
            store=True
        )

        # --- LOG TOKEN USAGE ONLY IF DEBUG MODE ---
        if debug_mode and hasattr(res, "usage"):
            usage_data = res.usage

            st.write("🔍 Token usage:")
            st.write(f"Input tokens: {usage_data.input_tokens}")
            st.write(f"Output tokens: {usage_data.output_tokens}")

            cached_tokens = 0
            if hasattr(usage_data, "input_tokens_details"):
                details = usage_data.input_tokens_details
                cached_tokens = getattr(details, "cached_input_tokens", 0)
                st.write(f"  Cached input tokens: {cached_tokens}")
                st.write(f"  Cache creation input tokens: {getattr(details, 'cache_creation_input_tokens', 0)}")

            # Cost calculation for gpt-4.1-mini
            input_tokens = usage_data.input_tokens
            output_tokens = usage_data.output_tokens
            input_cost = (input_tokens - cached_tokens) * 0.0000004
            cached_cost = cached_tokens * 0.0000001
            output_cost = output_tokens * 0.0000016
            total_cost = input_cost + cached_cost + output_cost

            st.write(f"💲 Estimated cost this call: ${total_cost:.6f}")

        # --- Extract text output ---
        for out in res.output:
            if hasattr(out, "text") and hasattr(out.text, "value"):
                return out.text.value.strip()
            if hasattr(out, "content"):
                for block in out.content:
                    if hasattr(block, "text"):
                        return block.text.strip()

        return f"⚠️ No written response was generated for rule `{rule_id}`."

    except Exception as e:
        st.error(f"❌ Rule lookup failed: {e}")
        return None

# --- CACHE WRAPPER (TOP-LEVEL) ---
@st.cache_data(show_spinner=False)
def cached_rule_lookup(rule_id: str):
    return ask_rule_lookup(rule_id)

# --- GENERAL Q&A ---
async def _qa_agent_call(prompt: str, group_id: str | None = None) -> str:
    agent = Agent(name="Rules QA Assistant", instructions="Answer NFHS football rules questions.")
    with trace(workflow_name="NFHS_QA", group_id=group_id or None):
        result = await Runner.run(agent, prompt)
    return result.final_output

def ask_general(prompt: str) -> str | None:
    try:
        group_id = st.session_state.get("qa_thread_id") or "default-thread"
        return asyncio.run(_qa_agent_call(prompt, group_id))
    except Exception as e:
        st.error(f"❌ QA lookup failed: {e}")
        return None

# --- RULE LOOKUP UI ---
def render_rule_section():
    if st.session_state.get("rule_input"):
        for key in ("qa_thread_id", "qa_last_prompt", "qa_last_reply"):
            st.session_state[key] = ""

    st.markdown("### 🔍 Search by Rule ID (e.g., 8-5-3d) or type a question/scenario")
    rule_input = st.text_input("Enter your search here", key="rule_input")
    if st.button("Look Up", key="rule_button"):
        if rule_input.strip():
            result = cached_rule_lookup(rule_input.strip())
            st.session_state.rule_result = result
        else:
            st.warning("Please enter a rule ID to look up or enter a question or scenario.")

    if st.session_state.get("rule_result"):
        st.markdown("### 📘 Rule Lookup Result")
        render_output_with_watermark(st.session_state.rule_result or "⚠️ No response.")

# --- GENERAL Q&A UI ---
def render_general_section() -> None:
    for key in ("qa_thread_id", "qa_last_prompt", "qa_last_reply"):
        st.session_state.setdefault(key, "")

    if st.session_state.get("qa_prompt"):
        st.session_state["rule_input"] = ""
        st.session_state["rule_result"] = ""

    st.markdown("## 💬 Ask a Question About Rules or Scenarios")
    prompt = st.text_area(
        "Enter a question or test-style scenario:",
        placeholder="e.g., Can Team A recover their own punt after a muff?",
        key="qa_prompt"
    )

    if st.button("Ask", key="qa_button"):
        st.session_state.qa_last_prompt = prompt.strip()

    if st.session_state.qa_last_prompt:
        reply = ask_general(st.session_state.qa_last_prompt)
        st.session_state.qa_last_reply = reply or ""
        st.markdown("### 🧠 Assistant Reply")
        render_output_with_watermark(st.session_state.qa_last_reply or "⚠️ No response received.")

# --- MAIN ---
def main() -> None:
    render_rule_section()
    # Uncomment to enable Q&A
    # st.markdown("---")
    # render_general_section()

if __name__ == "__main__":
    main()

# --- FOOTER ---
st.markdown(
    f"""
    <style>
    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: gray;
        text-align: center;
        font-size: 12px;
        padding: 5px;
        z-index: 9999;
    }}
    </style>
    <div class="footer">
        NFHS Football Rules Assistant – 2025 Edition v1.{RULE_VERSION}<br>
        © 2025 Tommy Smith. All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True
)
