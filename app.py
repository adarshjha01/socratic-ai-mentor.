import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

st.set_page_config(page_title="AI Project Mentor", layout="centered")
st.title("Socratic AI Mentor")
st.markdown("I will help you build your project, but I will **not** write the code for you.")

# Securely fetch the API key from Streamlit Cloud Secrets
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except KeyError:
    st.error("API Key not found in Streamlit Secrets. Please configure it in the app settings.")
    st.stop()

# Handling Streamlit's Memory (Session State)
if "chat_history" not in st.session_state:
    system_instruction = """You are a strict but encouraging AI Mentor for Computer Science students.
    A student is trying to build a feature for their project and is stuck.

    YOUR STRICT RULES:
    1. NEVER write the direct code solution for them. This is an absolute rule.
    2. Use a 3-level hint system based on the conversation history:
       - Level 1 (Nudge): Ask a leading question to make them think about the logic.
       - Level 2 (Clue): Give a conceptual analogy or point them to a specific documentation concept.
       - Level 3 (Explanation): Explain the underlying logic or algorithm step-by-step in plain English, but leave the exact syntax to them.
    """
    st.session_state.chat_history = [SystemMessage(content=system_instruction)]

# Main Chat Interface
llm = ChatGroq(model="llama-3.1-8b-instant")

st.divider()

# Render all previous messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

# Capture new user input
user_query = st.chat_input("Tell me what you are stuck on...")

if user_query:
    with st.chat_message("user"):
        st.write(user_query)
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = llm.invoke(st.session_state.chat_history)
                st.write(response.content)
                st.session_state.chat_history.append(AIMessage(content=response.content))
            except Exception as e:
                st.error(f"Connection error: {e}")
