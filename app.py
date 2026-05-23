import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM

# Streamlit Page Configuration
st.set_page_config(page_title="Groq AI Agent Dashboard", page_icon="🤖", layout="wide")

st.title("🤖 Groq Ultra-Fast AI Agent Dashboard")
st.write("Yeh ek autonomous AI Agent hai jo background me aapke tasks ko execute karta hai.")

# Sidebar me API Key aur Model select karne ka option
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    
    model_choice = st.selectbox(
        "Select Groq Model:",
        ["llama3-70b-8192", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    )
    
    st.info("💡 Pro Tip: Agentic reasoning ke liye Llama3-70b sabse best model hai.")

# Main Interface for Task Input
task_input = st.text_area(
    "📝 Agent ko kya task dena hai?", 
    placeholder="Example: Write a Python script to sort files in a folder based on extensions."
)

if st.button("🚀 Run Agent Task"):
    if not groq_api_key:
        st.error("Bhai, pehle sidebar me Groq API Key toh daalo! 😅")
    elif not task_input.strip():
        st.warning("Kuch task toh likho jise Agent execute kare!")
    else:
        with st.spinner("🤖 Agent dimaag chala raha hai... Please wait..."):
            try:
                # 1. CrewAI Native LLM class use karke Groq connect karein
                # Isse Pydantic v2 validation error 100% fix ho jata hai
                agent_llm = LLM(
                    model=f"groq/{model_choice}",
                    api_key=groq_api_key,
                    temperature=0.0
                )
                
                # 2. Agent Define karein
                task_runner_agent = Agent(
                    role='Expert Execution Specialist',
                    goal='To execute complex logical, analytical, or coding tasks flawlessly.',
                    backstory="""You are an elite autonomous agent. You take a core task, 
                    break it down logically, and deliver the perfect final result without any fluff.""",
                    verbose=True,
                    allow_delegation=False,
                    llm=agent_llm
                )
                
                # 3. Task create karein
                custom_task = Task(
                    description=task_input,
                    expected_output='The complete, fully executed response or solution to the user\'s request.',
                    agent=task_runner_agent
                )
                
                # 4. Crew setup karein
                agent_crew = Crew(
                    agents=[task_runner_agent],
                    tasks=[custom_task]
                )
                
                # Execution shuru karein
                final_result = agent_crew.kickoff()
                
                # 5. UI par Output dikhayein
                st.success("✅ Task Completed Successfully!")
                st.subheader("🏁 Agent Final Output:")
                
                # Output ko string me cast karke display karein
                st.markdown(str(final_result))
                
            except Exception as e:
                st.error(f"Oops! Kuch dikkat aayi bhai: {e}")
