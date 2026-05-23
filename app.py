import streamlit as st
import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

# Streamlit Page Configuration
st.set_page_config(page_title="Groq AI Agent Dashboard", page_icon="🤖", layout="wide")

st.title("🤖 Groq Ultra-Fast AI Agent")
st.write("Yeh ek AI Agent hai jo sirf chat nahi karta, aapke diye gaye tasks ko execute karta hai!")

# Sidebar me API Key aur Model select karne ka option
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    
    model_choice = st.selectbox(
        "Select Groq Model:",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    )
    
    st.info("💡 Tip: Agentic workflows ke liye Llama3-70b sabse accurate results deta hai.")

# Main Dashboard Interface
task_input = st.text_area(
    "📝 Agent ko kya task dena hai?", 
    placeholder="Example: Write a Python script to scrape top 5 news headlines from a sample website and format it as markdown."
)

if st.button("🚀 Run Agent Task"):
    if not groq_api_key:
        st.error("Bhai, pehle sidebar me Groq API Key toh daalo! 😅")
    elif not task_input.strip():
        st.warning("Kuch task toh likho jise Agent execute kare!")
    else:
        # Environment variable set karein
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        with st.spinner("🤖 Agent dimaag chala raha hai... Please wait..."):
            try:
                # 1. Groq LLM Initialize karein
                llm = ChatGroq(
                    temperature=0,
                    model_name=model_choice
                )
                
                # 2. Agent Define karein
                task_runner_agent = Agent(
                    role='Expert Execution Specialist',
                    goal='To execute any complex logical, analytical, or coding task flawlessly.',
                    backstory="""You are a highly efficient autonomous agent. You take a core task, 
                    break it down logically, and deliver the perfect final result without any fluff.""",
                    verbose=True,
                    allow_delegation=False,
                    llm=llm
                )
                
                # 3. User ke input ke hisab se Task create karein
                custom_task = Task(
                    description=task_input,
                    expected_output='The complete, fully executed response or solution to the user\'s request.',
                    agent=task_runner_agent
                )
                
                # 4. Crew setup karke execute karein
                agent_crew = Crew(
                    agents=[task_runner_agent],
                    tasks=[custom_task],
                    process=Process.sequential
                )
                
                # Kickoff execution
                final_result = agent_crew.kickoff()
                
                # 5. UI par Output dikhayein
                st.success("✅ Task Completed Successfully!")
                st.subheader("🏁 Agent Final Output:")
                st.markdown(final_result)
                
            except Exception as e:
                st.error(f"Opps! Kuch error aaya bhai: {e}")
