import streamlit as st
import os
from crewai import Agent, Task, Crew, LLM
# Naya safe aur direct import path
from langchain_community.tools import DuckDuckGoSearchRun

# Streamlit Page Configuration
st.set_page_config(page_title="Supercharged Groq Agent", page_icon="🚀", layout="wide")

st.title("🚀 Supercharged Groq AI Agent (With Live Web Search)")
st.write("Yeh agent live internet search kar sakta hai aur output file download karne ka option deta hai!")

# Safe DuckDuckGo Search Tool Initialize karein
search_tool = DuckDuckGoSearchRun()

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    
    model_choice = st.selectbox(
        "Select Groq Model:",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    )
    st.info("💡 Llama3-70b tools ko sabse achhe se use karna jaanta hai.")

# Main Task Input
task_input = st.text_area(
    "📝 Agent ko kya task ya research kaam dena hai?", 
    placeholder="Example: Search the internet for latest AI news and summarize them."
)

if st.button("🔥 Run Advanced Agent"):
    if not groq_api_key:
        st.error("Bhai, pehle sidebar me Groq API Key daalo! 😅")
    elif not task_input.strip():
        st.warning("Kuch task toh likho!")
    else:
        with st.spinner("🕵️‍♂️ Agent internet par research kar raha hai..."):
            try:
                # 1. Native LLM Setup
                agent_llm = LLM(
                    model=f"groq/{model_choice}",
                    api_key=groq_api_key,
                    temperature=0.2
                )
                
                # 2. Agent with TOOLS
                researcher_agent = Agent(
                    role='Advanced Research and Execution Specialist',
                    goal='To look up information on the internet and execute tasks with absolute accuracy.',
                    backstory="""You are an elite autonomous agent equipped with internet access. 
                    When asked about recent events or technical data, you use your search tool to find facts first, 
                    then synthesize a flawless response.""",
                    verbose=True,
                    allow_delegation=False,
                    llm=agent_llm,
                    tools=[search_tool]  # <-- Connected nicely
                )
                
                # 3. Task Setup
                custom_task = Task(
                    description=task_input,
                    expected_output='A highly comprehensive, factual response backed by the latest web search data if needed.',
                    agent=researcher_agent
                )
                
                # 4. Crew Setup
                agent_crew = Crew(
                    agents=[researcher_agent],
                    tasks=[custom_task]
                )
                
                # Execute
                final_result = agent_crew.kickoff()
                result_text = str(final_result)
                
                # 5. UI Output Display
                st.success("✅ Task Completed Successfully!")
                st.subheader("🏁 Agent Final Output:")
                st.markdown(result_text)
                
                # Download Button
                st.write("---")
                st.download_button(
                    label="📥 Download Output as Markdown (.md)",
                    data=result_text,
                    file_name="agent_output.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"Oops! Kuch dikkat aayi bhai: {e}")
