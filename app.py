import streamlit as st
import os
import requests
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool

# Streamlit Page Configuration
st.set_page_config(page_title="Supercharged Groq Agent", page_icon="🚀", layout="wide")

st.title("🚀 Supercharged Groq AI Agent (With Custom Web Search)")
st.write("Yeh agent live internet search kar sakta hai aur output file download karne ka option deta hai!")

# --- DUNIYA KA SABSE STABLE CUSTOM SEARCH TOOL ---
@tool("Web Search Tool")
def custom_search_tool(query: str) -> str:
    """Search the internet for the given query and return results. Use this whenever the user asks about recent events, latest tech, or news."""
    try:
        # DuckDuckGo ka public API endpoint use karke direct HTML results fetch karna
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Simple text parsing taaki heavy BeautifulSoup ki zaroorat na pade
            text = response.text
            # Top snippets extract karke clean format me return kar dena
            snippets = []
            start = 0
            for _ in range(5):  # Top 5 results nikalenge
                start = text.find('class="result__snippet"', start)
                if start == -1:
                    break
                start = text.find('>', start) + 1
                end = text.find('</a>', start)
                snippet = text[start:end].replace('<b>', '').replace('</b>', '').strip()
                snippets.append(snippet)
                start = end
            
            if snippets:
                return "\n\n".join(snippets)
            
        return "Bhai, search results nahi mil paaye, par internet check kiya tha."
    except Exception as e:
        return f"Search error: {str(e)}"

# Sidebar Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    
    model_choice = st.selectbox(
        "Select Groq Model:",
        ["llama3-70b-8192", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
    )
    st.info("💡 Llama3-70b custom tools ko sabse achhe se execute karta hai.")

# Main Task Input
task_input = st.text_area(
    "📝 Agent ko kya task ya research kaam dena hai?", 
    placeholder="Example: Search the internet for the top AI trends in 2026 and summarize them."
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
                
                # 2. Agent with Custom TOOL
                researcher_agent = Agent(
                    role='Advanced Research and Execution Specialist',
                    goal='To look up information on the internet and execute tasks with absolute accuracy.',
                    backstory="""You are an elite autonomous agent equipped with internet access. 
                    When asked about recent events or technical data, you use your Web Search Tool to find facts first, 
                    then synthesize a flawless response.""",
                    verbose=True,
                    allow_delegation=False,
                    llm=agent_llm,
                    tools=[custom_search_tool]  # <-- Humara custom native tool link ho gaya!
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
