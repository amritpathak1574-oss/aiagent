import streamlit as st
import os
import requests
import zipfile
from crewai import Agent, Task, Crew, LLM, Process
from crewai.tools import tool

# Streamlit Page Configuration
st.set_page_config(page_title="Manus-Style Hierarchical Crew", page_icon="🧠", layout="wide")

st.title("🧠 Hierarchical AI Crew (Manus AI Lite Architecture)")
st.write("Isme ek Manager, ek Coder, aur ek Reviewer agent milkar aapka kaam karte hain aur code execute bhi kar sakte hain!")

# --- TOOL 1: FIXED WEB SEARCH TOOL ---
# Iska naam humne 'web_search' kar diya hai, taaki LLM confuse na ho
@tool("web_search")
def custom_search_tool(query: str) -> str:
    """Useful to search the internet for web results, latest news, and articles about any topic. Input should be a simple search query string."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            text = response.text
            snippets = []
            start = 0
            for _ in range(3):
                start = text.find('class="result__snippet"', start)
                if start == -1: break
                start = text.find('>', start) + 1
                end = text.find('</a>', start)
                snippets.append(text[start:end].replace('<b>', '').replace('</b>', '').strip())
                start = end
            return "\n\n".join(snippets) if snippets else "No results found on the internet."
        return "Search failed due to network error."
    except Exception as e:
        return f"Error executing search: {str(e)}"

# --- TOOL 2: PYTHON REPL / CODE EXECUTOR TOOL ---
@tool("python_code_executor")
def python_executor_tool(code: str) -> str:
    """Execute arbitrary Python code securely in the background and return stdout/stderr output. Use this to verify scripts or run calculations."""
    import sys
    from io import StringIO
    
    if code.startswith("```python"):
        code = code[9:-3]
    elif code.startswith("```"):
        code = code[3:-3]
        
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()
    try:
        exec(code, {}, {})
        sys.stdout = old_stdout
        return redirected_output.getvalue() if redirected_output.getvalue() else "Code executed successfully with no print output."
    except Exception as e:
        sys.stdout = old_stdout
        return f"Execution Error: {str(e)}"

# --- TOOL 3: FILE WRITER & ZIPPER TOOL ---
@tool("file_creator_and_zipper")
def file_creator_tool(filename: str, content: str) -> str:
    """Create a file with specific content and automatically pack it into a deployable zip file called 'project_output.zip'."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
            
        zip_name = "project_output.zip"
        with zipfile.ZipFile(zip_name, 'w') as zipf:
            zipf.write(filename)
            
        return f"Successfully created '{filename}' and packed it inside '{zip_name}'!"
    except Exception as e:
        return f"File Creation Error: {str(e)}"


# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Crew Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    
    model_choice = st.selectbox(
        "Select Core LLM Engine:",
        [
            "llama-3.1-8b-instant",       # High limit, zero rate limit error, super fast
            "llama-3.3-70b-versatile",    # Ultra smart reasoning
            "gemma2-9b-it"                # Fast and efficient fallback
        ]
    )
    st.info("💡 Bhai, function validation error se bachne ke liye 'llama-3.3-70b-versatile' use karein toh better hai, woh tools sahi se call karta hai.")

# Main input layout
task_input = st.text_area(
    "🛸 Apni team ko bada task assign karein:", 
    placeholder="Example: Search for Claude Fable 5, write a python code to print it, run the code, and save it to a file named fable.py"
)

if st.button("🚀 Activate Hierarchical Crew"):
    if not groq_api_key:
        st.error("Bhai, bina Groq API Key ke team kaam nahi karegi! 😅")
    elif not task_input.strip():
        st.warning("Kuch task toh batao team ko!")
    else:
        with st.spinner("🕵️‍♂️ Team function validation checks clear karke kaam shuru kar rahi hai..."):
            try:
                # Optimized LLM Instance
                agent_llm = LLM(
                    model=f"groq/{model_choice}",
                    api_key=groq_api_key,
                    temperature=0.1,
                    max_tokens=1000
                )
                
                # 1. CODER AGENT
                coder_agent = Agent(
                    role='Python Developer',
                    goal='Write clean code, run web searches using web_search, and use tools to verify and save final scripts.',
                    backstory='An expert programmer focused strictly on writing working code, using web_search for queries, and saving it via tools.',
                    llm=agent_llm,
                    tools=[custom_search_tool, python_executor_tool, file_creator_tool],
                    verbose=True
                )
                
                # 2. REVIEWER AGENT
                reviewer_agent = Agent(
                    role='Code Reviewer',
                    goal='Review the code logic and ensure correctness before finalizing output.',
                    backstory='A swift quality controller who verifies code accuracy without unnecessary words.',
                    llm=agent_llm,
                    verbose=True
                )
                
                # 3. MANAGER AGENT (Strictly Management - No tools)
                manager_agent = Agent(
                    role='Project Manager',
                    goal='Deconstruct tasks logically, delegate tasks efficiently to workers, and compile concise results.',
                    backstory='An efficient orchestrator who manages team workflow directly without text bloat or tool confusion.',
                    llm=agent_llm,
                    verbose=True
                )
                
                # Task Definition
                crew_task = Task(
                    description=task_input,
                    expected_output='A clean, finalized solution report with execution results. If file creation was requested, confirm file name.',
                    agent=manager_agent
                )
                
                # Hierarchical Crew Setup
                manus_crew = Crew(
                    agents=[coder_agent, reviewer_agent],
                    tasks=[crew_task],
                    manager_agent=manager_agent,
                    process=Process.hierarchical
                )
                
                # Run the whole system
                final_output = manus_crew.kickoff()
                result_text = str(final_output)
                
                st.success("🎯 Mission Accomplished! Team has delivered.")
                st.subheader("🏁 Final Solution Package:")
                st.markdown(result_text)
                
                # Check if zip file was created
                st.write("---")
                if os.path.exists("project_output.zip"):
                    with open("project_output.zip", "rb") as fp:
                        st.download_button(
                            label="🎁 Download Project Files (.zip)",
                            data=fp,
                            file_name="project_output.zip",
                            mime="application/zip"
                        )
                
                # Standard Markdown backup download
                st.download_button(
                    label="📄 Download Response Summary (.md)",
                    data=result_text,
                    file_name="crew_summary.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"Oops! Crew meeting me gaddbadd hui: {e}")
