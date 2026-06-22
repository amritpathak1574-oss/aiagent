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

# --- TOOL 1: WEB SEARCH TOOL ---
@tool("Web Search Tool")
def custom_search_tool(query: str) -> str:
    """Search the internet for any given query to get the latest 2026 data."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
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
            return "\n\n".join(snippets) if snippets else "No results found."
        return "Search failed."
    except Exception as e:
        return f"Error: {str(e)}"

# --- TOOL 2: PYTHON REPL / CODE EXECUTOR TOOL ---
@tool("Python Code Executor")
def python_executor_tool(code: str) -> str:
    """Execute arbitrary Python code securely in the background and return stdout/stderr output. Use this to verify scripts or run calculations."""
    import sys
    from io import StringIO
    
    # Clean code formatting if wrapped in markdown
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
@tool("File Creator and Zipper")
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


# Sidebar Configuration (Model Updated Here)
with st.sidebar:
    st.header("⚙️ Crew Configuration")
    groq_api_key = st.text_input("Enter Groq API Key:", type="password")
    
    model_choice = st.selectbox(
        "Select Core LLM Engine:",
        ["llama-3.3-70b-versatile", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    )
    st.info("💡 Pro Tip: 'llama-3.3-70b-versatile' naye tools aur complex workflow ke liye sabse best aur accurate hai.")

# Main input layout
task_input = st.text_area(
    "🛸 Apni team ko bada task assign karein:", 
    placeholder="Example: Write a python script to calculate fibonacci series up to 10 numbers, execute it to verify if it works, and save it to a file."
)

if st.button("🚀 Activate Hierarchical Crew"):
    if not groq_api_key:
        st.error("Bhai, bina Groq API Key ke team kaam nahi karegi! 😅")
    elif not task_input.strip():
        st.warning("Kuch task toh batao team ko!")
    else:
        with st.spinner("🕵️‍♂️ Manager Agent planning kar raha hai aur baaki team ko instructions bhej raha hai..."):
            try:
                # Common LLM instance
                agent_llm = LLM(
                    model=f"groq/{model_choice}",
                    api_key=groq_api_key,
                    temperature=0.1
                )
                
                # 1. CODER AGENT
                coder_agent = Agent(
                    role='Senior Python Developer',
                    goal='Write clean, bug-free Python code and execute it using tools to verify it works flawlessly.',
                    backstory='You are a master coder. You write code, test it using the Python Code Executor tool, look at the error logs if any, fix it, and only output 100% verified scripts.',
                    llm=agent_llm,
                    tools=[python_executor_tool, file_creator_tool],
                    verbose=True
                )
                
                # 2. REVIEWER AGENT
                reviewer_agent = Agent(
                    role='Quality Assurance & Code Reviewer',
                    goal='Review the code, audit the logical output, and check for any conceptual or security loopholes.',
                    backstory='You are a meticulous inspector. You double-check the developer\'s work. If something is missing, you suggest final structural improvements before delivery.',
                    llm=agent_llm,
                    verbose=True
                )
                
                # 3. MANAGER AGENT
                manager_agent = Agent(
                    role='Product Manager & Orchestrator',
                    goal='Oversee the entire operational workflow, split tasks logically, delegate to Coder/Reviewer, and compile the perfect final delivery.',
                    backstory='You are the central brain. You coordinate between the client request, web research, developer, and reviewer. You manage the team dynamically.',
                    llm=agent_llm,
                    tools=[custom_search_tool],
                    verbose=True
                )
                
                # Task Definition
                crew_task = Task(
                    description=task_input,
                    expected_output='A finalized, reviewed product including code verification logs and confirmation of the zipped files if file creation was required.',
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
