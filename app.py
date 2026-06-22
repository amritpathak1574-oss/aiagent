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
    elif code.startswith("
```"):
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
            "llama-3.1-8b-instant",       
            "llama-3.3-70b-versatile",    
            "gemma2-9b-it"                
        ]
    )
    st.info("💡 Pro Tip: 'llama-3.3-70b-versatile' use karein aur use real factual analysis karne dein.")

# Main input layout
task_input = st.text_area(
    "🛸 Apni team ko bada task assign karein:", 
    placeholder="Example: Search for Claude Fable 5, write a python code to print it, run the code, and save it to a file."
)

if st.button("🚀 Activate Hierarchical Crew"):
    if not groq_api_key:
        st.error("Bhai, bina Groq API Key ke team kaam nahi karegi! 😅")
    elif not task_input.strip():
        st.warning("Kuch task toh batao team ko!")
    else:
        with st.spinner("🕵️‍♂️ Team strict boundaries me reh kar kaam kar rahi hai..."):
            try:
                # Common LLM setup
                agent_llm = LLM(
                    model=f"groq/{model_choice}",
                    api_key=groq_api_key,
                    temperature=0.0, # Temperature 0 kiya taaki gappe na maare model!
                    max_tokens=1000
                )
                
                # 1. CODER AGENT
                coder_agent = Agent(
                    role='Python Developer',
                    goal='Search real info using web_search, write code using ONLY print statements, run it, and save it.',
                    backstory='You never use tkinter, tkinter is strictly banned. You only use web_search to find facts, write simple prints, test them, and write files.',
                    llm=agent_llm,
                    tools=[custom_search_tool, python_executor_tool, file_creator_tool],
                    verbose=True
                )
                
                # 2. REVIEWER AGENT
                reviewer_agent = Agent(
                    role='Code Auditor',
                    goal='Strictly reject any code containing tkinter, root.mainloop, or fake data.',
                    backstory='You ensure the coder actually used web_search and did not hallucinate fake fables like Aesop.',
                    llm=agent_llm,
                    verbose=True
                )
                
                # 3. MANAGER AGENT
                manager_agent = Agent(
                    role='Strict Project Manager',
                    goal='Force the team to use tools, prevent hallucinations, and coordinate structured delivery.',
                    backstory='An ultimate boss who demands real data from search and ensures final files are zipped.',
                    llm=agent_llm,
                    verbose=True
                )
                
                # --- AUTO-ENGINEERED STRICT PROMPT PACKAGING ---
                # Hum user ke prompt ko ek strict instructions wrapper me band kar rahe hain
                strict_system_prompt = f"""
                USER REQUEST: {task_input}
                
                STRICT COMPLIANCE RULES FOR THE CREW:
                1. You MUST execute 'web_search' for the exact terms requested. Do not assume or guess.
                2. The python script MUST use only standard console outputs (print statements). 
                3. DO NOT USE TKINTER, DO NOT USE GUI. GUI is strictly forbidden and will break the platform.
                4. You MUST invoke 'file_creator_and_zipper' to save the code to the requested python file name.
                5. If search returns limited data, print the actual search output instead of making up a fake story about tortoises or hares.
                """
                
                crew_task = Task(
                    description=strict_system_prompt,
                    expected_output='A verified console-based Python script containing actual search data, executed successfully, and saved into a zipped file.',
                    agent=manager_agent
                )
                
                manus_crew = Crew(
                    agents=[coder_agent, reviewer_agent],
                    tasks=[crew_task],
                    manager_agent=manager_agent,
                    process=Process.hierarchical
                )
                
                final_output = manus_crew.kickoff()
                result_text = str(final_output)
                
                st.success("🎯 Mission Accomplished! Team has delivered.")
                st.subheader("🏁 Final Solution Package:")
                st.markdown(result_text)
                
                st.write("---")
                if os.path.exists("project_output.zip"):
                    with open("project_output.zip", "rb") as fp:
                        st.download_button(
                            label="🎁 Download Project Files (.zip)",
                            data=fp,
                            file_name="project_output.zip",
                            mime="application/zip"
                        )
                
                st.download_button(
                    label="📄 Download Response Summary (.md)",
                    data=result_text,
                    file_name="crew_summary.md",
                    mime="text/markdown"
                )
                
            except Exception as e:
                st.error(f"Oops! Crew meeting me gaddbadd hui: {e}")
