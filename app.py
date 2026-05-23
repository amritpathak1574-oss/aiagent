import os
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq

# 1. Apni Groq API Key set karein
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

# 2. Groq LLM ko initialize karein (Speed king!)
groq_llm = ChatGroq(
    temperature=0,  # Agent ke liye 0 temperature best hai taaki wo zyada creative na ho, accurate kaam kare
    model_name="llama-3.3-70b-versatile"  # Aap llama3-8b-8192 ya mixtral-8x7b-32768 bhi use kar sakte ho
)

# 3. Agent Define Karein (Groq LLM ke saath)
coding_assistant_agent = Agent(
    role='Expert Python Developer',
    goal='Write clean, optimized, and production-ready Python code and automation scripts.',
    backstory="""You are a world-class Python developer. You specialize in automation, 
    writing efficient algorithms, and creating robust software structures.""",
    verbose=True,
    allow_delegation=False,
    llm=groq_llm  # Yahan humne Groq connect kar diya
)

# 4. Agent ko Task dijiye
coding_task = Task(
    description='Create a Python script that automatically organizes files in a directory based on their extensions (e.g., .txt to TextFolder, .jpg to ImageFolder).',
    expected_output='A complete, well-commented Python script ready to be executed.',
    agent=coding_assistant_agent
)

# 5. Kickoff the Agent
dev_crew = Crew(
    agents=[coding_assistant_agent],
    tasks=[coding_task],
    process=Process.sequential
)

print("#### Groq Agent Active... Thinking Fast... ####")
result = dev_crew.kickoff()

print("\n#### Agent Ka Final Output: ####\n")
print(result)
