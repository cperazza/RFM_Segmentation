# src/agents/project_manager.py

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import os
from fpdf import FPDF

# Set environment variables for Ollama
os.environ["OPENAI_API_KEY"] = "NA"

# Configure the LLM
llm = ChatOpenAI(
    model="crewai-llama3-8B",
    base_url="http://localhost:11434/v1"
)

# Define the agent
project_manager = Agent(
    role="Project Manager",
    goal="Define and manage the project scope for the RFM analysis",
    backstory="You are a meticulous project manager, skilled in organizing and planning complex projects.",
    verbose=True,
    memory=True,
    llm=llm
)

# Define the task
task = Task(
    description="Gather project guidelines, create a detailed execution plan, and generate a PDF report.",
    agent=project_manager,
    expected_output="A detailed execution plan in PDF format."
)

# Create a crew with the agent and task
crew = Crew(
    agents=[project_manager],
    tasks=[task]
)

def generate_execution_plan_pdf(execution_plan, file_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="Project Execution Plan", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(200, 10, txt=execution_plan)
    pdf.output(file_path)
    print(f"Execution plan saved to '{file_path}'.")

def run_project_manager():
    project_guidelines = input("Please provide detailed project guidelines: ")
    reference_file_path = input("Do you have any reference files (CSV, TXT, PDF)? If yes, provide the file path or type 'NO': ")

    if reference_file_path.strip().lower() != 'no':
        try:
            with open(reference_file_path, 'r') as file:
                reference_file_content = file.read()
            project_guidelines += f"\n\nReference File Content:\n{reference_file_content}"
        except Exception as e:
            print(f"Error reading reference file: {e}")
    
    recommendations = analyze_project_guidelines(project_guidelines)

    while True:
        print("Recommendations:")
        print(recommendations)
        user_feedback = input("Do you approve this execution plan? (yes/no): ")
        if user_feedback.strip().lower() == 'yes':
            break
        else:
            feedback = input("Please provide your feedback for adjustments: ")
            project_guidelines += f"\n\nUser Feedback:\n{feedback}"
            recommendations = analyze_project_guidelines(project_guidelines)

    os.makedirs("/Users/cperazza/Documents/Projects/RFM_Tool/data", exist_ok=True)
    execution_plan_path = "/Users/cperazza/Documents/Projects/RFM_Tool/data/project_execution_plan.pdf"
    generate_execution_plan_pdf(recommendations, execution_plan_path)

def analyze_project_guidelines(project_guidelines):
    prompt = f"Project Guidelines: {project_guidelines}\nPlease provide a detailed execution plan for the project."
    response = project_manager.llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip() if isinstance(response, AIMessage) else "No recommendations received."

if __name__ == "__main__":
    run_project_manager()