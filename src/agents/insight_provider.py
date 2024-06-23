# src/agents/insight_provider.py

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import os
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# Set environment variables for Ollama
os.environ["OPENAI_API_KEY"] = "NA"

# Configure the LLM
llm = ChatOpenAI(
    model="crewai-llama3-8B",
    base_url="http://localhost:11434/v1"
)

# Define the agent
insight_provider = Agent(
    role="Insight Provider",
    goal="Provide detailed insights and recommendations based on RFM analysis",
    backstory="You are an insightful analyst, skilled in deriving actionable insights from customer segmentation data.",
    verbose=True,
    memory=True,
    llm=llm
)

# Define the task
task = Task(
    description="Analyze the RFM segmented data, generate insights, visualizations, and a final report.",
    agent=insight_provider,
    expected_output="Detailed insights and recommendations based on RFM analysis, including visualizations and a final report."
)

# Create a crew with the agent and task
crew = Crew(
    agents=[insight_provider],
    tasks=[task]
)

def load_rfm_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return f"Error loading CSV file: {e}"

def analyze_segments(df, bucket_type):
    segment_column = f'RFM_Segment_{bucket_type}'
    if segment_column not in df.columns:
        raise KeyError(f"Column '{segment_column}' not found in the data.")
    
    segment_counts = df[segment_column].value_counts().to_dict()
    summary_statistics = df.describe().to_dict()
    
    analysis_prompt = f"""
    Based on the RFM data provided, here are the segment counts and summary statistics for {bucket_type}:
    
    Segment Counts: {segment_counts}
    
    Summary Statistics: {summary_statistics}
    
    Please provide a detailed analysis of these segments, including characteristics, insights, and marketing recommendations for each segment.
    """
    analysis_response = insight_provider.llm.invoke([HumanMessage(content=analysis_prompt)])
    
    if isinstance(analysis_response, AIMessage):
        detailed_analysis = analysis_response.content
    else:
        detailed_analysis = "No detailed analysis received from LLM."
    
    segment_analysis = {
        'segment_counts': segment_counts,
        'summary_statistics': summary_statistics,
        'detailed_analysis': detailed_analysis
    }
    return segment_analysis

def generate_visualizations(df):
    # Distribution plot for RFM segments
    plt.figure(figsize=(10, 6))
    df['RFM_Segment_4'].value_counts().plot(kind='bar')
    plt.title('Distribution of Customers by RFM Segments (Quartiles)')
    plt.xlabel('RFM Segment')
    plt.ylabel('Number of Customers')
    plt.savefig('/Users/cperazza/Documents/Projects/RFM_Tool/data/rfm_segment_distribution_quartiles.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    df['RFM_Segment_5'].value_counts().plot(kind='bar')
    plt.title('Distribution of Customers by RFM Segments (Quintiles)')
    plt.xlabel('RFM Segment')
    plt.ylabel('Number of Customers')
    plt.savefig('/Users/cperazza/Documents/Projects/RFM_Tool/data/rfm_segment_distribution_quintiles.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    df['RFM_Segment_10'].value_counts().plot(kind='bar')
    plt.title('Distribution of Customers by RFM Segments (Deciles)')
    plt.xlabel('RFM Segment')
    plt.ylabel('Number of Customers')
    plt.savefig('/Users/cperazza/Documents/Projects/RFM_Tool/data/rfm_segment_distribution_deciles.png')
    plt.close()

def generate_report(df, quartile_analysis, quintile_analysis, decile_analysis, output_file_name):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, txt="RFM Analysis Report", ln=True, align='C')
    
    # Analysis for Quartiles
    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(200, 10, txt="Analysis for Quartiles (4 Buckets):", ln=True)
    pdf.set_font("Arial", size=10)
    for segment, count in quartile_analysis['segment_counts'].items():
        pdf.cell(200, 10, txt=f"Segment {segment}: {count} customers", ln=True)
    
    pdf.cell(200, 10, txt="Summary Statistics:", ln=True)
    for key, value in quartile_analysis['summary_statistics'].items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    
    pdf.cell(200, 10, txt="Detailed Analysis and Recommendations:", ln=True)
    pdf.multi_cell(200, 10, txt=quartile_analysis['detailed_analysis'], align='L')
    
    pdf.image('/Users/cperazza/Documents/Projects/RFM_Tool/data/rfm_segment_distribution_quartiles.png', x=10, y=None, w=150)
    
    # Analysis for Quintiles
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(200, 10, txt="Analysis for Quintiles (5 Buckets):", ln=True)
    pdf.set_font("Arial", size=10)
    for segment, count in quintile_analysis['segment_counts'].items():
        pdf.cell(200, 10, txt=f"Segment {segment}: {count} customers", ln=True)
    
    pdf.cell(200, 10, txt="Summary Statistics:", ln=True)
    for key, value in quintile_analysis['summary_statistics'].items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    
    pdf.cell(200, 10, txt="Detailed Analysis and Recommendations:", ln=True)
    pdf.multi_cell(200, 10, txt=quintile_analysis['detailed_analysis'], align='L')
    
    pdf.image('/Users/cperazza/Documents/Projects/RFM_Tool/data/rfm_segment_distribution_quintiles.png', x=10, y=None, w=150)
    
    # Analysis for Deciles
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(200, 10, txt="Analysis for Deciles (10 Buckets):", ln=True)
    pdf.set_font("Arial", size=10)
    for segment, count in decile_analysis['segment_counts'].items():
        pdf.cell(200, 10, txt=f"Segment {segment}: {count} customers", ln=True)
    
    pdf.cell(200, 10, txt="Summary Statistics:", ln=True)
    for key, value in decile_analysis['summary_statistics'].items():
        pdf.cell(200, 10, txt=f"{key}: {value}", ln=True)
    
    pdf.cell(200, 10, txt="Detailed Analysis and Recommendations:", ln=True)
    pdf.multi_cell(200, 10, txt=decile_analysis['detailed_analysis'], align='L')
    
    pdf.image('/Users/cperazza/Documents/Projects/RFM_Tool/data/rfm_segment_distribution_deciles.png', x=10, y=None, w=150)
    
    # Save the PDF file
    report_path = f"/Users/cperazza/Documents/Projects/RFM_Tool/data/{output_file_name}_report.pdf"
    pdf.output(report_path)
    print(f"Report saved to '{report_path}'.")

def generate_chart_code(output_file_name):
    chart_code = """
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('rfm_segmented_data.csv')

# Distribution plot for RFM segments (Quartiles)
plt.figure(figsize=(10, 6))
df['RFM_Segment_4'].value_counts().plot(kind='bar')
plt.title('Distribution of Customers by RFM Segments (Quartiles)')
plt.xlabel('RFM Segment')
plt.ylabel('Number of Customers')
plt.savefig('rfm_segment_distribution_quartiles.png')
plt.show()

# Distribution plot for RFM segments (Quintiles)
plt.figure(figsize=(10, 6))
df['RFM_Segment_5'].value_counts().plot(kind='bar')
plt.title('Distribution of Customers by RFM Segments (Quintiles)')
plt.xlabel('RFM Segment')
plt.ylabel('Number of Customers')
plt.savefig('rfm_segment_distribution_quintiles.png')
plt.show()

# Distribution plot for RFM segments (Deciles)
plt.figure(figsize=(10, 6))
df['RFM_Segment_10'].value_counts().plot(kind='bar')
plt.title('Distribution of Customers by RFM Segments (Deciles)')
plt.xlabel('RFM Segment')
plt.ylabel('Number of Customers')
plt.savefig('rfm_segment_distribution_deciles.png')
plt.show()
"""
    with open(f"/Users/cperazza/Documents/Projects/RFM_Tool/data/{output_file_name}_chart_code.py", 'w') as f:
        f.write(chart_code)
    print(f"Chart code saved to '/Users/cperazza/Documents/Projects/RFM_Tool/data/{output_file_name}_chart_code.py'.")

def run_insight_provider(file_path, output_file_name):
    while True:
        df = load_rfm_data(file_path)
        
        if isinstance(df, str):
            print(df)  # Print the error message
            return
        
        quartile_analysis = analyze_segments(df, '4')
        quintile_analysis = analyze_segments(df, '5')
        decile_analysis = analyze_segments(df, '10')
        
        generate_visualizations(df)
        generate_report(df, quartile_analysis, quintile_analysis, decile_analysis, output_file_name)
        generate_chart_code(output_file_name)

        user_satisfaction = input("Are you happy with the final report and insights? (yes/no): ")
        if user_satisfaction.strip().lower() == 'yes':
            break
        else:
            feedback = input("Please provide your feedback for adjustments: ")
            # Process feedback and re-run the task based on feedback
            print("Adjusting the analysis based on feedback...")
            # Adjustments can be implemented here

if __name__ == "__main__":
    processed_data_path = input("Please provide the path to the rfm segmented data file (rfm_segmented_data.csv): ")
    output_file_name = input("Please provide the desired base name for the output files (without extension): ")
    run_insight_provider(processed_data_path, output_file_name)