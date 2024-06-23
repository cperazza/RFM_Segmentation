# src/agents/rfm_calculator.py

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import os
import pandas as pd

# Set environment variables for Ollama
os.environ["OPENAI_API_KEY"] = "NA"

# Configure the LLM
llm = ChatOpenAI(
    model="crewai-llama3-8B",
    base_url="http://localhost:11434/v1"
)

# Define the agent
rfm_calculator = Agent(
    role="RFM Calculator",
    goal="Calculate RFM metrics and segment customers",
    backstory="You are an experienced data analyst, proficient in customer segmentation techniques.",
    verbose=True,
    memory=True,
    llm=llm
)

# Define the task
task = Task(
    description="Calculate RFM metrics and segment customers into quartiles, quintiles, and deciles.",
    agent=rfm_calculator,
    expected_output="RFM segmentation data with quartiles, quintiles, and deciles."
)

# Create a crew with the agent and task
crew = Crew(
    agents=[rfm_calculator],
    tasks=[task]
)

def load_rfm_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return f"Error loading CSV file: {e}"

def calculate_rfm_segments(df):
    # Ensure 'recency', 'frequency', and 'monetary' columns are numeric
    df['recency'] = pd.to_numeric(df['recency'], errors='coerce')
    df['frequency'] = pd.to_numeric(df['frequency'], errors='coerce')
    df['monetary'] = pd.to_numeric(df['monetary'], errors='coerce')

    quantiles = df[['recency', 'frequency', 'monetary']].quantile(q=[0.25, 0.5, 0.75, 0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.7, 0.9]).to_dict()

    def rfm_score(x, metric, quantiles, bucket_type):
        if metric == 'recency':
            if bucket_type == 4:
                if x <= quantiles[metric][0.25]:
                    return 4
                elif x <= quantiles[metric][0.50]:
                    return 3
                elif x <= quantiles[metric][0.75]:
                    return 2
                else:
                    return 1
            elif bucket_type == 5:
                if x <= quantiles[metric][0.2]:
                    return 5
                elif x <= quantiles[metric][0.4]:
                    return 4
                elif x <= quantiles[metric][0.6]:
                    return 3
                elif x <= quantiles[metric][0.8]:
                    return 2
                else:
                    return 1
            elif bucket_type == 10:
                if x <= quantiles[metric][0.1]:
                    return 10
                elif x <= quantiles[metric][0.3]:
                    return 9
                elif x <= quantiles[metric][0.5]:
                    return 8
                elif x <= quantiles[metric][0.7]:
                    return 7
                elif x <= quantiles[metric][0.9]:
                    return 6
                else:
                    return 5
        else:
            if bucket_type == 4:
                if x <= quantiles[metric][0.25]:
                    return 1
                elif x <= quantiles[metric][0.50]:
                    return 2
                elif x <= quantiles[metric][0.75]:
                    return 3
                else:
                    return 4
            elif bucket_type == 5:
                if x <= quantiles[metric][0.2]:
                    return 1
                elif x <= quantiles[metric][0.4]:
                    return 2
                elif x <= quantiles[metric][0.6]:
                    return 3
                elif x <= quantiles[metric][0.8]:
                    return 4
                else:
                    return 5
            elif bucket_type == 10:
                if x <= quantiles[metric][0.1]:
                    return 1
                elif x <= quantiles[metric][0.3]:
                    return 2
                elif x <= quantiles[metric][0.5]:
                    return 3
                elif x <= quantiles[metric][0.7]:
                    return 4
                elif x <= quantiles[metric][0.9]:
                    return 5
                else:
                    return 10

    for bucket_type, label in [(4, "quartile"), (5, "quintile"), (10, "decile")]:
        df[f'r_{label}_{bucket_type}'] = df['recency'].apply(rfm_score, args=('recency', quantiles, bucket_type))
        df[f'f_{label}_{bucket_type}'] = df['frequency'].apply(rfm_score, args=('frequency', quantiles, bucket_type))
        df[f'm_{label}_{bucket_type}'] = df['monetary'].apply(rfm_score, args=('monetary', quantiles, bucket_type))
        df[f'RFM_Segment_{bucket_type}'] = df[f'r_{label}_{bucket_type}'].map(str) + df[f'f_{label}_{bucket_type}'].map(str) + df[f'm_{label}_{bucket_type}'].map(str)
        df[f'RFM_Score_{bucket_type}'] = df[[f'r_{label}_{bucket_type}', f'f_{label}_{bucket_type}', f'm_{label}_{bucket_type}']].sum(axis=1)

    return df

def run_rfm_calculator(file_path):
    df = load_rfm_data(file_path)
    
    if isinstance(df, str):
        print(df)  # Print the error message
        return
    
    rfm_df = calculate_rfm_segments(df)
    print("RFM Segmented Data:")
    print(rfm_df.head())
    
    # Save the RFM segmented data to a CSV file
    os.makedirs("/Users/cperazza/Documents/Projects/RFM_Tool/data", exist_ok=True)
    output_path = "/Users/cperazza/Documents/Projects/RFM_Tool/data/rfm_segmented_data.csv"
    rfm_df.to_csv(output_path, index=False)
    print(f"RFM segmented data saved to '{output_path}'.")

if __name__ == "__main__":
    processed_data_path = input("Please provide the path to the processed data file (processed_data.csv): ")
    run_rfm_calculator(processed_data_path)