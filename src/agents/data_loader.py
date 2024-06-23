# src/agents/data_loader.py

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
data_loader = Agent(
    role="Data Loader",
    goal="Load and preprocess sales transaction data",
    backstory="You are a meticulous data engineer, skilled in handling and preprocessing large datasets.",
    verbose=True,
    memory=True,
    llm=llm
)

# Define the task
task = Task(
    description="Load the sales transaction data, perform EDA, and prepare the data by calculating RFM, AOV, and AOI metrics.",
    agent=data_loader,
    expected_output="Preprocessed sales transaction data with additional fields for RFM, AOV, and AOI metrics."
)

# Create a crew with the agent and task
crew = Crew(
    agents=[data_loader],
    tasks=[task]
)

def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        return f"Error loading CSV file: {e}"

def perform_eda(df):
    summary = {
        "Header": list(df.columns),
        "Number of rows": len(df),
        "Statistics": df.describe().to_dict()
    }
    return summary

def calculate_rfm(df):
    current_date = pd.Timestamp.now()
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce', infer_datetime_format=True)
    rfm_df = df.groupby('customer_id').agg({
        'date': lambda x: (current_date - x.max()).days,
        'order_id': 'nunique',
        'revenue': 'sum'
    }).reset_index()
    rfm_df.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    return rfm_df

def calculate_aov_aoi(df, rfm_df):
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df['order_id'] = pd.to_numeric(df['order_id'], errors='coerce')

    aov_df = df.groupby('customer_id').agg({
        'revenue': 'sum',
        'order_id': 'nunique'
    }).reset_index()
    aov_df['aov'] = aov_df['revenue'] / aov_df['order_id']

    df = df.sort_values(by=['customer_id', 'date'])
    df['previous_order_date'] = df.groupby('customer_id')['date'].shift(1)
    df['order_interval'] = (df['date'] - df['previous_order_date']).dt.days

    aoi_df = df.groupby('customer_id').agg({'order_interval': 'mean'}).reset_index()
    aoi_df.columns = ['customer_id', 'aoi']

    # Fill NaN AOI values with a default value, e.g., 0
    aoi_df['aoi'].fillna(0, inplace=True)

    final_df = rfm_df.merge(aov_df[['customer_id', 'aov']], on='customer_id', how='left')
    final_df = final_df.merge(aoi_df, on='customer_id', how='left')
    return final_df

def run_data_loader(file_path):
    df = load_data(file_path)

    if isinstance(df, str):
        print(df)  # Print the error message
        return

    eda_summary = perform_eda(df)
    print("Exploratory Data Analysis Summary:")
    print(eda_summary)

    while True:
        user_question = input("Do you have any questions about the data? (Type 'no' to continue): ")
        if user_question.strip().lower() == 'no':
            break
        else:
            prompt = f"User's question: {user_question}\nData Summary: {eda_summary}"
            response = data_loader.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content.strip() if isinstance(response, AIMessage) else "No answer received."
            print("Answer:", answer)

    rfm_df = calculate_rfm(df)
    final_df = calculate_aov_aoi(df, rfm_df)
    print("Final Data with RFM, AOV, and AOI metrics:")
    print(final_df.head())

    # Save the final dataframe to a CSV file
    os.makedirs("/Users/cperazza/Documents/Projects/RFM_Tool/data", exist_ok=True)
    output_path = "/Users/cperazza/Documents/Projects/RFM_Tool/data/processed_data.csv"
    final_df.to_csv(output_path, index=False)
    print(f"Processed data saved to '{output_path}'.")

if __name__ == "__main__":
    project_file_path = input("Please provide the path to the project data file: ")
    run_data_loader(project_file_path)
    