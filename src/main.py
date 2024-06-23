# src/main.py

import os

def run_project_manager():
    os.system("python /Users/cperazza/Documents/Projects/RFM_Tool/src/agents/project_manager.py")

def run_data_loader():
    os.system("python /Users/cperazza/Documents/Projects/RFM_Tool/src/agents/data_loader.py")

def run_rfm_calculator():
    os.system("python /Users/cperazza/Documents/Projects/RFM_Tool/src/agents/rfm_calculator.py")

def run_insight_provider():
    os.system("python /Users/cperazza/Documents/Projects/RFM_Tool/src/agents/insight_provider.py")

if __name__ == "__main__":
    print("Starting Project Manager...")
    run_project_manager()
    
    proceed = input("Do you want to proceed to Data Loader? (yes/no): ").strip().lower()
    if proceed == 'yes':
        print("Starting Data Loader...")
        run_data_loader()
    else:
        print("Process stopped by user.")
        exit()
    
    proceed = input("Do you want to proceed to RFM Calculator? (yes/no): ").strip().lower()
    if proceed == 'yes':
        print("Starting RFM Calculator...")
        run_rfm_calculator()
    else:
        print("Process stopped by user.")
        exit()

    proceed = input("Do you want to proceed to Insight Provider? (yes/no): ").strip().lower()
    if proceed == 'yes':
        print("Starting Insight Provider...")
        run_insight_provider()
    else:
        print("Process stopped by user.")
        exit()

    print("All tasks completed.")