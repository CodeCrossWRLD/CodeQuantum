import os
import json
import pandas as pd
from typing import List
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

# 1. Load Environment Variables from .env file
load_dotenv()

@tool
def inspect_csv_columns(file_name: str) -> str:
    """
    Read the column names and first 2 rows of a CSV file to understand its structure.
    Files: RaceTimes.csv, RaceResults.csv, QualTimes.csv, LapTimes.csv
    """
    try:
        df = pd.read_csv(file_name, nrows=2)
        return f"Columns in {file_name}: {df.columns.tolist()}\nSample data:\n{df.to_string()}"
    except Exception as e:
        return f"Error reading {file_name}: {str(e)}"

@tool
def python_data_analyzer(query: str) -> str:
    """
    Executes pandas code to analyze the CSV files.
    Example: df = pd.read_csv('RaceResults.csv'); print(df.groupby('TeamName')['Points'].sum())
    """
    # We use a shared dictionary to store the results of the execution
    local_vars = {"pd": pd}
    try:
        # We redirect stdout to capture 'print' statements from the AI
        import sys
        from io import StringIO

        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        # exec allows for multi-line pandas logic
        exec(query, {"pd": pd}, local_vars)

        sys.stdout = old_stdout
        return redirected_output.getvalue() or "Code executed successfully (no output)."
    except Exception as e:
        sys.stdout = old_stdout
        return f"Execution Error: {str(e)}"

@tool
def write_json(file_name: str, data: str) -> str:
    """
    Writes data to a JSON file.
    IMPORTANT: The 'data' argument must be a valid JSON-formatted string.
    """

# 3. Initialize Gemini Assistant
# Using the full path 'models/gemini-1.5-flash' explicitly
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

tools = [inspect_csv_columns, python_data_analyzer, write_json]

SYSTEM_PROMPT = (
    "You are a Data Analysis Assistant specializing in F1 data. "
    "You have access to CSV files like RaceResults.csv and LapTimes.csv. "
    "To solve problems: "
    "1. Use inspect_csv_columns to see what data is available. "
    "2. Use python_data_analyzer to perform calculations (always use print() to output results). "
    "3. Use write_json only if the user explicitly asks to save a file. "
    "Be concise and accurate."
)

# Initialize the agent
agent_executor = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)

# 4. The Interactive Loop
if __name__ == "__main__":
    print("=" * 60)
    print("F1 Agent Active (Powered by Google Gemini)")
    print("Type 'exit' to quit.")
    print("=" * 60)

    history: List[BaseMessage] = []

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        # Add user message to history
        history.append(HumanMessage(content=user_input))

        try:
            # The agent decides which tools to call automatically
            response = agent_executor.invoke({"messages": history})

            # Get the final answer from the agent
            ai_message = response["messages"][-1]
            history.append(ai_message)

            print(f"\nAgent: {ai_message.content}")

        except Exception as e:
            print(f"\nError: {str(e)}")