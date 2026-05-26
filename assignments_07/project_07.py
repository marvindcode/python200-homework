import os
os.environ["MPLBACKEND"] = "Agg"

from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIServerModel, tool

load_dotenv()


api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)



BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "assignments_01" / "outputs" / "merged_happiness.csv"

RESOURCE_DIR = BASE_DIR / "assignments" / "resources" / "happiness_project"

OUTPUT_DIR = BASE_DIR / "assignments_07" / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = None


@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Returns:
        A dictionary containing the dataset shape and column names.
    """

    global df

    # First try the merged file from Week 1
    if DATA_PATH.exists():

        df = pd.read_csv(DATA_PATH)

    else:

        
        files = sorted(RESOURCE_DIR.glob("*.csv"))

        if not files:
            return {"error": "No happiness CSV files were found."}

        frames = []

        for file in files:

            temp_df = pd.read_csv(file)

            year_text = "".join(
                char for char in file.stem if char.isdigit()
            )

            if year_text:
                temp_df["year"] = int(year_text[-4:])

            frames.append(temp_df)

        df = pd.concat(frames, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": list(df.columns)
    }


@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for one column.

    Args:
        column: The column name to summarize.

    Returns:
        A dictionary of summary statistics.
    """

    if df is None:
        return {"error": "No data is loaded yet."}

    if column not in df.columns:
        return {"error": f"Column '{column}' was not found."}

    return df[column].describe().to_dict()


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute Pearson correlation between two numeric columns.

    Args:
        col1: The first numeric column.
        col2: The second numeric column.

    Returns:
        A dictionary with correlation results.
    """

    if df is None:
        return {"error": "No data is loaded yet."}

    if col1 not in df.columns or col2 not in df.columns:
        return {"error": "One or both columns were not found."}

    clean_df = df[[col1, col2]].dropna()

    if clean_df.empty:
        return {"error": "No valid rows after removing missing vslues."}

    r, p_value = pearsonr(
        clean_df[col1],
        clean_df[col2]
    )

    return {
        "col1": col1,
        "col2": col2,
        "pearson_r": round(float(r), 4),
        "p_value": round(float(p_value), 4)
    }

@tool
def get_top_n_countries(
    column: str,
    year: int,
    n: int = 5
) -> dict:
    """Return the top N countries ranked by a column.

    Args:
        column: The column used for ranking.
        year: The year to filter.
        n: Number of countries to return.

    Returns:
        A dictionary containing ranked countries.
    """

    if df is None:
        return {"error": "No data is loaded yet."}

    if column not in df.columns:
        return {"error": f"Column '{column}' was not found."}

    if "year" not in df.columns:
        return {"error": "The dataset does not contain a year column."}

    if "country" in df.columns:
        country_col = "country"

    elif "Country" in df.columns:
        country_col = "Country"

    else:
        return {"error": "No country column was found."}

    year_df = df[df["year"] == year]

    if year_df.empty:
        return {"error": f"No data found for year {year}."}

    top_rows = (
        year_df
        .sort_values(column, ascending=False)
        .head(n)[[country_col, column]]
        .rename(columns={country_col: "country"})
    )

    return {
        "year": year,
        "ranked_by": column,
        "top_countries": top_rows.to_dict(orient="records")
    }



SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient.
Be concise and student friendly in your responses.
"""

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)

agent = CodeAgent(
    tools=[
        load_happiness_data,
        summarize_column,
        compute_correlation,
        get_top_n_countries,
    ],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=[
        "pandas",
        "matplotlib",
        "matplotlib.pyplot",
        "scipy.stats",
        "pathlib",
    ],
    max_steps=8,
)


def run_guided_queries():
    queries = [
        "Load the happiness data and tell me its shape and column names.",
        "Summarize the Happiness score column.",
        "What is the correlation between GDP per capita and Happiness score?",
        "Show me the top 5 countries ranked by Happiness score in 2020.",
        f"Plot Happiness score over the years as a line chart, with one line per Regional indicator. Save the plot to {OUTPUT_DIR / 'happiness_by_region.png'}.",
    ]

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(
            query,
            reset=False,
            additional_args={
                "DATA_PATH": DATA_PATH,
                "OUTPUT_DIR": OUTPUT_DIR,
            }
        )
        print(response)


def run_my_queries():
    my_query_1 = "Which Regional indicator had the highest average Happiness score across all years? Please calculate it."
    response_1 = agent.run(
        my_query_1,
        reset=False,
        additional_args={
            "DATA_PATH": DATA_PATH,
            "OUTPUT_DIR": OUTPUT_DIR,
        }
    )
    print("\n--- My Query 1 ---")
    print(response_1)

    my_query_2 = "Show me the top 10 countries by GDP per capita in 2020."
    response_2 = agent.run(
        my_query_2,
        reset=False,
        additional_args={
            "DATA_PATH": DATA_PATH,
            "OUTPUT_DIR": OUTPUT_DIR,
        }
    )
    print("\n--- My Query 2 ---")
    print(response_2)

if __name__ == "__main__":
    print("Starting project_07.py...")

    if not api_key:
        raise ValueError("OPENAI_API_KEY was not found.")

    print("API key loaded.")

    print("Running guided queries...")
    run_guided_queries()

    print("Running custom queries...")
    run_my_queries()

    plot_path = OUTPUT_DIR / "happiness_by_region.png"

    if plot_path.exists():
        print(f"\nPlot saved successfully: {plot_path}")
    else:
        print(f"\nPlot was not found. Check the agent response and path: {plot_path}")

    print("Project finished.")

    # --- Reflection ---
    #
    # 1. In Query 3, the agent should use the p-value from compute_correlation
    #    to decide whether the result is statistically significant. A common threshold
    #    is p < 0.05.
    # In Query 3, the agent explained that the correlation was statistically significant
    # because the p-value was 0.0, which is lower than the common threshold of 0.05.
    # The  p-value value shows that there is a statistically significant positive correlation
    # between GDP prr capita and Happines score.


    # 2. Did any of the agent's responses surprise you — either by being more capable than
    #    you expected, or less? Describe one specific example.
    # The agent surprised me when was showing the code execution failed.  But it was the reasoning process not the code that was failing.
    # The agent was able to correct itself on the next steps after discovering the instruction to load the data. Also it was interesting to see that it is
    # case sensitive when looking for the country column, it first looked for "country" and then for "Country". It was also interesting to see that the agent was able to use the compute_correlation tool to get the p-value and explain the statistical significance of the correlation result.
    
    # 3. What one additional tool would make this agent meaningfully more useful?
    #    Describe what it would do and what kind of question it would help the agent answer.
    #    (You do not need to implement it.)
    # An additional tool that would make this agent more useful is a plotting tool that can create different types of grapghs based on the data. This would allow the agent to visually represent trends and relationships in the data, making it easier for users to understand complex information. 