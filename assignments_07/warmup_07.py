import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from scipy.stats import pearsonr

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# ---Lesson 02 --- #Q1

# Define the following Python function:

def celsius_to_fahrenheit(celsius: float) -> str:
    """Convert a Celsius temperature to Fahrenheit and return it as a formatted string."""
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C is {fahrenheit}°F"

# Next, write the JSON schema dictionary that describes this function to an LLM — exactly like the get_current_time schema in the lesson. Your schema should include name, description, and parameters (with a celsius property of type "number").
# Finally, call the function directly (not through an agent yet) with 0, 100, and -40 and print each result.


#JSON schema dictionary for the celsius_to_fahrenheit function
celsius_to_fahrenheit_schema = {
    "type": "function",
    "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {
            "type": "object",
            "properties": {
                "celsius": {
                    "type": "number",
                    "description": "Enter the temperature in Celsius."
                }
            },
            "required": ["celsius"],
        }
    }
}

print("\n---Q1--")
print(celsius_to_fahrenheit(0))
print(celsius_to_fahrenheit(100))
print(celsius_to_fahrenheit(-40))


# ---Lesson 02 --- #Q2
# Copy the run_agent function from the lesson — the one that uses get_current_time as its only tool. Before calling it, add a comment block that predicts:
# 1. Will calling run_agent("Convert 100 degrees Celsius to Fahrenheit") trigger a tool call? Why or why not?
# I think that the run_agent function will trigger a tool call because it can convert celsius to fahrenheit using the celsius_to_fahrenheit function, which is defined as a tool in the agent's tools list. 

# 2. How many API calls will be made to answer this query?
# It will take 2 API calls to answer this query.  It will have the tools_choice so the model reads the conversation, decides if needs a tool, returns a message that may include tool_calls, so the first answer is not a final answer.  Then we will call the API again with the updated messages list.

# Then call run_agent("Convert 100 degrees Celsius to Fahrenheit") and print the result. Was your prediction correct?


def get_current_time() -> str:
    """Return the current local time as a formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


get_current_time_schema = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "Get the current local time.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
}


tools = [get_current_time_schema]



def run_agent(user_prompt: str) -> str:
    '''Run a minimal ReAct-style agent for a single user prompt.'''

    SYSTEM_PROMPT = '''You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time.'''
    
    # Step 1: start the conversation with system and user messages
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]


    # Step 2: first API call - the model decides whether to call a tool
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    print("First response received from model...")
    print(first_response)
    first_message = first_response.choices[0].message

    # Record what the model said so far
    messages.append(
        {
            'role': 'assistant',
            'content': first_message.content,
            'tool_calls': first_message.tool_calls,
        }
    )

    # Step 3: check if the model requested any tools
    if first_message.tool_calls:
        print("Agentic mode engaged...")
        for tool_call in first_message.tool_calls:
            function_name = tool_call.function.name
            # In this example we only have one tool: get_current_time
            if function_name == 'get_current_time':
                tool_result = get_current_time()
            else:
                tool_result = f'Error: unknown tool {function_name}.'

            # Print for debugging so we can see what happened
            print('Tool called:', function_name)
            print('Tool result:', tool_result)

            # Step 3b: append the tool output so the model can see it
            messages.append(
                {
                    'role': 'tool',
                    'tool_call_id': tool_call.id,
                    'name': function_name,
                    'content': tool_result,
                }
            )

        # Step 4: second API call - model sees the tool result and gives final answer
        second_response = client.chat.completions.create(
            model='gpt-4.1-mini',
            messages=messages,
        )
        print("Second response received from model...")
        print(second_response)

        final_message = second_response.choices[0].message
        return final_message.content or ''
    else:
        print("No tools needed....")

    # If there were no tool calls, the first response was already the final answer
    return first_message.content or ''

if __name__ == "__main__":
    result = run_agent("Convert 100 degrees Celsius to Fahrenheit")
    print("n---Q2---")
    print(result)   


# ---Lesson 02 --- #Q3

# Now extend the agent to support both tools. Update your tools list to include celsius_to_fahrenheit (using the schema from Q1), and update run_agent to dispatch it when the model requests it.
# Test the extended agent on both of these queries:

# response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
# print("Response A:", response_a)

# response_b = run_agent("What is the boiling point of water in plain English?")
# print("Response B:", response_b)

# Add a comment after each print() explaining whether a tool was called and why.



def run_agent(user_prompt: str) -> str:

    client = OpenAI()

    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time.
                     Use the tool get_current_time whenever a user asks about the time."""
    SYSTEM_PROMPT = """You are a simple assistant that can tell the current time
                     and convert Celsius temperatures to Fahrenheit.
                     Use the tool get_current_time whenever a user asks about the time.
                     Use the tool celsius_to_fahrenheit whenever a user asks for a Celsius to Fahrenheit conversion."""
    
    tools = [get_current_time_schema]
    tools = [celsius_to_fahrenheit_schema]

    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

   
    first_response = client.chat.completions.create(
        model='gpt-4.1-mini',
        messages=messages,
        tools=tools,
        tool_choice='auto',  # model chooses whether to use a tool
    )

    first_message = first_response.choices[0].message

    if not first_message.tool_calls:
        return first_message.content

    messages.append(first_message)

    for tool_call in first_message.tool_calls:
        function_name = tool_call.function.name

        if function_name == "get_current_time":
            tool_result = get_current_time()

        elif function_name == "celsius_to_fahrenheit":
            arguments = json.loads(tool_call.function.arguments)
            tool_result = celsius_to_fahrenheit(arguments["celsius"])

        else:
            tool_result = f"Error: unknown tool {function_name}."

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(tool_result)
        })

    second_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return second_response.choices[0].message.content

print("\n---Q3---")
response_a = run_agent("What is 37 degrees Celsius in Fahrenheit?")
print("Response A:", response_a)

# After the model reasons, the model returned an answer using a formula.  No tool was called. The output showed "No tools needed...." the model thought no tool was needed because it knew the formula to convert Celsius to Fahrenheit.

response_b = run_agent("What is the boiling point of water in plain English?")
print("Response B:", response_b)

# After the model reasons, the model returned an answer. No tool was called. The output showed "No tools needed...." the model thought no tool was needed because it knew the boiling point of water without needing to call a tool.


# ---Lesson 03 --- #Q4

# For Q4-Q6, use the full CsvManager class and run_agent_cycle setup from the lesson (copy them into your file). You will extend them.
# Q4

# The lesson ended with the agent hitting the tool-round limit when asked to compute a correlation, because no tool existed for it. Fix that.

# Add a compute_correlation method to CsvManager:

# def compute_correlation(self, col1: str, col2: str):
#     """
#     Compute the Pearson correlation between two columns in the loaded DataFrame.
#     Returns the correlation coefficient and p-value.
#     """
#     # your code here

# Use scipy.stats.pearsonr to compute the correlation. Return a dictionary with keys "col1", "col2", "pearson_r", and "p_value" (round each float to 4 decimal places). Return {"error": "..."} if either column is not found or no CSV is loaded.
# Also add its JSON schema entry to tools_schema and its entry to node_tools.

from dotenv import load_dotenv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openai import OpenAI
from scipy.stats import pearsonr

RESOURCES_DIR = Path("resources")


def compute_correlation(self, col1: str, col2: str):
    """
    Compute the Pearson correlation between two columns in the loaded DataFrame.
    Returns the correlation coefficient and p-value.
    """

class CsvManager:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self.df = None
        self.csv_name = None

    # --- Small internal helpers --------------------------------------

    def _normalize_csv_name(self, filename: str) -> str:
        if not filename.lower().endswith(".csv"):
            return filename + ".csv"
        return filename

    def _available_csv_files(self) -> list[str]:
        if not self.resources_dir.exists():
            return []
        return sorted(
            [
                p.name
                for p in self.resources_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".csv"
            ]
        )

    def _ensure_loaded(self):
        if self.df is None:
            files = self._available_csv_files()
            example = files[0] if files else "your_file.csv"
            return {
                "error": (
                    "No CSV is loaded yet. First load one from resources/. "
                    f"For example: load_csv '{example}'."
                )
            }
        return None

    # --- Tools (public methods) --------------------------------------

    def list_csv_files(self):
        """
        List available CSV files in resources/.
        """
        files = self._available_csv_files()
        if not files:
            return {
                "message": (
                    "No CSV files found in resources/. "
                    "Create a resources/ folder and put one or more .csv files inside it."
                ),
                "files": [],
            }
        return {"files": files}

    def load_csv(self, filename: str):
        """
        Load a CSV file from resources/ and make it the active dataset.
        """

        filename = self._normalize_csv_name(filename)
        path = self.resources_dir / filename

        if not path.exists():
            return {
                "error": f"Could not find '{filename}' in resources/.",
                "available_files": self._available_csv_files(),
            }

        try:
            # First try normal CSV loading
            self.df = pd.read_csv(path)

        except Exception:
            try:
                # Try automatic delimiter detection
                self.df = pd.read_csv(path, sep=None, engine="python")

            except Exception:
                try:
                    # Try tab-separated files
                    self.df = pd.read_csv(path, sep="\t")

                except Exception as e:
                    return {
                        "error": f"Could not load CSV file: {str(e)}"
                    }

        self.csv_name = filename

        return {
            "message": f"Loaded {filename} with shape {self.df.shape}.",
            "columns": self.df.columns.tolist(),
        }

    def get_columns(self):
        """
        Return column names for the currently loaded CSV.
        """
        error = self._ensure_loaded()
        if error:
            return error
        return self.df.columns.tolist()

    def summarize_columns(self, columns: list[str] | None = None):
        """
        Return basic summary stats for one or more columns.

        If columns is None, summarize all columns.
        Uses pandas.describe(include="all") to stay simple and readable.
        """
        error = self._ensure_loaded()
        if error:
            return error

        if columns is None:
            data = self.df
        else:
            missing = [c for c in columns if c not in self.df.columns]
            if missing:
                return {"error": f"These columns are not in the data: {missing}"}
            data = self.df[columns]

        summary = data.describe(include="all").transpose().round(3)
        return summary.to_dict()

    def describe_column(self, column: str):
        """
        Simple summary for a single column using pandas.describe().
        """
        error = self._ensure_loaded()
        if error:
            return error

        if column not in self.df.columns:
            return {"error": f"'{column}' is not a column. Options: {self.df.columns.tolist()}"}

        s = self.df[column]
        summary = s.describe().to_dict()

        cleaned = {}
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                cleaned[key] = round(value, 3)
            else:
                cleaned[key] = value

        return cleaned

    def plot_data(self, y: str, x: str | None = None, plot_type: str = "line"):
        """
        Plot from the active CSV.
    
        - If x is None: plot y vs row index.
        - If x is provided: plot y vs x.
        """
        error = self._ensure_loaded()
        if error:
            return error
    
        if plot_type not in ["scatter", "line"]:
            return "Error: I can only do 'scatter' or 'line'."
    
        if y not in self.df.columns:
            return f"Error: column '{y}' is not in {self.df.columns.tolist()}"
    
        # If someone accidentally passes x == y, treat it like "plot y"
        if x == y:
            x = None
    
        # Scatter needs x
        if plot_type == "scatter" and x is None:
            return "Error: scatter plots need both x and y columns."
    
        title_csv = self.csv_name or "current CSV"
    
        if x is None:
            ax = self.df[y].plot(kind="line")
            ax.set_title(f"{title_csv} | Line plot: {y} vs row index")
            plt.show()
            return f"Plotted {y} vs row index as a line plot."
    
        if x not in self.df.columns:
            return f"Error: column '{x}' is not in {self.df.columns.tolist()}"
    
        ax = self.df.plot(x=x, y=y, kind=plot_type)
        ax.set_title(f"{title_csv} | {plot_type.title()} plot: {y} vs {x}")
        plt.show()
        
        return f"Plotted {y} vs {x} as a {plot_type}."
    
    def compute_correlation(self, col1: str, col2: str):
        """
        Compute the Pearson correlation between two columns in the loaded DataFrame.
        Returns the correlation coefficient and p-value.
        """

        # Make sure a CSV has been loaded first.
        error = self._ensure_loaded()
        if error:
            return error

        # Make sure both columns exist in the DataFrame.
        if col1 not in self.df.columns:
            return {"error": f"'{col1}' is not a column. Options: {self.df.columns.tolist()}"}

        if col2 not in self.df.columns:
            return {"error": f"'{col2}' is not a column. Options: {self.df.columns.tolist()}"}

        # Remove rows where either column has missing values.
        clean_df = self.df[[col1, col2]].dropna()

        if clean_df.empty:
            return {"error": "No valid rows available after removing missing values."}

        # Calculate Pearson correlation and p-value.
        r, p_value = pearsonr(clean_df[col1], clean_df[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p_value), 4)
        }

print("Class defined")


csv_backend = CsvManager(RESOURCES_DIR)

node_tools = {
    "list_csv_files": csv_backend.list_csv_files,
    "load_csv": csv_backend.load_csv,
    "get_columns": csv_backend.get_columns,
    "summarize_columns": csv_backend.summarize_columns,
    "describe_column": csv_backend.describe_column,
    "plot_data": csv_backend.plot_data,
    "compute_correlation": csv_backend.compute_correlation,
}

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "list_csv_files",
            "description": "List available CSV files in the resources/ folder.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_csv",
            "description": "Load a CSV file from the resources/ folder and make it the active dataset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "CSV filename in resources/, e.g. 'bike_commute.csv'.",
                    }
                },
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_columns",
            "description": "Get the column names of the currently loaded CSV.",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_columns",
            "description": "Show basic summary statistics for columns (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of column names. If omitted, summarize all columns.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_column",
            "description": "Show basic summary statistics for a single column (uses pandas.describe).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column name to describe.",
                    }
                },
                "required": ["column"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_data",
            "description": "Plot data from the active CSV. If only y is provided, plot y vs row index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "y": {"type": "string", "description": "Column name for y-axis."},
                    "x": {"type": "string", "description": "Optional column name for x-axis."},
                    "plot_type": {
                        "type": "string",
                        "enum": ["scatter", "line"],
                        "description": "Type of plot to create.",
                    },
                },
                "required": ["y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_correlation",
            "description": "Compute the Pearson correlation coefficient and p-value between two numeric columns in the active CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "col1": {
                        "type": "string",
                        "description": "The first numeric column name."
                    },
                    "col2": {
                        "type": "string",
                        "description": "The second numeric column name."
                    }
                },
                "required": ["col1", "col2"],
            },
        },
    }
]


# ---Lesson 03 --- #Q5

# Recreate the scenario from the lesson that hit the tool-round limit. Set up the agent with the system prompt from the lesson, then run:

# messages = [{"role": "system", "content": SYSTEM_PROMPT}]
# result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
# print(result)

# With the new tool in place, the agent should now succeed. Print the agent's final response.

SYSTEM_PROMPT = (
    "You are a small data assistant for CSV files stored in resources/. "
    "Use the available tools to do any data work (do not guess). "
    "If no CSV is loaded yet, load one first (or list available CSV files). "
    "Keep answers short and student-friendly."
)

def run_agent_cycle(messages, user_text, max_tool_rounds=5):
    """
    Run through one react-agent loop using a simple tool-using agent.
    `messages` parameter will usually just contain a system prompt, 
    and then user text will be appended.  

    The loop has three main steps:

    REASON:
      - Call the model with the conversation so far.
      - The model either replies normally, or asks to call a tool from tool set.

    ACT:
      - If tools are requested, run the Python functions

    OBSERVE:
      - Append each requested tool result back into the LLMs conversation history.
      - On the next iteration, the model reads those tool call results and determines
        whether it has reached the goal.

    Stop condition:
      - If the model returns an assistant message with no tool calls, this is the 
        final answer for this react cycle, this implies that reasoning alone without 
        tool calls was enough.  
      - max_tool_rounds is a safety cap to prevent infinite loops.
    """
    messages.append({"role": "user", "content": user_text})

    def observe_tool_result(tool_call_id, result):
        """
        Return a tool's return value as a message that can be appended to the
        LLMs conversation history. The model will read this tool output on the next
        REASON step.
        """
        content = json.dumps(result, default=str) if not isinstance(result, str) else result
        tool_message = {"role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": content,}
        return tool_message

    for loop_idx in range(max_tool_rounds):
        # REASON: call the model
        # Here it will make use of any previous tool outputs it appended ("observed")
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=messages,
            tools=tools_schema,
        )

        msg = response.choices[0].message

        # Append the assistant message to the conversation history.
        # Use a plain dict so `messages` stays simple and inspectable.
        assistant_entry = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            assistant_entry["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_entry)

        # No tool calls means the model is answering directly.
        if not msg.tool_calls:
            return msg.content 

        # ACT + OBSERVE: run each tool call, then append its result.
        # Note there may be multiple tool calls
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments or "{}")

            print(f"ACT: {name}({tool_args})")

            fn = node_tools.get(name)
            if fn is None:
                result = {"error": f"Tool '{name}' not found."}
            else:
                try:
                    result = fn(**tool_args) if tool_args else fn()
                except Exception as e:
                    print(f"Tool error in {name}: {type(e).__name__}: {e}")
                    result = {"error": f"Tool '{name}' failed: {type(e).__name__}: {e}"}
                    
            # OBSERVE: append the tool result back into the conversation history.
            messages.append(observe_tool_result(tool_call.id, result))
            
            # After we appending information about all tool outputs, we loop back and REASON again.

    return "I hit the tool-round limit. Try a simpler request."


print("\n---Q5---")
messages = [{"role": "system", "content": SYSTEM_PROMPT}]
result = run_agent_cycle(messages, "Load bike_commute.csv and compute the correlation between avg_traffic_density and avg_speed_kmh.")
print(result)


# ---Lesson 03 --- #Q6

# After Q5 runs, print the full messages list. Each item in the list is a dictionary with a "role" key. Add a comment above the print that identifies what each role (system, user, assistant, tool) represents in the ReAct loop.

# The roles in the messages list represent the following:
# "system": Gives instructions to the agent. 
# "user": Contains what. is needed and gives the instructions how to do it.
# "assistant": This include the reasoning and tool calls.
# "tool": Gives the output of the tool calls that the assistant requested.


# Hint:
print("\n---Q6---")
import json
print(json.dumps(messages, indent=2, default=str))


# Lesson 04: smolagents

# For Q7-Q9, use the smolagents setup from the lesson (ToolCallingAgent, CodeAgent, OpenAIServerModel, and the @tool decorator). Reuse the CsvManager instance from above.

from smolagents import tool, ToolCallingAgent, CodeAgent, OpenAIServerModel

# ---Lesson 04 --- #Q7

# Re-wrap compute_correlation as a smolagents tool using the @tool decorator. The decorated function should call csv_manager.compute_correlation(col1, col2) under the hood.
# After defining it, run:

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)


@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """
    Compute the Pearson correlation coefficient and p-value between two numeric columns in the active CSV.

    Args:
        col1 (str): The name of the first numeric column.
        col2 (str): The name of the second numeric column.

    Returns:
        dict: A dictionary with the column names, Pearson correlation coefficient, and p-value.
    """
    return csv_backend.compute_correlation(col1, col2)
   

print("\n---Q7---")
print(compute_correlation.description)


# Add a comment comparing what smolagents generates automatically to the JSON schema you wrote manually in Q4. What information does smolagents need from you (the developer) in order to produce a good description?

# The schema in Q4 was written manually, describing the tool name, description, parameters and arguments.  Whith smolagents the @tool decorator reads teh function,type hints and docstring to generate the toold description automatically.   The smolagents needs from me, the developer, the hints and a strong docstring to produce a good description.  


# ---Lesson 04 --- #Q8

# Create both a ToolCallingAgent and a CodeAgent using the same TOOLS list from the lesson (including your new compute_correlation tool) and the same OpenAIServerModel. Run the following prompt through both:

# prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

# response_tool = tool_agent.run(prompt)
# response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_manager})


# Print both responses, then add a comment block answering:

# What did each agent actually produce? Did the ToolCallingAgent change the dot color? Did the CodeAgent?
# What does this reveal about when each type of agent is more useful?


# Q8 setup: smolagents tool wrappers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



@tool
def list_csv_files() -> dict:
    """List available CSV files in the resources folder.

    Returns:
        A dictionary containing available CSV file names.
    """
    return csv_backend.list_csv_files()


@tool
def load_csv(filename: str) -> dict:
    """Load a CSV file from the resources folder.

    Args:
        filename: The CSV filename, such as bike_commute.csv.

    Returns:
        A dictionary with a loading message and the CSV columns.
    """
    return csv_backend.load_csv(filename)


@tool
def get_columns() -> list:
    """Get the column names for the currently loaded CSV.

    Returns:
        A list of column names.
    """
    return csv_backend.get_columns()


@tool
def describe_column(column: str) -> dict:
    """Describe one column in the loaded CSV.

    Args:
        column: The column name to describe.

    Returns:
        Summary statistics for the selected column.
    """
    return csv_backend.describe_column(column)


@tool
def plot_data(y: str, x: str = None, plot_type: str = "line") -> str:
    """Plot data from the active CSV file.

    Args:
        y: The column name to use for the y-axis.
        x: The optional column name to use for the x-axis.
        plot_type: The type of plot to create. Use "line" or "scatter".

    Returns:
        A message describing whether the plot was created successfully.
    """
    return csv_backend.plot_data(y=y, x=x, plot_type=plot_type)
    


# Q8

TOOLS = [
    list_csv_files,
    load_csv,
    get_columns,
    describe_column,
    plot_data,
    compute_correlation,
]

model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)

tool_agent = ToolCallingAgent(
    tools=TOOLS,
    model=model,
    max_steps=6
)

code_agent = CodeAgent(
    tools=TOOLS,
    model=model,
    additional_authorized_imports=["pandas", "matplotlib", "matplotlib.pyplot"],
    max_steps=6
)

prompt = "Load bike_commute.csv. Plot avg_heart_rate vs duration_min as a scatter plot with green dots."

response_tool = tool_agent.run(prompt)
response_code = code_agent.run(prompt, additional_args={"csv_manager": csv_backend})

print("\n--- Q8 ToolAgent ---")
print(response_tool)

print("\n--- Q8 CodeAgent ---")
print(response_code)


# The ToolCallingAgent created the scatter plot and mentioned green dots.
# This means the agent was able to use the available plotting tool to complete the request.
#
# The CodeAgent also created the scatter plot by generating and running Python code.
# The response was more generic, but the CodeAgent had the flexibility to write custom matplotlib code if needed.
#
# ToolCallingAgent is useful when predefined tools already support the task.
# CodeAgent is more useful for flexible or open-ended problems where the agent may need
# to write custom Python code.



# ---Lesson 04 --- #Q9

# Add a comment block at the bottom of your warmup file answering both questions:

# 1. Describe a task where a ToolCallingAgent would be a better choice than a CodeAgent. What property of the task makes it a good fit for a tool-based approach?
# A ToolCallingAgent is a better choice for tasks has clear inputs, a known operation, and a predictable output. The agent does not need to invent new code; it only needs to choose the right predefined tool, pass the correct column names, and report the result. This makes the workflow more controlled, safer, and easier to debug than letting a CodeAgent generate and execute arbitrary Python.


# 2. What is one meaningful risk of using a CodeAgent that does not apply to a ToolCallingAgent? (Think about what's actually happening when the agent generates and runs code.)

# One meaningful risk of using a CodeAgent is that it generates and executes new code, so it might accidentally do something unintended: overwrite files, read sensitive data, install packages, run expensive computations, or produce unsafe side effects.

# A ToolCallingAgent is more constrained. It can only call the specific tools the developer exposed, with the parameters those tools allow, so its action space is much smaller and easier to control.
