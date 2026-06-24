# Prefect Question 1
# In a comment block, answer: what is the difference between a @task and a @flow in Prefect? You have a helper function that converts a temperature from Celsius to Fahrenheit -- a pure, in-memory calculation with no I/O. Would you decorate it with @task? Why or why not?
# The difference is that @task is for a single unit of task, like loading data, calling an API or writing a file.  A flow combines different tasks in order and manages them as a whole.   I would not decorate the conversion from Celsius to Fahrenheit with @task because it is a pure, in-memory calculation with no I/O.  It does not need to be retried or logged, and it does not have any side effects that would require Prefect's orchestration features.  A Python function would be enough foir this simple calculation.

# Prefect Question 2
# Write the decorator (just the decorator line, not the full function) for a task named call_api that retries up to 3 times with a 30-second delay between attempts.
#@task(retries=3, retry_delay_seconds=30)

# Prefect Question 3
# You run your pipeline and the Prefect UI shows: extract is Completed, transform is Failed, load never ran. In a comment block, describe: where in the UI do you look to understand what went wrong, and what specific information would you expect to find there?
# In the UI I will have to click on the transform task to see the logs and error messages. I would find the stack trace or error message that indicates what went wrong during the transform step, such as an exception raised by the OpenAI API or a data processing error. This information will help me identify the root cause of the failure and fix it before re-running the pipeline.


# Production Patterns
# Production Question 1
# In a comment block, explain what raise_for_status() does and why it is better than writing if response.status_code != 200: print("error") in a pipeline task. What happens to downstream tasks in each case when the API returns a 500 error?
# Wioth raise_for_status() the task can handle temporary API problems and fail clearly if the API response is not successful. Without it, the task would print "error" but continue running, which could lead to more errors downstream when the code tries to access data that wasn't properly retrieved. With raise_for_status(), the task will raise an exception immediately when it encounters a 500 error, preventing downstream tasks from running with invalid data and making it easier to identify and fix the issue.

# Production Question 2
# Your pipeline uploads results to final/{today}/weather_etl.json with overwrite=True. The pipeline crashes halfway through the transform step. You fix the bug and re-run it from the beginning. In a comment block, explain: what does overwrite=True protect you from in this scenario, and what would happen without it?
# The load task converts the enriched records into JSON and uploads the result to Azure Blob Storage. The output is saved in my pipeline-data container under final/2026-06-15/weather_etl.json. I used overwrite=True, so if I rerun the pipeline, it updates the same file path instead of creating conflicts. This is important because if the pipeline crashes halfway through the transform step, I can fix the bug and re-run it without worrying about duplicate files or manual cleanup. Without overwrite=True, I would end up with multiple files for the same date, which could lead to confusion and make it harder to manage the outputs.

# Production Question 3
# Write a task stub -- just the function signature, decorator, and a single log line -- that uses get_run_logger() to log an INFO message saying how many records were loaded. The function should accept records (a list) and blob_path (a string) as arguments.
from prefect import task, get_run_logger

@task
def load(records: list, blob_path: str):
    logger = get_run_logger()
    logger.info(f"Loaded {len(records)} records to {blob_path}")

