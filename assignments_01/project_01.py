from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
from prefect import task, flow, get_run_logger

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "resources" / "happiness_project"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

#Task 1: Load multiple years of data
@task(retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    logger = get_run_logger()
    frames = []

    for year in range(2015, 2025):
        file_path = DATA_DIR / f"world_happiness_{year}.csv"
        logger.info(f"Loading {file_path.name}")

        try:
            df = pd.read_csv(file_path)
        except pd.errors.ParserError:
            logger.info(
                f"Default parser failed for {file_path.name}; trying semicolon format"
            )
            df = pd.read_csv(file_path, sep=";", decimal=",", engine="python")

        if len(df.columns) == 1 and ";" in str(df.columns[0]):
            logger.info(
                f"{file_path.name} appears to be semicolon-delimited; reloading with sep=';'"
            )
            df = pd.read_csv(file_path, sep=";", decimal=",", engine="python")

        df["year"] = year
        frames.append(df)

    merged_df = pd.concat(frames, ignore_index=True)

    output_path = OUTPUT_DIR / "merged_happiness.csv"
    merged_df.to_csv(output_path, index=False)

    logger.info(f"Merged dataset saved to {output_path}")
    logger.info(f"Merged shape: {merged_df.shape}")

    return merged_df



#Task 2: Descriptive Statistics

# Compute and log overall descriptive statistics for happiness_score: mean, median, and standard deviation.
# Then compute and log the mean happiness score grouped by year and by region. Looking at the regional breakdown is often the most interesting part of this dataset -- you may already have a hypothesis about which regions rank highest before you run the numbers.

@task
def describe_stats(merged_df):
    logger = get_run_logger()

    mean_happiness = merged_df["happiness_score"].mean()
    median_happiness = merged_df["happiness_score"].median()
    std_happiness = merged_df["happiness_score"].std()

    logger.info(f"Mean Happiness Score: {mean_happiness}")
    logger.info(f"Median Happiness Score: {median_happiness}")
    logger.info(f"Standard Deviation: {std_happiness}")

    mean_by_year = merged_df.groupby("year")["happiness_score"].mean()
    mean_by_region = merged_df.groupby("region")["happiness_score"].mean().sort_values(ascending=False)

    logger.info(f"Mean happiness grouped by year: {mean_by_year.to_dict()}")
    logger.info(f"Mean happiness grouped by region: {mean_by_region.to_dict()}")

    return {
        "mean": mean_happiness,
        "median": median_happiness,
        "std": std_happiness,
        "mean_by_year": mean_by_year.to_dict(),
        "mean_by_region": mean_by_region.to_dict(),
    }



#Task 3: Visual Exploration

# Create and save the following visualizations to assignments_01/outputs/:
# A histogram of all happiness scores across all years. Save as happiness_histogram.png.
# A boxplot comparing happiness score distributions across years (one box per year). Save as happiness_by_year.png.
# A scatter plot showing the relationship between GDP per capita and happiness score. Save as gdp_vs_happiness.png.
# A correlation heatmap (using sns.heatmap() with annot=True) showing the Pearson correlations between all numeric columns. Save as correlation_heatmap.png.
# Log a message after each plot is saved so you can see the progress in the Prefect dashboard.

@task
def plot_visualizations(merged_df):
    logger = get_run_logger()
    import matplotlib.pyplot as plt
    import seaborn as sns   

    plt.hist(merged_df["happiness_score"], bins=15, edgecolor="blue")
    plt.title("Happiness Scores per Year")
    plt.xlabel("Happiness Score")
    plt.ylabel("Frequency")
    plt.savefig(OUTPUT_DIR / "happiness_histogram.png")
    logger.info("Happiness histogram saved.")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=merged_df, x="year", y="happiness_score")
    plt.title("Happiness Score Distributions by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "happiness_by_year.png")
    plt.close()
    logger.info("Happiness by year boxplot saved.")

    plt.scatter(merged_df["gdp_per_capita"], merged_df["happiness_score"])
    plt.title("GDP per Capita and Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")       
    plt.savefig(OUTPUT_DIR / "gdp_vs_happiness.png")
    logger.info("GDP vs happiness scatter plot saved.")

    corr = merged_df.select_dtypes(include=["number"]).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png")
    logger.info("Correlation heatmap saved.")   



#Task 4: Hypothesis Testing
# The pandemic began in early 2020. Did it affect global happiness scores? Test this directly: run an independent samples t-test comparing happiness scores from 2019 to 2020.
# Log the t-statistic, p-value, the mean happiness for each group, and a plain-language interpretation of the result at alpha = 0.05. Your interpretation should say something meaningful -- not just "we reject the null hypothesis" but what that actually means in terms of this data.
# Add a second test of your choice (for example, comparing two specific regions that you expect to differ based on the descriptive statistics you computed earlier).

@task
def hypothesis_testing(merged_df):
    logger = get_run_logger()
 
    pre_pandemic = merged_df[merged_df["year"] == 2019]["happiness_score"]
    post_pandemic = merged_df[merged_df["year"] == 2020]["happiness_score"] 

    t_stat, p_value = ttest_ind(pre_pandemic, post_pandemic, equal_var=False)
    mean_pre = pre_pandemic.mean()
    mean_post = post_pandemic.mean()    

    logger.info(f"  T-test comparing 2019 and 2020 scores:")
    logger.info(f"  T-statistic: {t_stat}")
    logger.info(f"  P-value: {p_value}")
    logger.info(f"  Mean happiness in 2019: {mean_pre}")
    logger.info(f"  Mean happiness in 2020: {mean_post}")

    if p_value < 0.05:
        pre_post_pandemic_result = "There is evidence that the average happiness scores changed between 2019 and 2020."
    else:
        pre_post_pandemic_result = "There is not enough evidence to conclude that the average happiness scores changed between 2019 and 2020."

    logger.info(pre_post_pandemic_result)

    return {
        "pre_post_result": pre_post_pandemic_result,
        "p_value": p_value,
        "mean_2019": mean_pre,
        "mean_2020": mean_post,
    }

#Task 5: Correlation and Multiple Comparisons

# For each numeric explanatory variable, compute the Pearson correlation with happiness score using scipy.stats.pearsonr and log the coefficient and p-value.
# Each time you run a statistical test at alpha = 0.05, you accept a 5% chance of a false positive -- concluding a relationship is real when it isn't. That's a reasonable risk for a single test. But when you run many tests at once, those small risks add up. If you run 20 independent tests, you'd expect roughly one false positive just by chance, even if none of the relationships are actually real. The more tests you run, the more likely you are to stumble onto something that looks significant but isn't.

# This is called the multiple comparisons problem, and it's one of the most common sources of misleading findings in data analysis. A simple and widely used fix is the Bonferroni correction: divide your significance threshold by the number of tests you ran.
# Count how many correlation tests you performed, then compute:

# adjusted_alpha = 0.05 / number_of_tests
# Log which correlations are significant at the original alpha = 0.05, and which remain significant after applying the correction. You may find that some results that looked significant at first don't hold up under the stricter threshold -- that's a useful finding in itself.

@task
def correlation_analysis(merged_df):
    logger = get_run_logger()

    numeric_cols = merged_df.select_dtypes(include=["number"]).columns.drop("happiness_score")
    num_tests = len(numeric_cols)
    adjusted_alpha = 0.05 / num_tests

    logger.info(f"Bonferroni correction: adjusted alpha = {adjusted_alpha}")

    corr_results = {}
    significant_original = []
    significant_adjusted = []

    for col in numeric_cols:
        pair_df = merged_df[[col, "happiness_score"]].dropna()
        coeff, p_value = pearsonr(pair_df[col], pair_df["happiness_score"])
        corr_results[col] = (coeff, p_value)

        logger.info(f"Correlation between {col} and happiness score: coefficient={coeff}, p-value={p_value}")

        if p_value < 0.05:
            significant_original.append(col)
        if p_value < adjusted_alpha:
            significant_adjusted.append(col)

    logger.info(f"Significant at alpha=0.05: {significant_original}")
    logger.info(f"Significant after Bonferroni correction: {significant_adjusted}")

    return {
        "corr_results": corr_results,
        "adjusted_alpha": adjusted_alpha,
        "significant_adjusted": significant_adjusted,
    }


#Task 6: Summary Report

# Your final task should log a human-readable summary of the key findings from the entire pipeline. Think of it as the "report" step from the lesson -- the thing you'd share with a non-technical colleague. It should include:

# Total number of countries and years in the merged dataset.
# The top 3 and bottom 3 regions by mean happiness score.
# The result of the pre/post-2020 t-test in plain language.
# The variable most strongly correlated with happiness score (after Bonferroni correction).
# Log each of these as a separate logger.info() message so they're easy to find in the Prefect dashboard.


@task
def summary_report(merged_df, test_results, corr_summary):
    logger = get_run_logger()

    num_countries = merged_df["country"].nunique()
    num_years = merged_df["year"].nunique()
    logger.info(f"Number of countries: {num_countries}")
    logger.info(f"Number of years: {num_years}")

    region_means = merged_df.groupby("region")["happiness_score"].mean().sort_values()
    logger.info("Top 3 regions by mean happiness score:")
    for region, score in region_means.tail(3).items():
        logger.info(f"{region}: {score}")

    logger.info("Bottom 3 regions by mean happiness score:")
    for region, score in region_means.head(3).items():
        logger.info(f"{region}: {score}")

    logger.info(test_results["pre_post_result"])

    significant_vars = {
        var: vals
        for var, vals in corr_summary["corr_results"].items()
        if vals[1] < corr_summary["adjusted_alpha"]
    }

    if significant_vars:
        strongest = max(significant_vars.items(), key=lambda x: abs(x[1][0]))
        logger.info(
            f"The most correlated variable with happiness score after Bonferroni correction is "
            f"{strongest[0]} with coefficient {strongest[1][0]}"
        )
    else:
        logger.info("There are no variables significantly correlated with happiness score after Bonferroni correction.")


#Running the Pipeline

# Structure your file so that all tasks are called inside a single @flow function, and the flow runs when the script is executed directly:
# if __name__ == "__main__":
#     happiness_pipeline()
# The full pipeline should be runnable with:

# python project_01.py
# When you run it, it should execute all tasks in order, produce all outputs, and save them to the specified locations. You should be able to run the file multiple times without errors -- each run should overwrite the previous outputs cleanly.

# Once you have it working, try running prefect server start in a separate terminal, then re-run the pipeline and explore the logs and task states in the Prefect dashboard. You'll see exactly what the lesson described. You can also experiment with breaking the pipeline (for example, by changing a file name to something that doesn't exist) to see how Prefect handles errors and retries.

@task
def standardize_columns(merged_df):
    logger = get_run_logger()

    # Drop any malformed single-column header if it still appears
    bad_cols = [col for col in merged_df.columns if ";" in str(col)]
    if bad_cols:
        merged_df = merged_df.drop(columns=bad_cols)
        logger.info(f"Dropped malformed columns: {bad_cols}")

    # Rename columns to consistent names
    merged_df = merged_df.rename(columns={
        "Country": "country",
        "Regional indicator": "region",
        "GDP per capita": "gdp_per_capita",
        "Social support": "social_support",
        "Healthy life expectancy": "healthy_life_expectancy",
        "Freedom to make life choices": "freedom_to_make_life_choices",
        "Generosity": "generosity",
        "Perceptions of corruption": "perceptions_of_corruption",
        "Ranking": "ranking",
    })

    # Build one consistent happiness_score column
    if "Happiness score" in merged_df.columns and "Ladder score" in merged_df.columns:
        merged_df["happiness_score"] = merged_df["Happiness score"].fillna(merged_df["Ladder score"])
    elif "Happiness score" in merged_df.columns:
        merged_df["happiness_score"] = merged_df["Happiness score"]
    elif "Ladder score" in merged_df.columns:
        merged_df["happiness_score"] = merged_df["Ladder score"]

    logger.info(f"Columns after standardization: {merged_df.columns.tolist()}")

    return merged_df

@flow
def happiness_pipeline():
    merged_df = load_and_merge_data()
    merged_df = standardize_columns(merged_df)
    describe_stats(merged_df)
    plot_visualizations(merged_df)
    test_results = hypothesis_testing(merged_df)
    corr_summary = correlation_analysis(merged_df)
    summary_report(merged_df, test_results, corr_summary)


if __name__ == "__main__":
    happiness_pipeline()