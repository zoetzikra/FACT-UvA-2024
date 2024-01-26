import pandas as pd

# Assuming your CSV file is named 'your_file.csv'
file_path = 'results\\table_all.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Group by 't_type' and 'r_type', and calculate the mean for 'intv-cost_mean' and 'intv-cost_std'
result = df.groupby(['t_type', 'r_type'])[['intv-cost_mean', 'intv-cost_std']].mean().reset_index()

# Convert the DataFrame to a LaTeX table and print it
latex_table = result.to_latex(index=False)
print(latex_table)
