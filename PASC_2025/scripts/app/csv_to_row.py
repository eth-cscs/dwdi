import pandas as pd

def generate_jobid_query(input_csv_path, output_csv_path):
    # Read the CSV file
    df = pd.read_csv(input_csv_path)

    # Check if 'jobid' column exists
    if 'jobid' not in df.columns:
        raise ValueError("Input CSV must contain a 'jobid' column.")

    # Drop NaNs and strip whitespace
    df_filtered = df[df['account'] == 'a-a09']

    # Drop NaNs and strip whitespace from jobid
    jobids = df_filtered['jobid'].dropna().astype(str).str.strip()

    # Format job IDs: "jobid1" OR "jobid2" ...
    query = ','.join(f'{jobid}' for jobid in jobids)

    # Write to CSV
    pd.DataFrame({'jobid_query': [query]}).to_csv(output_csv_path, index=False)

generate_jobid_query('csv/df_Alps-Clariden.csv', 'csv/jobid_clariden-a09.csv')