import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from datetime import datetime
import os
from sqlalchemy import create_engine
from typing import Dict, List, Tuple, Optional
import argparse


def load_and_filter_data(file_path: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load data from CSV and apply initial filtering.

    Args:
        file_path: Path to the CSV file
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        Filtered DataFrame
    """
    df = pd.read_csv(file_path)
    df['end'] = pd.to_datetime(df['end'], format="%Y-%m-%d %H:%M:%S")
    return df[(df['end'] >= start_date) & (df['end'] < end_date)]


def apply_data_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply accuracy and power filters to the DataFrame.

    Args:
        df: Input DataFrame

    Returns:
        Filtered DataFrame
    """
    # Filter by accuracy
    df_filtered = df[df['global_accuracy_percentage'] > 90].copy()

    # Calculate and filter by power
    df_filtered['power'] = df_filtered['total_energy'] / (df_filtered['elapsed'] * df_filtered['total_nodes'])
    df_filtered = df_filtered[df_filtered['power'] <= 2700]

    # Handle NaN values
    df_filtered['total_energy'] = df_filtered['total_energy'].where(df_filtered['total_energy'].notna(), None)

    return df_filtered


def anonymize_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Anonymize account names by mapping them to project_0, project_1, etc.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with anonymized accounts
    """
    df = df.copy()
    unique_accounts = df['account'].unique()
    account_mapping = {account: f"project_{i}" for i, account in enumerate(unique_accounts)}
    df['account_'] = df['account'].map(account_mapping)
    return df


def create_pie_chart(df: pd.DataFrame, cluster: str, quantity: str) -> None:
    """
    Create and save a pie chart for the given cluster and quantity.

    Args:
        df: Input DataFrame
        cluster: Cluster name
        quantity: Quantity to plot ('node_hours' or 'total_energy')
    """
    # Prepare data based on quantity type
    if quantity == 'node_hours':
        df = df.copy()
        df[quantity] = df['elapsed'] * df['total_nodes'] / 3600.0
        df_pie = df[['account_', 'node_hours']]
        df_pie_real = df[['account', 'node_hours']]
        quantity_graph = 'node-hours'
        unit = 'nh'
    else:
        df_pie = df[['account_', 'total_energy']]
        df_pie_real = df[['account', 'total_energy']]
        unit = 'GJ'
        quantity_graph = 'energy'

    # Group and get top 5
    pie = df_pie.groupby(['account_']).sum().reset_index()
    pie_real = df_pie_real.groupby(['account']).sum().reset_index()

    # Save real account data
    top_10_real = pie_real.nlargest(5, quantity).reset_index()
    top_10_real.to_csv(f"top10_real_{cluster}{quantity}.csv")

    # Prepare data for pie chart
    top_10 = pie.nlargest(5, quantity).reset_index()
    others = pie.loc[:, ['account_', quantity]].drop(top_10.index)

    # Add "others" category
    top_10.loc[len(top_10), quantity] = others[quantity].sum()
    top_10.loc[len(top_10) - 1, 'account_'] = 'others'

    # Create pie chart
    colors = plt.cm.tab20.colors[:6]
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        top_10[quantity],
        autopct=lambda pct: f'{pct:.1f}%',
        colors=colors,
        pctdistance=0.6,
        labeldistance=1.2,
        textprops={'fontsize': 8}
    )

    # Create legend with absolute values
    if quantity == 'node_hours':
        legend_labels = [f"{acc}: {val:.0f} {unit}" for acc, val in zip(top_10['account_'], top_10[quantity])]
    else:
        legend_labels = [f"{acc}: {val:.2f} {unit}" for acc, val in zip(top_10['account_'], round(top_10[quantity] / (10.0**9), 2))]

    ax.legend(wedges, legend_labels, fontsize=10, title="Projects", bbox_to_anchor=(1, 0.25, 0.5, 0.5))

    plt.ylabel(ylabel='', fontsize=18)
    plt.title(f"Top 5 {quantity_graph} {cluster} projects", fontsize=16)
    plt.savefig(f"results/Top5_{quantity_graph}_{cluster}.png", bbox_inches='tight', dpi=200)
    plt.show()


def create_histogram(df: pd.DataFrame, cluster: str, postfix: str = '') -> None:
    """
    Create and save a histogram for the given cluster.

    Args:
        df: Input DataFrame
        cluster: Cluster name
        postfix: Optional postfix for the filename
    """
    plt.hist(df['total_energy'], color='blue', edgecolor='black', bins=100)
    plt.title(f"{cluster}", fontsize=30)
    plt.xlabel('Energy [J]', fontsize=24)
    plt.ylabel('# jobs', fontsize=24)
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize=24)
    plt.xlim(1, 1e9)
    plt.ylim(1, 10**5)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(24)

    plt.savefig(f"results/histogram_{cluster}{postfix}.png", bbox_inches='tight', dpi=200)
    plt.show()


def format_with_apostrophes(n: int) -> str:
    """
    Format a number with apostrophes as thousand separators.

    Args:
        n: Number to format

    Returns:
        Formatted string
    """
    return format(n, ',').replace(',', "'")


def create_heatmap(df: pd.DataFrame, cluster: str, threshold: int) -> None:
    """
    Create and save a heatmap of job size vs energy.

    Args:
        df: Input DataFrame
        cluster: Cluster name
        threshold: Energy threshold for filtering
    """
    # Prepare data
    df = df.copy()
    df['node_hours'] = df['elapsed'] * df['total_nodes'] / 3600.0
    sample_data = df[df['total_energy'] < threshold].copy()

    # Create energy bins
    energy_max = int(sample_data['total_energy'].max())
    delta = energy_max // 20
    bins = list(range(0, energy_max, delta))

    # Create bin labels
    labels = [f"[{format_with_apostrophes(bins[i])}-{format_with_apostrophes(bins[i+1])}]"
              for i in range(len(bins) - 1)]

    # Bin the energy data
    sample_data['range_energy'] = pd.cut(sample_data['total_energy'], bins=bins, labels=labels, include_lowest=True)

    # Create heatmap data
    heatmap_data = sample_data[['total_nodes', 'range_energy', 'node_hours']].groupby(
        ['total_nodes', 'range_energy']).sum().reset_index()
    heatmap_data['node_hours'] = heatmap_data['node_hours'].replace(0, np.nan)

    # Pivot for heatmap
    heatmap_matrix = heatmap_data.pivot(index='total_nodes', columns='range_energy', values='node_hours')

    # Plot heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(heatmap_matrix, cmap='coolwarm', linewidths=0.5, annot=True, fmt=".0f")

    plt.title(f"{cluster}: Job Size [# nodes] vs Energy [J] (Color = Node_hours)", fontsize=20)
    plt.ylabel("Job Size [#nodes]", fontsize=18)
    plt.xlabel("Energy [J]", fontsize=18)
    plt.xticks(rotation=90, fontsize=18)
    plt.yticks(fontsize=18)

    plt.savefig(f"results/heatmap_sizeVSenergy_{cluster}.png", bbox_inches='tight', dpi=200)
    plt.show()


def process_cluster_data(file_path: str, cluster_name: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Process data for a single cluster.

    Args:
        file_path: Path to the CSV file
        cluster_name: Name of the cluster
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        Processed DataFrame
    """
    # Load and filter data
    df = load_and_filter_data(file_path, start_date, end_date)

    # Apply filters
    df_filtered = apply_data_filters(df)

    # Anonymize accounts
    df_anonymized = anonymize_accounts(df_filtered)

    return df_anonymized


def main():
    """Main function to run the analysis."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate cluster analysis visualizations')
    parser.add_argument('--cluster', type=str, help='Specific cluster to analyze (e.g., Alps-Clariden, Alps-Daint, Alps-Santis)')
    args = parser.parse_args()

    # Define time window
    start_date = datetime(2025, 2, 1)
    end_date = datetime(2025, 3, 1)

    # Process data for each cluster
    clusters = {
        'Alps-Clariden': 'data/raw_data_clariden_jan_feb_mar.csv',
        'Alps-Daint': 'data/raw_data_alps_daint_jan_feb_mar.csv',
        'Alps-Santis': 'data/raw_data_alps_santis_jan_feb_mar.csv'
    }

    # Process each cluster
    cluster_dfs = {}
    if args.cluster:
        if args.cluster not in clusters:
            print(f"Error: Cluster '{args.cluster}' not found. Available clusters: {', '.join(clusters.keys())}")
            return
        file_path = clusters[args.cluster]
        cluster_dfs[args.cluster] = process_cluster_data(file_path, args.cluster, start_date, end_date)
    else:
        for cluster_name, file_path in clusters.items():
            cluster_dfs[cluster_name] = process_cluster_data(file_path, cluster_name, start_date, end_date)

    # Create visualizations for each cluster
    for cluster_name, df in cluster_dfs.items():
        # Create pie charts
        create_pie_chart(df, cluster=cluster_name, quantity='total_energy')
        create_pie_chart(df, cluster=cluster_name, quantity='node_hours')

        # Create histogram
        create_histogram(df, cluster_name)

        # Create heatmap with appropriate threshold
        thresholds = {
            'Alps-Clariden': 1800000000,
            'Alps-Daint': 750000000,
            'Alps-Santis': 100000000
        }
        create_heatmap(df, cluster_name, thresholds[cluster_name])

    # Only combine data if no specific cluster was requested
    if not args.cluster:
        # Combine all data
        df_all = pd.concat(list(cluster_dfs.values()))
        df_all.to_csv('df_all.csv')

        # Create visualizations for combined data
        create_pie_chart(df_all, cluster='big 3', quantity='total_energy')
        create_pie_chart(df_all, cluster='big 3', quantity='node_hours')

        # Print node hours sum for Clariden
        print('node_hours sum')
        print(cluster_dfs['Alps-Clariden']['node_hours'].sum())


if __name__ == "__main__":
    """
    usage: graphs.py [-h] [--cluster CLUSTER]
    required: mkdir -p results

    Generate cluster analysis visualizations

    options:
      -h, --help         show this help message and exit
      --cluster CLUSTER  Specific cluster to analyze (e.g., Alps-Clariden, Alps-Daint, Alps-Santis)
    """
    main()
