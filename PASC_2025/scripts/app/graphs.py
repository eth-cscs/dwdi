import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime
import seaborn as sns
import numpy as np

def autopct_format(pct, all_values):
    absolute = int(round(pct/100. * sum(all_values)))  # Convert percentage to absolute value
    formatted_absolute = f"{absolute:,.0f}".replace(",", "'")  # Format with apostrophes
    return f"{pct:.1f}%\n({formatted_absolute})"

df_clariden=pd.read_csv('data/raw_data_clariden_jan_feb_mar.csv')
df_daint=pd.read_csv('data/raw_data_alps_daint_jan_feb_mar.csv')
df_santis=pd.read_csv('data/raw_data_alps_santis_jan_feb_mar.csv')

df_clariden['end']=pd.to_datetime(df_clariden['end'],format="%Y-%m-%d %H:%M:%S")
df_daint['end']=pd.to_datetime(df_daint['end'],format="%Y-%m-%d %H:%M:%S")
df_santis['end']=pd.to_datetime(df_santis['end'],format="%Y-%m-%d %H:%M:%S")
df_clariden=df_clariden[(df_clariden['end']>=datetime(2025,2,1))&(df_clariden['end']<datetime(2025,3,1))]
df_daint=df_daint[(df_daint['end']>=datetime(2025,2,1))&(df_daint['end']<datetime(2025,3,1))]
df_santis=df_santis[(df_santis['end']>=datetime(2025,2,1))&(df_santis['end']<datetime(2025,3,1))]

# check negative energy data
# outliers_clariden_raw=df_clariden[df_clariden['total_energy']<=0]
# outliers_clariden_raw.to_csv('negative_energy_clariden_raw.csv')
# outliers_daint_raw=df_daint[df_daint['total_energy']<=0]
# outliers_daint_raw.to_csv('negative_energy_daint_raw.csv')
# outliers_santis_raw=df_santis[df_santis['total_energy']<=0]
# outliers_santis_raw.to_csv('negative_energy_santis_raw.csv')

# plt.figure(figsize=(30, 10))
# plt.plot(df_clariden['end'],df_clariden['total_energy'],color='blue',linewidth=1.5,marker='.')
# plt.plot(df_daint['end'],df_daint['total_energy'],color='green',linewidth=1.5,marker='.')
# plt.plot(df_santis['end'],df_santis['total_energy'],color='orange',linewidth=1.5,marker='.')
# plt.yscale('log')
# plt.ylim(bottom=1)
# plt.xlabel("End", fontsize=20)  # Adjust x-axis label font size
# plt.ylabel("Total Energy", fontsize=20)  # Adjust y-axis label font size
# plt.xticks(fontsize=18)  # Adjust x-axis tick labels font size
# plt.yticks(fontsize=18)
# plt.legend(['clariden','alps-daint','alps-santis'],fontsize=16)
# plt.savefig('raw.png',bbox_inches='tight',dpi=200)
# plt.show()


##DATA FILTER
df_clariden1=df_clariden[df_clariden['global_accuracy_percentage']>90]
df_daint1=df_daint[df_daint['global_accuracy_percentage']>90]
df_santis1=df_santis[df_santis['global_accuracy_percentage']>90]

df_clariden1['power']=df_clariden1['total_energy']/(df_clariden1['elapsed']*df_clariden1['total_nodes'])
df_daint1['power']=df_daint1['total_energy']/(df_daint1['elapsed']*df_daint1['total_nodes'])
df_santis1['power']=df_santis1['total_energy']/(df_santis1['elapsed']*df_santis1['total_nodes'])
df_clariden2 = df_clariden1[df_clariden1['power']<=2700]
df_daint2 = df_daint1[df_daint1['power']<=2700]
df_santis2 = df_santis1[df_santis1['power']<=2700]

df_clariden2['total_energy'] = df_clariden2['total_energy'].where(df_clariden2['total_energy'].notna(), None)
df_daint2['total_energy'] = df_daint2['total_energy'].where(df_daint2['total_energy'].notna(), None)
df_santis2['total_energy'] = df_santis2['total_energy'].where(df_santis2['total_energy'].notna(), None)

unique_accounts_clariden = df_clariden2['account'].unique()
account_mapping_clariden = {account: f"project_{i}" for i, account in enumerate(unique_accounts_clariden)}
# Apply the mapping
df_clariden2['account_'] = df_clariden2['account'].map(account_mapping_clariden)

unique_accounts_daint = df_daint2['account'].unique()
account_mapping_daint = {account: f"project_{i}" for i, account in enumerate(unique_accounts_daint)}
# Apply the mapping
df_daint2['account_'] = df_daint2['account'].map(account_mapping_daint)

unique_accounts_santis = df_santis2['account'].unique()
account_mapping_santis = {account: f"project_{i}" for i, account in enumerate(unique_accounts_santis)}
# Apply the mapping
df_santis2['account_'] = df_santis2['account'].map(account_mapping_santis)


def pie_charts(df, cluster, quantity):
    if quantity == 'node_hours':
        df[quantity] = df['elapsed'] * df['total_nodes'] / 3600
        df_pie = df[['account_', 'node_hours']]
        df_pie_real = df[['account', 'node_hours']]

        quantity_graph='node-hours'
        unit='nh'
    else:
        df_pie = df[['account_', 'total_energy']]
        df_pie_real = df[['account', 'total_energy']]
        unit='GJ'
        quantity_graph='energy'
    
    pie = df_pie.groupby(['account_']).sum().reset_index()
    pie_real=df_pie_real.groupby(['account']).sum().reset_index()
    top_10_real=pie_real.nlargest(5, quantity).reset_index()
    top_10_real.to_csv(f"top10_real_{cluster}{quantity}.csv")
    top_10 = pie.nlargest(5, quantity).reset_index()
    others = pie.loc[:, ['account_', quantity]].drop(top_10.index)
    
    top_10.loc[len(top_10), quantity] = others[quantity].sum()
    top_10.loc[len(top_10) - 1, 'account_'] = 'others'
    colors = plt.cm.tab20.colors[:6]
    
    fig, ax = plt.subplots(figsize=(5, 5))
    wedges, texts, autotexts = ax.pie(
        top_10[quantity],
        autopct=lambda pct: f'{pct:.1f}%',
        # labels=top_10['account_'],
        colors=colors,
        # fontsize=16,
        pctdistance=0.6,
        labeldistance=1.2,
        textprops={'fontsize': 8}
    )
    
    # Create legend with absolute values
    if quantity=='node_hours':
        legend_labels = [f"{acc}: {val:.0f} {unit}" for acc, val in zip(top_10['account_'], top_10[quantity])]
    else:
        legend_labels = [f"{acc}: {val:.2f} {unit}" for acc, val in zip(top_10['account_'], round(top_10[quantity]/(10.0**9),2))]
    ax.legend(wedges, legend_labels, fontsize=10,title="Projects", bbox_to_anchor=(1, 0.25, 0.5, 0.5))
    
    plt.ylabel(ylabel='', fontsize=18)
    plt.title(f"Top 5 {quantity_graph} {cluster} projects", fontsize=16)
    plt.savefig(f"results/Top5_{quantity_graph}_{cluster}.png", bbox_inches='tight', dpi=200)
    plt.show()

pie_charts(df_clariden2,'Alps-Clariden','total_energy')
pie_charts(df_clariden2,'Alps-Clariden','node_hours')
pie_charts(df_daint2,'Alps-Daint','total_energy')
pie_charts(df_daint2,'Alps-Daint','node_hours')
pie_charts(df_santis2,'Alps-Santis','total_energy')
pie_charts(df_santis2,'Alps-Santis','node_hours')

df_all=pd.concat([df_clariden2,df_daint2,df_santis2])
df_all.to_csv('df_all.csv')
pie_charts(df_all,'big 3','total_energy')
pie_charts(df_all,'big 3','node_hours')

### energy histograms
def histogram(df,cluster,postfix):
    plt.hist(df['total_energy'], color='blue', edgecolor='black',bins=100)  # You can adjust the number of bins
    plt.title(f"{cluster}",fontsize=30)
    plt.xlabel('Energy [J]',fontsize=24)
    plt.ylabel('# jobs',fontsize=24)
    plt.yscale('log')
    plt.tick_params(axis='both', labelsize=24)
    plt.xlim(1,1e9)
    plt.ylim(1,10**5)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())  # Use standard notation instead of scientific
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))  # Force scientific notation
    ax.xaxis.get_offset_text().set_fontsize(24)  # Set font size for exponent (e.g., "1e9")
    plt.savefig(f"results/histogram_{cluster}{postfix}.png",bbox_inches='tight',dpi=200)
    plt.show()

histogram(df_clariden2,'Alps-Clariden','')

histogram(df_daint2,'Alps-Daint','')

histogram(df_santis2,'Alps-Santis','')



### PIE CHARTS FOR TOTAL WHOLE BIG 3 CLUSTERS

def format_with_apostrophes(n):
    return format(n, ',').replace(',', "'")

def heatmap_energy_size(df,cluster,threshold):
### heatmap job size vs energy with color number of jobs
    df['node_hours']=df['elapsed']*df['total_nodes']/3600.0
    sample_data=df[['total_nodes','total_energy','node_hours']]
    sample_data=df[df['total_energy']<threshold]
    energy_max=int(sample_data['total_energy'].max())
    delta=energy_max//(20)
    # sample_data['range_energy']=(sample_data['total_energy']//delta) * energy_max

    # Define bin edges
    bins = list(range(0, energy_max, delta))  # Creates bins [0-1000], [1000-2000], ..., [7000-8000]

    # Create labeled categories
    labels = [f"[{format_with_apostrophes(bins[i])}-{format_with_apostrophes(bins[i+1])}]" 
          for i in range(len(bins) - 1)]    # Bin the energy data
    binned_series = pd.cut(sample_data['total_energy'], bins=bins, labels=labels, include_lowest=True)

    # Display result
    # binned_series.to_csv('binned_energy_series')
    sample_data['range_energy']=binned_series

    # sample_data['job_count']=1
    # Create a heatmap data matrix
    sample_H1=sample_data[['total_nodes','range_energy','node_hours']].groupby(['total_nodes','range_energy']).sum().reset_index()
    sample_H1['node_hours']=sample_H1['node_hours'].replace(0,np.nan)
    # Plot the heatmap
    heatmap_data = sample_H1.pivot(index='total_nodes', columns='range_energy', values='node_hours')
    plt.figure(figsize=(20, 10))
    # print(heatmap_data)
    sns.heatmap(heatmap_data, cmap='coolwarm', linewidths=0.5,annot=True, fmt=".0f")

    plt.title(f"{cluster}: Job Size [# nodes] vs Energy [J] (Color = Node_hours)",fontsize=20)
    plt.ylabel("Job Size [#nodes]",fontsize=18)
    plt.xlabel("Energy [J]",fontsize=18)
    plt.xticks(rotation=90,fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(f"results/heatmap_sizeVSenergy_{cluster}.png",bbox_inches='tight',dpi=200)
    plt.show()
heatmap_energy_size(df_clariden2,'Alps-Clariden',1800000000)
heatmap_energy_size(df_daint2,'Alps-Daint',750000000)
heatmap_energy_size(df_santis2,'Alps-Santis',100000000)
### heatmap job size vs elapsed time in hours with color average energy per job

