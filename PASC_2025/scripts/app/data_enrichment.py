import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt


def pie_charts_domain(df,cluster,quantity):
    if cluster!='Alps-Clariden':
        if quantity == 'node_hours':
            df[quantity] = df['elapsed'] * df['total_nodes'] / 3600
            df_pie = df[['domain_subdomain', 'node_hours']]
            quantity_graph='node-hours'
            unit='nh'
        else:
            df_pie = df[['domain_subdomain', 'total_energy']]
            unit='GJ'
            quantity_graph='energy'
  
        pie = df_pie.groupby(['domain_subdomain']).sum().reset_index()
        top_10 = pie.nlargest(5, quantity).reset_index()
        others = pie.loc[:, ['domain_subdomain', quantity]].drop(top_10.index)
        
        top_10.loc[len(top_10), quantity] = others[quantity].sum()
        top_10.loc[len(top_10) - 1, 'domain_subdomain'] = 'others'
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
            legend_labels = [f"{acc}: {val:.0f} {unit}" for acc, val in zip(top_10['domain_subdomain'], top_10[quantity])]
        else:
            legend_labels = [f"{acc}: {val:.2f} {unit}" for acc, val in zip(top_10['domain_subdomain'], round(top_10[quantity]/(10.0**9),2))]
        ax.legend(wedges, legend_labels, fontsize=10,title="Application field", bbox_to_anchor=(1, 0.25, 0.5, 0.5))
        
        plt.ylabel(ylabel='', fontsize=18)
        plt.title(f"Top 5 {quantity_graph} {cluster} application field", fontsize=16)
        plt.savefig(f"results/Top5_{quantity_graph}_{cluster}_domain.png", bbox_inches='tight', dpi=200)
        plt.show()
    else:
        if quantity == 'node_hours':
            df[quantity] = df['elapsed'] * df['total_nodes'] / 3600
            df_pie = df[['description', 'node_hours']]
            quantity_graph='node-hours'
            unit='nh'
        else:
            df_pie = df[['description', 'total_energy']]
            unit='GJ'
            quantity_graph='energy'
        pie = df_pie.groupby(['description']).sum().reset_index()
        top_10 = pie.nlargest(5, quantity).reset_index()
        others = pie.loc[:, ['description', quantity]].drop(top_10.index)
        
        top_10.loc[len(top_10), quantity] = others[quantity].sum()
        top_10.loc[len(top_10) - 1, 'description'] = 'others'
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
            legend_labels = [f"{acc}: {val:.0f} {unit}" for acc, val in zip(top_10['description'], top_10[quantity])]
        else:
            legend_labels = [f"{acc}: {val:.2f} {unit}" for acc, val in zip(top_10['description'], round(top_10[quantity]/(10.0**9),2))]
        ax.legend(wedges, legend_labels, fontsize=10,title="Application field", bbox_to_anchor=(1, 0.25, 0.5, 0.5))
        
        plt.ylabel(ylabel='', fontsize=18)
        plt.title(f"Top 5 {quantity_graph} {cluster} application fields", fontsize=16)
        plt.savefig(f"results/Top5_{quantity_graph}_{cluster}_description.png", bbox_inches='tight', dpi=200)
        plt.show()


dfc=pd.read_csv('dfc.csv').drop(columns='Unnamed: 0').drop_duplicates()
dfd=pd.read_csv('dfd.csv').drop(columns='Unnamed: 0').drop_duplicates()
dfs=pd.read_csv('dfs.csv').drop(columns='Unnamed: 0').drop_duplicates()

dfc['account']=dfc['project_group_id']
dfd['account']=dfd['project_group_id']
dfs['account']=dfs['project_group_id']


df_clariden=pd.read_csv('data/raw_data_clariden_jan_feb_mar.csv')
df_daint=pd.read_csv('data/raw_data_alps_daint_jan_feb_mar.csv')
df_santis=pd.read_csv('data/raw_data_alps_santis_jan_feb_mar.csv')

df_clariden['node_hours']=df_clariden['elapsed']*df_clariden['total_nodes']/3600.0
df_daint['node_hours']=df_daint['elapsed']*df_daint['total_nodes']/3600.0
df_santis['node_hours']=df_santis['elapsed']*df_santis['total_nodes']/3600.0

df_clariden['end']=pd.to_datetime(df_clariden['end'],format="%Y-%m-%d %H:%M:%S")
df_daint['end']=pd.to_datetime(df_daint['end'],format="%Y-%m-%d %H:%M:%S")
df_santis['end']=pd.to_datetime(df_santis['end'],format="%Y-%m-%d %H:%M:%S")
df_clariden=df_clariden[(df_clariden['end']>=datetime(2025,2,1))&(df_clariden['end']<datetime(2025,3,1))]
df_daint=df_daint[(df_daint['end']>=datetime(2025,2,1))&(df_daint['end']<datetime(2025,3,1))]
df_santis=df_santis[(df_santis['end']>=datetime(2025,2,1))&(df_santis['end']<datetime(2025,3,1))]

##accounting db node-hours check problem: we filter out many jobs and statistics about node-hours are faked.
print(df_clariden['node_hours'].sum())
print(df_daint['node_hours'].sum())
print(df_santis['node_hours'].sum())


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

df_clariden2['project_group_id']=df_clariden2['account'].str.replace(r'^(a-|w_)','', regex=True)

df_clariden3=pd.merge(left=df_clariden2,right=dfc,on='project_group_id')
df_daint3=pd.merge(left=df_daint2,right=dfd,on='account')
df_santis3=pd.merge(left=df_santis2,right=dfs,on='account')

df_clariden3['domain_subdomain']=df_clariden3['domain']+'/'+df_clariden3['subdomain']
df_santis3['domain_subdomain']=df_santis3['domain']+'/'+df_santis3['subdomain']
df_daint3['domain_subdomain']=df_daint3['domain']+'/'+df_daint3['subdomain']

pie_charts_domain(df_clariden3,'Alps-Clariden','node_hours')
pie_charts_domain(df_daint3,'Alps-Daint','node_hours')
pie_charts_domain(df_santis3,'Alps-Santis','node_hours')
pie_charts_domain(df_clariden3,'Alps-Clariden','total_energy')
pie_charts_domain(df_daint3,'Alps-Daint','total_energy')
pie_charts_domain(df_santis3,'Alps-Santis','total_energy')