import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_json('santis.json')

cluster="Alps-Santis"

def check_float(x):
    print(x)
    try: 
        x=float(x)
        if (float(x)>1.8*10**19):
            return None
        else:
            return x
    except:
        pass
df['abs_delta']=abs(df['total_energy'].astype('float')-df['energy_raw'].apply(check_float))

df=df.dropna().reset_index()

per=100*df.loc[df['abs_delta']>10**6,'abs_delta'].count()/df.loc[:,'abs_delta'].count()
# df['abs_delta'].plot()
# plt.yscale('log')
# plt.title(f"Absolute error{cluster}; Data >10^6:{round(per,2)}%")
# plt.savefig(f"{cluster}_error_range_png",bbox_inches='tight',dpi=200)
# plt.show()
print(per)


df['relative_tel_error']=df['abs_delta']/df['total_energy']

df['relative_tel_error'].plot(figsize=(20,10),fontsize=20)
plt.title(f"Relative telemetry error {cluster}")
plt.ylim(-2,100)
plt.ylabel('Energy [J]',fontsize=20)
plt.xlabel('Jobs Count',fontsize=20)
plt.savefig(f"{cluster}_rel_tel_error.png",bbox_inches='tight',dpi=200)
plt.show()


df.to_csv(f"df_{cluster}.csv")

plt.figure(figsize=(10, 10))
plt.plot(df['total_energy'],df['energy_raw'],marker='.',linestyle='None')
plt.xlabel('Telemetry[J]')
plt.ylabel('SLURM [J]')
plt.ylim(0,1e8)
plt.xlim(0,1e8)
plt.title(f"Telemetry vs SLURM Energy {cluster}" )
plt.show()

## santis: 0.5 % percentage to be in the lower or in the upper bucket (wrong bucket)