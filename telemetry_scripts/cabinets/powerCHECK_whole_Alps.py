import sys, os, re
import requests, json
import pandas as pd
import datetime as dt
import pylab as pl
import matplotlib.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from requests.auth import HTTPBasicAuth

base_url =os.getenv('ES_URL')
index_name = '.ds-metrics-facility.telemetry-alps*'

user =os.getenv('ES_ADMIN_USER')
passw =os.getenv('ES_ADMIN_PASSW')

scroll_time='5m'
##data_stream.type: "logs"
  ##      data_stream.dataset: "prealps"
    ##    data_stream.namespace: "cabinet.power" 
##

### ELASTISEARCH TIME UTC= TO SUBTRACT 1H TO REGULAR TIME FOR THE TWO TIMESTAMPS###
timestamp2_=dt.datetime(2025,5,15,10,30,0)
timestamp2=timestamp2_.strftime("%Y-%m-%dT%H:%M:%S")
timestamp1=dt.datetime(2025,4,30,22,0,0).strftime("%Y-%m-%dT%H:%M:%S")
#cabinet="x1100" # x1100-1105 , x1200-x1205

simple_query={
    "bool": { "must":[{"match":{"MessageId": "CrayTelemetry.Power"}},
             # MODIFICA QUI SOTTO SECONDO I CABINET DA ANALIZZARE
              # {"terms":{"Sensor.LocationDetail.Cabinet": ["x1100","x1101","x1102","x1103","x1105","x1200","x1201","x1202","x1203","x1204","x1205"]}},
              {"match":{"Sensor.PhysicalContext": "Rectifier"}},
             {"match":{"Sensor.PhysicalSubContext":"Input"}}
            
             ],
            "must_not":[{
               "terms":{"Sensor.LocationDetail.Cabinet":["x8001","x8000","x1500"]}
            }]
            ,
      "filter": [
        {
          "range": {
            "@timestamp": {
              "format": "strict_date_optional_time",
              "gte": timestamp1,
              "lte": timestamp2
            }
          }
        }
      ]
              }}

endpoint = f"{base_url}/{index_name}/_search?scroll="+scroll_time
search_params = {
        "size": 10000,  # Adjust this number based on your needs
        "query": simple_query
    }

response = requests.get(endpoint, json=search_params,auth=HTTPBasicAuth(user,passw))
search_results = response.json()
print(response.status_code)
print(response.reason)
all_hits=search_results["hits"]["hits"]
  # print(len(all_hits))
if len(all_hits)==10000:
  scroll_id = search_results.get("_scroll_id")

while True:
  # Set up the scroll request
  scroll_params = {
      "scroll": scroll_time,
      "scroll_id": scroll_id
  }

  # Make the scroll request
  scroll_response = requests.get(f"{base_url}/_search/scroll",json=scroll_params,auth=HTTPBasicAuth(user,passw))
  scroll_results = scroll_response.json()
  #print(scroll_results)

  all_hits.extend(scroll_results["hits"]["hits"])
  # Check if there are no more results
  if len(scroll_results["hits"]["hits"]) == 0:
    break
  
  # Update the scroll_id for the next scroll request
  scroll_id = scroll_results.get("_scroll_id")



# print(response.status_code)
# if response.status_code == 200:
#     results = response.json()
# else:
#     print(f"Error: {response.status_code} - {response.text}")

df_data=pd.json_normalize(all_hits)


df_dataC=df_data
df_dataC=df_dataC[["_source.@timestamp","_source.Sensor.Index","_source.Sensor.LocationDetail.Cabinet","_source.Sensor.LocationDetail.Chassie","_source.Sensor.Value"]]
df_dataC['_source.@timestamp']=pd.to_datetime(df_dataC['_source.@timestamp'],format="%Y-%m-%dT%H:%M:%S.%fZ")+dt.timedelta(hours=2)
df_dataC=df_dataC.set_index('_source.@timestamp')

def check(x):
    if len(x.dropna())==0:
        return 0
    else:
        return x
    

###PER SALVARE I DATI RAW TOGLIERE IL COMMENTO ALLA LINEA SOTTO
#df_dataC.to_csv('data12_4_2024.csv',header=True,index=True)

j=0
cabinet_list=df_dataC["_source.Sensor.LocationDetail.Cabinet"].unique()
for cab in cabinet_list:
  i=0
  chassis_list=df_dataC[df_dataC["_source.Sensor.LocationDetail.Cabinet"]==cab]['_source.Sensor.LocationDetail.Chassie'].unique()
  for ch in chassis_list:
    rectifier_list=df_dataC.loc[(df_dataC["_source.Sensor.LocationDetail.Cabinet"]==cab)&(df_dataC["_source.Sensor.LocationDetail.Chassie"]==ch),"_source.Sensor.Index"].unique()
    if len(rectifier_list)==4:
      df0c0=df_dataC[(df_dataC["_source.Sensor.Index"]==0) & (df_dataC["_source.Sensor.LocationDetail.Chassie"]==ch)& (df_dataC["_source.Sensor.LocationDetail.Cabinet"]==cab)]
      df1c0=df_dataC[(df_dataC["_source.Sensor.Index"]==1) & (df_dataC["_source.Sensor.LocationDetail.Chassie"]==ch)&(df_dataC["_source.Sensor.LocationDetail.Cabinet"]==cab)]
      df2c0=df_dataC[(df_dataC["_source.Sensor.Index"]==2) & (df_dataC["_source.Sensor.LocationDetail.Chassie"]==ch)&(df_dataC["_source.Sensor.LocationDetail.Cabinet"]==cab)]
      df3c0=df_dataC[(df_dataC["_source.Sensor.Index"]==3) & (df_dataC["_source.Sensor.LocationDetail.Chassie"]==ch)&(df_dataC["_source.Sensor.LocationDetail.Cabinet"]==cab)]
        
      df0c0S=pd.DataFrame(df0c0["_source.Sensor.Value"].resample('1S').max()).fillna(method='ffill')
      df1c0S=pd.DataFrame(df1c0["_source.Sensor.Value"].resample('1S').max()).fillna(method='ffill')
      df2c0S=pd.DataFrame(df2c0["_source.Sensor.Value"].resample('1S').max()).fillna(method='ffill')
      df3c0S=pd.DataFrame(df3c0["_source.Sensor.Value"].resample('1S').max()).fillna(method='ffill')
      if i>0: 
          if (len(dfch.dropna()>0)):
            dfch=dfch+check(df0c0S)+check(df1c0S)+check(df2c0S)+check(df3c0S)
          else:
              dfch=dfch       
      else: 
          dfch=check(df0c0S)+check(df1c0S)+check(df2c0S)+check(df3c0S)
    else:
      df0c0=df_dataC[(df_dataC["_source.Sensor.Index"]==0) & (df_dataC["_source.Sensor.LocationDetail.Chassie"]==ch)& (df_dataC["_source.Sensor.LocationDetail.Cabinet"]==cab)]
      df1c0=df_dataC[(df_dataC["_source.Sensor.Index"]==1) & (df_dataC["_source.Sensor.LocationDetail.Chassie"]==ch)&(df_dataC["_source.Sensor.LocationDetail.Cabinet"]==cab)]
      df2c0=df_dataC[(df_dataC["_source.Sensor.Index"]==2) & (df_dataC["_source.Sensor.LocationDetail.Chassie"]==ch)&(df_dataC["_source.Sensor.LocationDetail.Cabinet"]==cab)]
        
      df0c0S=pd.DataFrame(df0c0["_source.Sensor.Value"].resample('1S').max()).fillna(method='ffill')
      df1c0S=pd.DataFrame(df1c0["_source.Sensor.Value"].resample('1S').max()).fillna(method='ffill')
      df2c0S=pd.DataFrame(df2c0["_source.Sensor.Value"].resample('1S').max()).fillna(method='ffill')
      if i>0: 
          if (len(dfch.dropna()>0)):
            dfch=dfch+check(df0c0S)+check(df1c0S)+check(df2c0S)
          else:
              dfch=dfch       
      else: 
          dfch=check(df0c0S)+check(df1c0S)+check(df2c0S)

    i=i+1
      
  cabinet=dfch.copy(deep=True)
  print(cabinet.columns)
  cabinet.columns=[f"cabinet: {cab}"]
  # cabinet.columns=['Value']
  # cabinet.index.name='time'
  # cabinet.plot(figsize=(30,15),fontsize=30)
  # plt.title('Cabinet power '+str(cab)+' with maximum power: ' + str(cabinet.max()))
  # plt.xlabel('Datetime',fontsize=30)
  # plt.savefig('cabinet'+str(cabinet_list[j])+'.png',bbox_inches='tight',dpi=300)
  if j==0:
    cabinets=cabinet.copy(deep=True)
  else:
    cabinets=pd.concat([cabinets,cabinet],axis=1)
    # cabinets=cabinets_
  # cabinets.to_csv('wholeAlps_15May2025.csv')
  j=j+1
cabinets=cabinets.fillna(0)
n=0
for col in cabinets.columns:
  if n==0:
     cabinets['alps']=cabinets[col]
  else: 
    cabinets['alps']=cabinets['alps']+cabinets[col]
  n=n+1


cabinets.to_csv('wholeAlps_15May2025.csv')

print(cabinets.max())

      

