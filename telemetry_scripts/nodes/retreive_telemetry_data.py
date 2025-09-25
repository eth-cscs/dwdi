import requests
import pandas as pd
import datetime as dt
from requests.auth import HTTPBasicAuth
import pytz
import json
import os


base_url =os.getenv('ES_URL')
index_name = '.ds-metrics-facility.telemetry-alps*'

user =os.getenv('ES_ADMIN_USER')
passw =os.getenv('ES_ADMIN_PASSW')

scroll_time='1m'


##endtimestamp of the job it has to be in UTC
timestampE_=dt.datetime(2025,4,15,22,0,0)
timestampE=(timestampE_).strftime("%Y-%m-%dT%H:%M")

###starttimestamp of the job it has to be in UTC
timestampS_=dt.datetime(2025,4,14,22,0,0)
timestampS=timestampS_.strftime("%Y-%m-%dT%H:%M") 


###EXAMPLE : GPU TEMPERATURE OF ALL 4 GPUS IN the GH NODE// for CPUS to set Sensor.PhysicalContext:CPU
## define scroll search which does the loop in ES and is faster than search after

def scroll_search(index_name, base_url,timestamp1,timestamp2,user,passw,scroll_time):
  query={
    "bool": { 
       "must":[
                {"match":{"Sensor.ParentalContext": "Chassis"}}, # nids of the job, add them there
               {"match":{"Sensor.PhysicalContext": "VoltageRegulator"}},
               {"match":{"Sensor.PhysicalSubContext":"Input"}},
              # {"match":{"Sensor.Index":0}},
               {"match":{"MessageId":"CrayTelemetry.Power"}},
               {"exists": {"field": "Sensor.LocationDetail.Node"}}
             ],
        "must_not":[
           {"terms":{"Sensor.LocationDetail.Cabinet":["x8000","x8001","x1500"]}}
        ],
        
      "filter": [
        {
          "range": {
            "@timestamp": {
              "format": "strict_date_optional_time",
              "gte": timestamp1,
              "lt": timestamp2
            }
          }
        }
      ]
      }
      }
  # print(query)
  # Set up the initial search request
  endpoint = f"{base_url}/{index_name}/_search?scroll="+scroll_time

    # Set up the search parameters
  search_params = {
        "size": 10000,  # Adjust this number based on your needs
        "query": query
    }

    # Make the initial search request
  response = requests.get(endpoint, json=search_params,auth=HTTPBasicAuth(user,passw))
  print(response.reason)
  search_results = response.json()
  #print(search_results)
    # Extract the scroll_id from the response
  all_hits=search_results["hits"]["hits"]
  # print(len(all_hits))
  if len(all_hits)==10000:
    scroll_id = search_results.get("_scroll_id")
    # Keep scrolling until there are no more results
    while True:
        # Set up the scroll request
        scroll_params = {
            "scroll": scroll_time,
            "scroll_id": scroll_id
        }

        # Make the scroll request
        scroll_response = requests.get(f"{base_url}/_search/scroll",json=scroll_params,auth=HTTPBasicAuth(user,passw))
        scroll_results = scroll_response.json()
        # print(scroll_results)

        all_hits.extend(scroll_results["hits"]["hits"])
        # Check if there are no more results
        if len(scroll_results["hits"]["hits"]) == 0:
          break
        
        # Update the scroll_id for the next scroll request
        scroll_id = scroll_results.get("_scroll_id")

  return all_hits

# Application of scroll_search

results=scroll_search(index_name,base_url,timestampS,timestampE,user,passw,scroll_time)
df_data=pd.json_normalize(results)
print(df_data.columns)
df_data['_source.Sensor.Timestamp']=pd.to_datetime(df_data['_source.Sensor.Timestamp'],format="%Y-%m-%dT%H:%M:%S.%fZ") # this timestamp is UTC 
df_data=df_data[['_source.Sensor.Timestamp','_source.Sensor.Index','_source.nid','_source.Sensor.LocationDetail.XName','_source.Sensor.Value']]
### EXPLANATION of the data: messageId is the quantity (temperature), Sensor.PhysicalContext the location measured and sensor.index is the device in the location, for Sensor.PhysicalContext GPU this represents which of the 4 gpus you are seeing, sensor.VAlue actual value of the described sensor.

df_data.to_csv('power_data_april.csv',index=True,header=True)

