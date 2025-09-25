import requests
import pandas as pd
import datetime as dt
from requests.auth import HTTPBasicAuth
import pytz
import json
import os

#ti=dt.datetime.now()

base_url = os.getenv('ES_URL')
index_name = '.ds-metrics-facility.telemetry-alps.cdu*'

user =os.getenv('ES_ADMIN_USER')
passw = os.getenv('ES_ADMIN_PASSW')

scroll_time='1m'

### ELASTISEARCH TIME UTC= TO SUBTRACT 1H TO REGULAR TIME FOR THE TWO TIMESTAMPS###
delta=dt.timedelta(hours=1)
print(delta)
#print(delta)
# timestampE_=dt.datetime.now() - delta
timestampE_=dt.datetime(2024,10,31,23,0,0)
#timestampE_=dt.datetime(2024,7,17,16,15,0)
print(timestampE_)
timestampE=(timestampE_).strftime("%Y-%m-%dT%H:%M")
timestampS_=dt.datetime(2024,10,30,23,0,0)
timestampS=timestampS_.strftime("%Y-%m-%dT%H:%M")
print(timestampS)



## define scroll search which does the loop in ES and is faster than search after

def scroll_search(index_name, base_url,timestamp1,timestamp2,user,passw,scroll_time):
  query={
    "bool": { 
       "must":[{"match":{"Cabinet": "x1103"}}
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
        print(scroll_results)

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
df_data['_source.Timestamp']=pd.to_datetime(df_data['_source.Timestamp'],format="%Y-%m-%dT%H:%M:%S.%fZ") + dt.timedelta(hours=1)
df_data=df_data[['_source.Timestamp','_source.Cabinet','_source.Secondary_Cabinet_Return_Water_Temperature_2','_source.Secondary_Cabinet_Supply_Water_Temperature_2','_source.Primary_Facility_Supply_Water_Temperature','_source.Primary_Facility_Return_Water_Temperature']]
df_data.to_csv('data_cdux1103_ott_nov24.csv',index=True,header=True)

