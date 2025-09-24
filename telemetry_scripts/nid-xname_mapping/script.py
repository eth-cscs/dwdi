import requests
import pandas as pd
import datetime as dt
from requests.auth import HTTPBasicAuth
import pytz
import json
import os

ti=dt.datetime.now()

base_url = base_url = os.getenv('ES_URL')
index_name = '.ds-logs-accounting.platform-alps.v02*'

user =os.getenv('ES_ADMIN_USER')
passw = os.getenv('ES_ADMIN_PASSW')

scroll_time='1m'

### ELASTISEARCH TIME UTC= TO SUBTRACT 1H TO REGULAR TIME FOR THE TWO TIMESTAMPS###
delta=dt.timedelta(hours=2)
# timestampE_=dt.datetime.now() - delta
timestampE_=dt.datetime(2024,10,18,0,0,0)-delta ##INPUT THE DAY OF YOUR JOB
timestampE=(timestampE_).strftime("%Y-%m-%dT%H:%M:%S")
timestampS_=(timestampE_-dt.timedelta(hours=24))
timestampS=timestampS_.strftime("%Y-%m-%dT%H:%M:%S")
# timestamp10=(timestampE_-dt.timedelta(seconds=50)).strftime("%Y-%m-%dT%H:%M:%S")
# timestamp20=(timestampE_-dt.timedelta(seconds=40)).strftime("%Y-%m-%dT%H:%M:%S")
# timestamp30=(timestampE_-dt.timedelta(seconds=30)).strftime("%Y-%m-%dT%H:%M:%S")
# timestamp40=(timestampE_-dt.timedelta(seconds=20)).strftime("%Y-%m-%dT%H:%M:%S")
# timestamp50=(timestampE_-dt.timedelta(seconds=10)).strftime("%Y-%m-%dT%H:%M:%S")

# timestamp=list()
# timestamp.append(timestampS)
# timestamp.append(timestamp10)
# timestamp.append(timestamp20)
# timestamp.append(timestamp30)
# timestamp.append(timestamp40)
# timestamp.append(timestamp50)
# timestamp.append(timestampE)
# print(timestamp)

## define scroll search which does the loop in ES and is faster than search after

def scroll_search(index_name, base_url,timestamp1,timestamp2,user,passw,scroll_time):
  query={
    "bool": { 
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
        #print(scroll_results)

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
xname=df_data["_source.Entry.NodeId"].unique()
df_data_def=df_data[['_source.Entry.NodeNid','_source.Entry.NodeId']]
df_data_def=df_data_def.drop_duplicates()
df_data_def.to_csv('table_nid_xanme_extended.csv',header=True,index=True)
print(df_data_def)