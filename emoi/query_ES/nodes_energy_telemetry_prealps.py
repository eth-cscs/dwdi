#!/usr/bin/env python


import requests
import json
import os

from requests.auth import HTTPBasicAuth
from datetime import datetime, timedelta


base_url = 'https://sole.cscs.ch:9200'
index_name = 'metrics-facility.telemetry-prealps*'  # SANTIS
search_url = f"{base_url}/{index_name}/_search"
user = os.environ["YYY"]
passw = os.environ["XXX"]


# {{{ read json query
def read_query(infile):
    with open(infile, 'r') as file:
        jsquery = json.load(file)

    return jsquery
# }}}


def nodelist2xname2joules_telemetry_prealps(t0, nidname):
    """
"MessageId": "CrayTelemetry.Energy" and "Sensor.ParentalContext": "Chassis"
and "Sensor.PhysicalContext": "VoltageRegulator"
and "Sensor.PhysicalSubContext": "Input" and "Sensor.Index": 0
and "Sensor.Location": "x1103c1s1b0n0"
and @timestamp >= "2024-02-14T09:55:00" and @timestamp <= "2024-02-14T09:56:00"
    """
    simple_query = read_query('nodes_energy_telemetry_prealps_query.json')
    simple_query['size'] = 1
    xname = nidname
    simple_query['query']['bool']['filter'][0]['bool']['filter'][5]['bool'][
        'should'][0]['term']['Sensor.Location']['value'] = xname
    simple_query['query']['bool']['filter'][1]['range']['@timestamp'][
        'gte'] = t0

    response = requests.get(search_url, json=simple_query,
                            auth=HTTPBasicAuth(user, passw))

    response_js = response.json()
    return response_js['hits']['hits'][0]['fields']['Sensor.Value'][0]


# {{{ main
if __name__ == "__main__":
    t0 = '2024-02-07T18:40:29'
    t1 = '2024-02-07T18:40:49'
    t0minus1h = (datetime.fromisoformat(t0) + timedelta(hours=-1)).isoformat()
    t1minus1h = (datetime.fromisoformat(t1) + timedelta(hours=-1)).isoformat()
    #
    nidname = 'x1101c0s0b1n0'  # ['nid001258']
    J0 = nodelist2xname2joules_telemetry_prealps(t0minus1h, nidname)
    J1 = nodelist2xname2joules_telemetry_prealps(t1minus1h, nidname)
    print(J0, J1, int(J1)-int(J0))
# }}}
