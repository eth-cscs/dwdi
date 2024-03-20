# Copyright (c) 2024 CSCS, ETH Zurich
# SPDX-License-Identifier: BSD 3-Clause License
"""
Script to be run on Sole Metal to get mappings of key-value paris from a kafka
topic and write it to memcached.
"""
import json
import os
import logging

from setup_logging import setup_logging_from_env
from kafka_memcached.kafka_memcached_reader_writer import KafkaMemcachedReaderWriter

logger = logging.getLogger(__name__)


def main():
    """
    Gets all the configurations from the environemnt variables and writes mapping
    to memcached.
    """

    # The mapping with corresponding renaming to be written to memcached
    FIELD_MAP: list[dict] = json.loads(os.environ["FIELD_MAP"])
    RENAME_MAP: dict = json.loads(os.environ["RENAME_MAP"])

    # KakfaConsumer constants
    KAFKA_TOPIC = os.environ["TOPIC"] # The kafka topic
    KAFKA_CONFIG = {
        "bootstrap_servers": [os.environ["BOOTSTRAP_SERVER"]],
        "group_id": os.environ["GROUP_ID"],
        "auto_offset_reset": os.environ.get("AUTO_OFFSET_RESET", "latest"),
        "value_deserializer": lambda v: json.loads(v.decode('UTF-8'))}
    
    MEMCACHED_SERVER = (os.environ["MEMCACHED_HOST"], int(os.environ["MEMCACHED_PORT"]))

    with KafkaMemcachedReaderWriter(field_map=FIELD_MAP, rename_map=RENAME_MAP, 
                        kafka_topic=KAFKA_TOPIC, kafka_config=KAFKA_CONFIG, 
                        memcached_server=MEMCACHED_SERVER) as km:
        km.read_write()

if __name__ == "__main__":
    setup_logging_from_env()
    main()
