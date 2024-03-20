# Copyright (c) 2024 CSCS, ETH Zurich
# SPDX-License-Identifier: BSD 3-Clause License
import datetime
import logging
from kafka import KafkaConsumer
from pymemcache.client import base

logger = logging.getLogger(__name__)


class KafkaMemcachedReaderWriter():
    """ 
    Simple context manager for consuming from a kafka topic and writing to memcached
    """
    def __init__(
            self, field_map: list[dict], rename_map: dict, kafka_topic: str, 
            kafka_config: dict, memcached_server: tuple[str, int],
            # KafkaConsumer.poll limits:
            time_limit_poll: int = 5, # Time limit for iterating over the msgs in sec.
            max_records_poll: int = 10 # the maximum records to pull
            ) -> None:
        
        self.field_map = field_map
        self.rename_map = rename_map
        self.kafka_topic = kafka_topic
        self.kafka_config = kafka_config
        self.memcached_server = memcached_server
        self.time_limit_poll=time_limit_poll
        self.max_records_poll = max_records_poll
        super().__init__()

    def __enter__(self):
        self.consumer = KafkaConsumer(self.kafka_topic, **self.kafka_config)
        self.memcached_client = base.Client(self.memcached_server)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info("closing clients")
        self.memcached_client.close()
        self.consumer.close()
        if exc_type is not None:
            logger.warning("%s: %s \n %s", exc_type, exc_val, exc_tb)

    @staticmethod
    def datetime_from_timestamp_ms(ts) -> datetime.datetime:
        """ convert timestamp in ms to datetime """
        return datetime.datetime.fromtimestamp(ts/1000,
                                            tz=datetime.timezone.utc)
    
    def get_key_val_from_message(self, message, key_name, val_name):
        """ Extracts key and value to be written to memcached from the kafka message"""
        _key_name = self.rename_map.get(key_name, key_name)
        _val_name = self.rename_map.get(val_name, val_name)
        namespace= f"{_key_name}-{_val_name}"
        key = f"{namespace}:{message[key_name]}"
        return key, message[val_name]

    def write_message_to_memcached(self, message):
        """ Writing the mappings from FIELD MAP to memcached """
        logger.info("Writing content from Kafka message with timestamp=%s to memcached",
                    self.datetime_from_timestamp_ms(message.timestamp))
        logger.debug(f"{message.partition=}, {message.offset=}")
        for fm in self.field_map:
            key, value = self.get_key_val_from_message(message.value, fm["key"], fm["value"])
            self.memcached_client.set(key, value)
            logger.debug("New key=%s, written to memcached with value=%s",
                            key, self.memcached_client.get(key).decode('UTF-8'))
            
    def read_write(self) -> None:
        """ 
        Gets fields form Kafka topic and writes the mapping to memcached without time limit 
        """
        logger.info("Start consuming")
        for message in self.consumer:
            logger.debug(self.consumer.assignment())
            self.write_message_to_memcached(message)


    def poll_write(self) -> None:
        """ Gets fields form Kafka topic and writes the mapping to memcached using 
            KafkaConsumer.poll
        """
        messages_dict = self.consumer.poll(
            timeout_ms=self.time_limit_poll*1000, max_records=self.max_records_poll)
        for topic_partition, messages in messages_dict.items():
            logger.debug(topic_partition)
            for message in messages:
                self.write_message_to_memcached(message)

