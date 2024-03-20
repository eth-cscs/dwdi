# kafka_memcached

Python application to get mappings of key-value pairs from a kafka
topic and write it to memcached.

## Getting started

Build the image
```
docker build kafka-memcached .
```

Create an .env file containing all the environment variables as desribed in 
.sample_env and run:

```
docker run --env-file ./.env kafka-memcached

```