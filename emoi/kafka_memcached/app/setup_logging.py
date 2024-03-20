# Copyright (c) 2024 CSCS, ETH Zurich
# SPDX-License-Identifier: BSD 3-Clause License
import logging
import logging.config
import os

log_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '%(asctime)s %(name)s:%(levelname)s %(filename)s:'
                      '%(lineno)s:%(funcName)s(): %(message)s'
        },
    },
    'handlers': {
        'stdout': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'stream': "ext://sys.stdout",
        },
    },
    'loggers': {
        '__main__': {
            'handlers': ['stdout'],
            'propagate': False,
        },
        'kafka_memcached': {
            'handlers': ['stdout'],
            'propagate': False,
        }
    }

}

def setup_logging_from_env():
    """ Sets up the __main__ logger with LOG_LEVEL env variable"""
    for lc in log_config["loggers"].values():
        lc["level"] = os.environ.get('LOG_LEVEL', 'INFO')
    logging.config.dictConfig(log_config)

