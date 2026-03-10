import os

# Suppress noisy gRPC fork/logging messages before any gRPC imports
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

import logging as python_logging

from absl import logging

from keras_remote.core.core import run as run
from keras_remote.data import Data as Data

logging.use_absl_handler()

# Remove absl verbose prefixes (e.g. I0310... execution.py:297])
logging.get_absl_handler().setFormatter(python_logging.Formatter("%(message)s"))

# Default to INFO if the user is running a script outside of absl.app.run()
# This ensures that operations like container building and job status are visible.
log_level = os.environ.get("KERAS_REMOTE_LOG_LEVEL", "INFO").upper()

if log_level == "DEBUG":
  logging.set_verbosity(logging.DEBUG)
elif log_level == "INFO":
  logging.set_verbosity(logging.INFO)
elif log_level == "WARNING":
  logging.set_verbosity(logging.WARNING)
elif log_level == "ERROR":
  logging.set_verbosity(logging.ERROR)
elif log_level == "FATAL":
  logging.set_verbosity(logging.FATAL)
