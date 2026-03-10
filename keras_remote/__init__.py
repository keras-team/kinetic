import os

# Suppress noisy gRPC fork/logging messages before any gRPC imports
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

from absl import logging

from keras_remote.core.core import run as run
from keras_remote.data import Data as Data

logging.use_absl_handler()
