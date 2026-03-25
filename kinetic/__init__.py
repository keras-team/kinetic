import os

# Suppress noisy gRPC fork/logging messages before any gRPC imports
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "0")

import logging as python_logging
import os

from absl import logging
from rich.console import Console
from rich.logging import RichHandler

from kinetic.core.core import run as run
from kinetic.core.core import submit as submit
from kinetic.data import Data as Data
from kinetic.jobs import JobHandle as JobHandle
from kinetic.jobs import attach as attach
from kinetic.jobs import list_jobs as list_jobs

logging.use_absl_handler()

# Use rich to format the absl logs, making them slightly dimmed and links clickable
console = Console(stderr=True)
rich_handler = RichHandler(
  console=console,
  show_time=False,
  show_path=False,
  show_level=False,
  markup=True,
)
rich_handler.setFormatter(python_logging.Formatter("[dim]%(message)s[/dim]"))

absl_logger = logging.get_absl_logger()
absl_logger.handlers = [rich_handler]
absl_logger.propagate = False

# Default to INFO if the user is running a script outside of absl.app.run()
# This ensures that operations like container building and job status are visible.
log_level = os.environ.get("KINETIC_LOG_LEVEL", "INFO").upper()

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
