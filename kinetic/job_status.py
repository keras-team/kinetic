"""Job status enum shared across kinetic backends."""

from enum import Enum


class JobStatus(Enum):
  """Observable status of a remote job."""

  PENDING = "PENDING"
  RUNNING = "RUNNING"
  SUCCEEDED = "SUCCEEDED"
  FAILED = "FAILED"
  NOT_FOUND = "NOT_FOUND"
