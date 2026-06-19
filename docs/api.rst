Python API Reference
====================

.. currentmodule:: kinetic

Decorators
----------

.. autofunction:: run

.. autoclass:: RemoteCallable
   :members: run_async, run_async_map
   :special-members: __call__

Data API
--------

.. autoclass:: Data
   :members:
   :show-inheritance:

Detached Jobs
-------------

.. autoclass:: JobHandle
   :members:
   :show-inheritance:

.. autofunction:: attach

.. autofunction:: list_jobs

Batched Jobs
------------

.. autofunction:: kinetic.collections.map

.. autoclass:: BatchHandle
   :members: statuses, status_counts, wait, as_completed, results, failures, cancel, cleanup
   :show-inheritance:

.. autoclass:: BatchError
   :members:
   :show-inheritance:

.. autofunction:: attach_batch

Job Status
----------

.. autoclass:: kinetic.job_status.JobStatus
   :members:
   :undoc-members:
