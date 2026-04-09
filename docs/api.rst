Python API Reference
====================

.. currentmodule:: kinetic

Decorator
---------

.. autofunction:: run

Data API
--------

.. autoclass:: Data
   :members:
   :show-inheritance:

Async Jobs
----------

.. autofunction:: submit

.. autoclass:: JobHandle
   :members:
   :show-inheritance:

.. autofunction:: attach

.. autofunction:: list_jobs

Async Collections
-----------------

.. autofunction:: map

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
