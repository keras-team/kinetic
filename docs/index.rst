Kinetic: Run ML workloads on cloud TPUs and GPUs
================================================

.. toctree::
   :caption: Start Here
   :hidden:

   getting_started
   guides/execution_modes
   troubleshooting
   guides/faq

.. toctree::
   :caption: Core Workflows
   :hidden:

   guides/data
   guides/checkpointing
   guides/dependencies
   guides/env_vars
   guides/debugging
   guides/cost_optimization
   guides/distributed_training
   guides/containers
   guides/advanced

.. toctree::
   :caption: Examples & Tutorials
   :hidden:

   examples/keras_training
   examples/jax_training
   examples/pytorch_training
   examples/gemma4_finetuning
   examples/llm_finetuning
   examples

.. toctree::
   :caption: Reference
   :hidden:

   api
   cli
   accelerators
   configuration

.. toctree::
   :caption: Contributing
   :hidden:

   architecture
   contributing
   code-of-conduct

Run any Python function on a cloud TPU or GPU with one decorator. No
infrastructure to wire up, no images to build by hand, no multi-host
boilerplate.

.. code-block:: python

    import kinetic

    @kinetic.run(accelerator="tpu-v6e-8")
    def train_model():
        import keras
        model = keras.Sequential([...])
        model.fit(x_train, y_train)
        return model.history.history["loss"][-1]

    final_loss = train_model()  # runs on a TPU v6e-8 slice

Start here
----------

Three entry points cover what most new users need first:

.. list-table::
   :widths: 33 33 34
   :header-rows: 1

   * - Your first run
     - Long-running jobs
     - Data and checkpoints
   * - Install, point at a cluster, and run a real Keras job in minutes.
       :doc:`Getting Started <getting_started>`.
     - Switch from blocking ``run()`` to detached ``submit()`` for jobs
       that take hours. :doc:`Detached Jobs <guides/advanced>`.
     - Ship local files in, write durable artifacts back out via
       ``KINETIC_OUTPUT_DIR``. :doc:`Data <guides/data>` and
       :doc:`Checkpointing <guides/checkpointing>`.

How Kinetic works
-----------------

Five short phases on every job:

1. **Discover.** Your function, working directory, and ``Data(...)``
   arguments are captured. ``requirements.txt`` or ``pyproject.toml``
   is read.
2. **Build or fetch.** A container image is produced — built with your
   dependencies (bundled mode) or pulled from a published base
   (prebuilt mode). See :doc:`Execution Modes <guides/execution_modes>`.
3. **Schedule.** A Kubernetes resource (a ``Job`` for single-host
   workloads, a ``LeaderWorkerSet`` for multi-host TPU jobs on the
   Pathways backend) is submitted to your GKE cluster. The autoscaler
   provisions accelerator nodes if needed.
4. **Run.** Your function executes inside the pod with
   ``KINETIC_OUTPUT_DIR`` set; logs stream back to your terminal.
5. **Collect.** The return value is serialized to GCS and pulled back
   to your local process. ``@kinetic.run()`` cleans up the pod and GCS
   artifacts as soon as the result is collected. ``@kinetic.submit()``
   leaves the pod running until you call ``.result()`` or ``.cleanup()``
   on the returned ``JobHandle`` — important to remember on expensive
   accelerators.

Choose your execution mode
--------------------------

Three modes control how dependencies get into the container:

- **Bundled** (default) — Kinetic builds a custom image with your deps
  baked in. Best for stable workflows and reproducible runs.
- **Prebuilt** — pulls a published base image, installs your deps at
  pod startup. Best for fast iteration when deps change often.
- **Custom image** — bring your own image URI. Best when you need
  custom system libraries or a corporate-vetted base.

See :doc:`Execution Modes <guides/execution_modes>` for the full
recommendation matrix and per-mode startup expectations.
