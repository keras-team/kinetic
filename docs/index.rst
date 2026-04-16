Kinetic: Run ML workloads on cloud TPUs and GPUs
================================================

.. toctree::
   :caption: Documentation
   :hidden:

   getting_started
   architecture
   troubleshooting

.. toctree::
   :caption: Guides
   :hidden:

   guides/execution_modes
   guides/keras_training
   guides/jax_training
   guides/pytorch_training
   guides/data
   guides/dependencies
   guides/env_vars
   guides/llm_finetuning
   guides/distributed_training
   guides/checkpointing
   guides/cost_optimization

.. toctree::
   :caption: Reference
   :hidden:

   api
   cli
   accelerators
   configuration

.. toctree::
   :caption: Advanced Topics
   :hidden:

   advanced/async_jobs
   advanced/clusters
   advanced/containers

.. toctree::
   :caption: Community
   :hidden:

   contributing
   code-of-conduct

Kinetic is a library that enables running Python functions seamlessly on cloud
TPUs and GPUs using a simple decorator: ``@kinetic.run``. No infrastructure
management required.

.. code-block:: python
   :emphasize-lines: 3

    import kinetic

    @kinetic.run(accelerator="v6e-8")
    def train_model():
        import keras
        model = keras.Sequential([...])
        model.fit(x_train, y_train)
        return model.history.history["loss"][-1]

    # Executes on TPU v6e-8, returns the result
    final_loss = train_model()


How It Works
------------

When you call a decorated function, Kinetic handles the entire remote execution pipeline:

1. **Packages** your function, local code, and data dependencies.
2. **Builds a container** with your dependencies via Cloud Build.
3. **Runs the job** on a GKE cluster with the requested accelerator (TPU or GPU).
4. **Returns the result** to your local machine.

Get Started
-----------

Follow `this guide <getting_started.md>`__ to get started.
