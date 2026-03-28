Kinetic: Run Keras and JAX workloads on cloud TPUs and GPUs
=============================================================

.. toctree::
   :maxdepth: 2
   :caption: Documentation
   :hidden:

   getting_started
   usage
   troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api
   cli

.. toctree::
   :maxdepth: 1
   :caption: Community
   :hidden:

   contributing
   code-of-conduct

Run Keras and JAX workloads on cloud TPUs and GPUs with a simple decorator. No infrastructure management required.

.. code-block:: python

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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
