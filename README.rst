.. image:: https://github.com/jeffwigger/FastDynamicBatcher/actions/workflows/test_pip.yaml/badge.svg
     :target: https://github.com/JeffWigger/FastDynamicBatcher/actions
     :alt: Workflow Status

Fast Dynamic Batcher
====================

The bundling of several machine learning model inputs into a single larger input is the simplest way to achieve
significant inference speed-ups in ML workloads. The **Fast Dynamic Batcher** library has
been built to make it easy to use such dynamic batches in Python web frameworks like FastAPI. With our
dynamic batcher, you can combine the inputs of several requests into a
single batch, which can then be run more efficiently on GPUs. In our testing, we achieved up to 2.5x more throughput with it.

Example Usage
-------------

To use dynamic batching in FastAPI, you have to first
create an instance of the ``InferenceModel`` class. Initiate your ML
model in its ``init`` method and use it in its ``infer`` method:

.. code-block:: python

   from fast_dynamic_batcher.dyn_batcher import Task
   from fast_dynamic_batcher.inference_template import InferenceModel


   class DemoModel(InferenceModel):
      def __init__(self):
          super().__init__()
          # Initiate your ML model here

      def infer(self, tasks: list[Task]) -> list[Task]:
          # Process your input tasks
          inputs = [t.content for t in tasks]
          # Run your inputs as a batch for your model
          ml_output = None # Your inference outputs
          results = [
              Task(id=tasks[i].id, content=ml_output[i]) for i in range(len(tasks))
          ]
          return results

Subsequently, use your ``InferenceModel`` instance to initiate our
``DynBatcher``:

.. code-block:: python

   from contextlib import asynccontextmanager

   from anyio import CapacityLimiter
   from anyio.lowlevel import RunVar

   from fast_dynamic_batcher.dyn_batcher import DynBatcher


   @asynccontextmanager
   async def lifespan(app: FastAPI):
      RunVar("_default_thread_limiter").set(CapacityLimiter(16))
      global dyn_batcher
      dyn_batcher = DynBatcher(DemoModel, max_batch_size = 8, max_delay = 0.1)
      yield
      dyn_batcher.stop()

   app = FastAPI(lifespan=lifespan)

   @app.post("/predict/")
   async def predict(
      input_model: YourInputPydanticModel
   ):
      return await dyn_batcher.process_batched(input_model)

The ``DynBatcher`` can be initiated in the FastAPI lifespans as a global
variable. It can be further customized with the ``max_batch_size``
and ``max_delay`` variables. Subsequently, use it in your
FastAPI endpoints by registering your inputs by calling its
``process_batched`` method.

Our dynamic batching algorithm will then wait for either the number of
inputs to equal the ``max_batch_size``, or until ``max_delay`` seconds have
passed. In the latter case, a batch may contain between 1 and
``max_batch_size`` inputs. Once, either condition is met, a batch will
be processed by calling the ``infer`` method of your ``InferenceModel``
instance.

Installation
------------

The Fast Dynamic Batcher library can be installed with pip:

.. code-block:: bash

   pip install fast_dynamic_batcher


Performance Tests
-----------------

We tested the performance of our dynamic batching solution against a baseline without batching on a Colab instance with a T4 GPU as well as on a laptop with an Intel i7-1250U CPU.
The experiments were conducted by using this `testing script <https://github.com/JeffWigger/FastDynamicBatcher/blob/main/test/test_dyn_batcher.py>`_. The results are reported in the table below:

.. list-table:: Performance Experiments
   :widths: 40 30 30
   :header-rows: 1

   * - Hardware
     - No Batching
     - Dynamic Batch size of 16
   * - Colab T4 GPU
     - 7.65s
     - 3.07s
   * - CPU Intel i7-1250U
     - 117.10s
     - 88.47s

On GPUs, which benefit greatly from large batch sizes, we achieved a speed-up of almost 2.5x by creating dynamic batches of size 16. On, CPUs, the gains are more modest with a speed-up of 1.3x.
