.. image:: https://github.com/jeffwigger/FastDynamicBatcher/actions/workflows/test_pip.yaml/badge.svg
     :target: https://github.com/JeffWigger/FastDynamicBatcher/actions
     :alt: Test Workflow

Fast Dynamic Batcher
====================

Batching machine learning workloads is the easiest way to achieve
significant inference speedups. The **Fast Dynamic Batcher** Library has
been built to add easy support for dynamic batchtes to FastAPI. With our
dynamic batcher, you can combine the inputs of several requests into a
single batch, which will can then be run on your GPU. With our example
project we measured speed-ups of up to 2.8x with dynamic batching
compared to a baseline without it.

Example Usage
-------------

To use dynamic batching in your FastAPI workloads, you have to first
create an instance of the ``InferenceModel`` class. Initiate your ML
model in the ``init`` method and use it in the ``infer`` method:

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
and\ ``max_delay`` variables. Subsequently, it can be used in your
FastAPI endpoints by registering your inputs by calling its
``process_batched`` method.

Our dynamic batching algorithm will then wait for either the number of
inputs to equal ``max_batch_size``, or until ``max_delay`` seconds have
passed. In the latter case, a batch may contain between 1 and
``max_batch_size`` inputs. Once, either condition is met, a batch will
be processed by calling the ``infer`` method of your ``InferenceModel``
instance.
