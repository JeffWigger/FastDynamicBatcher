import asyncio
import logging
import multiprocessing as mp
import typing as t

from concurrent.futures import Future
from ctypes import c_bool
from threading import Event, Lock, Thread

from fast_dynamic_batcher.dyn_prog_loop import _main_loop, _return_loop
from fast_dynamic_batcher.inference_template import InferenceModel
from fast_dynamic_batcher.models import Task


try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


class DynBatcher:
    """The DynBatcher class is used to batch inputs across many requests into a single batch to accelerate ML workloads."""

    def __init__(
        self,
        inference_model: InferenceModel,
        max_batch_size: int = 8,
        max_delay: float = 0.1,
        queue_size: int = 10000,
    ):
        """
        Initializes the Dynamic Batcher.

        :param inference_model: Implementation of the abstract InferenceModel class
        :type inference_model: InferenceModel
        :param max_batch_size: The maximal number of inputs that are processed together, defaults to 8
        :type max_batch_size: int, optional
        :param max_delay: The maximal delay between receiving the first input and the start of the batch processing, defaults to 0.1
        :type max_delay: float, optional
        :param queue_size: The maximal size of the queue holding batched inputs, defaults to 10000
        :type queue_size: int, optional
        """
        # mp.set_start_method("spawn", force=True)  # otherwise torch does not work.
        self.logger = mp.get_logger()  # mp.log_to_stderr()
        self.logger.setLevel(logging.INFO)
        self._input_queue = mp.Queue(queue_size)
        self._output_queue = mp.Queue(queue_size)
        self._unique_id_counter = 0
        self._state_map: dict[int, Future] = {}
        self._state_lock = Lock()
        self._lock = asyncio.Lock()
        self.shared_bool = mp.Value(c_bool, True)  # TODO: Check if RawValue is faster because of no lock
        self._batcher = mp.Process(
            target=_main_loop,
            args=(inference_model, self._input_queue, self._output_queue, self.shared_bool, max_batch_size, max_delay),
            daemon=True,
        )

        # TODO: enable creation as singleton
        self._batcher.start()
        self.logger.info("Batcher process started")

        # TODO: create a coroutine / Task that runs endlessly and processes the output queue
        # Maybe with asyncio.to_thread (just runs it in a different thread, still needs await)
        # loop.create_task()
        self.event = Event()
        self.thread = Thread(
            target=_return_loop, args=(self._state_map, self._state_lock, self._output_queue, self.event)
        )
        self.thread.start()

        self.logger.info("Output thread started")

    def stop(self):
        """
        Stops all processes and threads created by the DynBatcher.
        """
        self._input_queue.close()
        self._output_queue.close()
        self.event.set()  # joins self.thread
        self.shared_bool.value = False
        self._batcher.join()
        self.thread.join()
        # TODO: Terminate if join not working

    async def process_batched(self, input: t.Any) -> t.Any:
        """
        Process an input as a batch.

        :param input: The input which will be passed to the `infer` of the registered InferenceModel as the content of a Task.
        :type input: t.Any
        :return: The output of the InferenceModel's `infer` method.
        :rtype: t.Any
        """
        result = await asyncio.to_thread(self._process_batched, input)
        return result

    def _process_batched(self, input: t.Any) -> t.Any:
        future = Future()
        # async with self._lock:
        task_id = self._unique_id_counter
        # self._state_lock.acquire_lock()  # Todo: is the lock actually needed
        self._state_map[self._unique_id_counter] = future
        # self._state_lock.release_lock()
        self._unique_id_counter += 1

        task = Task(id=task_id, content=input)
        self._input_queue.put(task, False)  # raises queue.Full
        # TODO: error handling
        future_result = future.result()
        return future_result
