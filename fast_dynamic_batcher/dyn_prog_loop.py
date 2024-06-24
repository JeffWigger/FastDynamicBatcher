import time

from asyncio import Future
from multiprocessing import Queue, Value
from threading import Event, Lock

from fast_dynamic_batcher.errors import DynamicBatchIndexError
from fast_dynamic_batcher.inference_template import InferenceModel
from fast_dynamic_batcher.models import Task


# TODO: Bound exp. backoff
RETURN_LOOP_SLEEP = 0.001
MAIN_LOOP_SLEEP = 0.001

TICKS_IN_NANO = 1000000000


def _main_loop(
    inference_model: InferenceModel,
    input_queue: Queue,
    output_queue: Queue,
    shared_bool: Value,
    max_batch_size: int = 8,
    max_delay: float = 0.1,
):
    inference_model = inference_model()
    start_time = time.monotonic_ns()
    max_delay_ns = int(TICKS_IN_NANO * max_delay)
    batch_list = []

    # TODO: add flag to stop the loop
    while shared_bool.value:
        if len(batch_list) == 0:
            start_time = time.monotonic_ns()
        no_op = True
        if not input_queue.empty():
            no_op = False
            # should not block
            input = input_queue.get()
            batch_list.append(input)
        if len(batch_list) == max_batch_size or start_time + max_delay_ns < time.monotonic_ns():
            no_op = False
            inputs = [t.content for t in batch_list]
            outputs = inference_model.infer(inputs)
            if len(outputs) != len(inputs):
                output_tasks = [
                    Task(
                        id=batch_list[i].id,
                        content=DynamicBatchIndexError(
                            f"Dynamic batch of input size {len(batch_list)} produced {len(outputs)}."
                        ),
                    )
                    for i in range(len(batch_list))
                ]
            else:
                output_tasks = [Task(id=batch_list[i].id, content=outputs[i]) for i in range(len(batch_list))]
            for output in output_tasks:
                output_queue.put(output)
            batch_list = []
        if no_op:
            time.sleep(MAIN_LOOP_SLEEP)


def _return_loop(state_map: dict[int, Future], state_lock: Lock, output_queue: Queue, event: Event):
    while not event.is_set():
        while not output_queue.empty():
            # shuold never block
            output = output_queue.get()
            # TODO: error handling
            state_lock.acquire()
            output_future = state_map[output.id]
            del state_map[output.id]
            state_lock.release()
            output_future.set_result(output.content)
        time.sleep(RETURN_LOOP_SLEEP)
