import asyncio

import pytest

from fast_dynamic_batcher.errors import DynamicBatchIndexError


@pytest.mark.asyncio
async def test_raised_errors_not_returned(dyn_batcher):
    await_list = []
    for i in range(5):
        await_list.append(dyn_batcher.process_batched(i))
    with pytest.raises(DynamicBatchIndexError) as exc_info:
        await asyncio.gather(*await_list, return_exceptions=False)
    assert exc_info.type is DynamicBatchIndexError
    assert exc_info.value.message == "Dynamic batch of input size 5 produced 4."


@pytest.mark.asyncio
async def test_raised_errors_returned(dyn_batcher):
    await_list = []
    for i in range(5):
        await_list.append(dyn_batcher.process_batched(i))
    results = await asyncio.gather(*await_list, return_exceptions=True)

    assert all([isinstance(res, DynamicBatchIndexError) for res in results])
    assert all([res.message == "Dynamic batch of input size 5 produced 4." for res in results])
