import asyncio
import json
import time

import pytest

from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient


LOCAL = False


# TODO: asyncio_mode=auto
@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_one(wrapped_app):
    async with LifespanManager(wrapped_app) as manager:
        start_s = time.monotonic()
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            response = await client.post("http://test/predict", json={"text": "hi"})
        took_s = time.monotonic() - start_s
        if LOCAL:
            assert (took_s > 0.05) and (took_s < 0.17)
        assert response.status_code == 200
        assert json.loads(response.json()) == {"text": "hi"}


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_eight(wrapped_app):
    async with LifespanManager(wrapped_app) as manager:
        start_s = time.monotonic()
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            requests = []
            for i in range(8):
                requests.append(client.post("http://test/predict", json={"text": f"{i}"}))
            results = await asyncio.gather(*requests)
        took_s = time.monotonic() - start_s
        if LOCAL:
            assert (took_s > 0.05) and (took_s < 0.2)
        assert all([response.status_code == 200 for response in results])


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_twelve(wrapped_app):
    async with LifespanManager(wrapped_app) as manager:
        start_s = time.monotonic()
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            requests = []
            for i in range(12):
                requests.append(client.post("http://test/predict", json={"text": f"{i}"}))
            results = await asyncio.gather(*requests)
        took_s = time.monotonic() - start_s
        if LOCAL:
            assert (took_s > 0.2) and (took_s < 0.3)
        assert all([response.status_code == 200 for response in results])


@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.asyncio
async def test_hundred(wrapped_app):
    async with LifespanManager(wrapped_app) as manager:
        start_s = time.monotonic()
        async with AsyncClient(transport=ASGITransport(app=manager.app), base_url="http://test") as client:
            requests = []
            for i in range(100):
                requests.append(client.post("http://test/predict", json={"text": f"{i}"}))
            results = await asyncio.gather(*requests)
        took_s = time.monotonic() - start_s
        print(took_s)
        if LOCAL:
            assert (took_s > 0.75) and (took_s < 0.9)
        assert all([response.status_code == 200 for response in results])
