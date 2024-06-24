import asyncio
import time

from urllib.request import urlretrieve

import httpx

from PIL import Image


urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")


async def main():
    urlretrieve("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
    async with httpx.AsyncClient() as client:
        requests = []
        dog = Image.open("dog.jpg")
        dog_bytes = dog.tobytes()
        for _ in range(30):
            files = {"file": ("dog.jpq", dog_bytes)}  # open('dog.jpg', 'rb')
            requests.append(
                client.post(
                    "http://localhost:8080/predict/",
                    files=files,
                    data={"mode": dog.mode, "size": dog.size},
                    timeout=100,
                )
            )
        res = await asyncio.gather(*requests)
        print([r.content for r in res])


if __name__ == "__main__":
    start_s = time.time()
    asyncio.run(main())
    print(f"Execution took {time.time() - start_s} seconds.")
