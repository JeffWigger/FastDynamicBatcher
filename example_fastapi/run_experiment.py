import subprocess
import time

from multiprocessing import Process


def f():
    a = subprocess.run(["python", "example_fastapi/app.py"], check=True)
    print(a)


def g():
    a = subprocess.run(["python", "example_fastapi/send_requests.py"], check=True, capture_output=True)
    print(a)


if __name__ == "__main__":
    t = time.time()
    p = Process(target=f)
    p.start()
    time.sleep(10)
    p2 = Process(target=g)
    p3 = Process(target=g)
    p2.start()
    p3.start()
    p2.join()
    p3.join()
    p.terminate()
    print(time.time() - t)
