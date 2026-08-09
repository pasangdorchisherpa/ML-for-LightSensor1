# main.py (additions)
import queue
import threading
from predictions import prediction_worker, ModelManager

data_queue = queue.Queue(maxsize=1)  # size 1: always the freshest reading
model_manager = ModelManager(main_model_path=r"path/to/main_model")

worker_thread = threading.Thread(
    target=prediction_worker,
    args=(data_queue, model_manager),
    daemon=True,
)
worker_thread.start()

while True:
    regs = poll_modbus()  # your existing polling code

    # non-blocking put; if predictions.py is still busy, drop the
    # stale item so we never predict on old data
    try:
        data_queue.put_nowait(regs)
    except queue.Full:
        data_queue.get_nowait()
        data_queue.put_nowait(regs)

    time.sleep(5)