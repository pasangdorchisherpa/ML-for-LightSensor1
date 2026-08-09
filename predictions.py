## Loads the main model
## Loads the new backup model on the back as well
## Idea for the second file is the be the one that changes when operator ends up finding 1 new better model
## As the code for predicitons need to run for 5 seconds everytime without fail, we want the swap to only occur to second on at anytime but then replacing (new)second model on the mainone only happens at the end of prediciton and then the swap to new model path occurs replacing the new model safely.

## Create some how for the code to know that a new second model path(right now the main_model and second model is the same thing, so when new second_model does come then we do the clean swap like this)

# predictions.py
import threading
import queue
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)


class ModelManager:
    """Holds the live model and handles safe hot-swaps between cycles."""
    def __init__(self, main_model_path):
        self._lock = threading.Lock()
        self.model_path = main_model_path
        self.model = tf.keras.models.load_model(main_model_path)
        self._pending_swap_path = None

    def request_swap(self, new_model_path):
        """Call this any time (e.g. from your retrain app) — just queues the swap."""
        with self._lock:
            self._pending_swap_path = new_model_path

    def maybe_swap(self):
        """Call ONLY between prediction cycles — this is the safe point."""
        with self._lock:
            if self._pending_swap_path:
                try:
                    new_model = tf.keras.models.load_model(self._pending_swap_path)
                    self.model = new_model
                    self.model_path = self._pending_swap_path
                    logger.info(f"Swapped model to {self._pending_swap_path}")
                except Exception as e:
                    logger.error(f"Model swap failed, keeping old model: {e}")
                finally:
                    self._pending_swap_path = None

    def predict(self, data):
        return self.model.predict(data, verbose=0)


def prediction_worker(data_queue: queue.Queue, model_manager: ModelManager, result_callback=None):
    """
    Runs in its own thread. Blocks on data_queue.get(), so it's synced
    exactly to whatever cadence main.py actually writes at — no drift.
    """
    while True:
        regs = data_queue.get()
        if regs is None:  # sentinel to shut the thread down cleanly
            break

        model_manager.maybe_swap()  # safe point: no inference in flight

        try:
            preds = model_manager.predict(regs)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            continue

        if result_callback:
            result_callback(preds)