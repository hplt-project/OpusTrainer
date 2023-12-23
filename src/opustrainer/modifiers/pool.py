import logging
import os
import random

from multiprocessing import Queue, Process
from logging.handlers import QueueHandler, QueueListener
from typing import List, Union
from itertools import chain

from opustrainer.modifiers import Modifier


class ModifierWorker(Process):
    """Process that runs batches of sentences through a list of modifiers"""
    tasks: Queue
    results: Queue
    messages: Queue
    modifiers: List[Modifier]

    def __init__(self, tasks:Queue, results:Queue, messages:Queue, modifiers:List[Modifier], **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.results = results
        self.messages = messages
        self.modifiers = modifiers
        
    def run(self):
        handler = QueueHandler(self.messages)
        logging.getLogger().addHandler(handler)
    
        while True:
            task = self.tasks.get()

            # If task is None, this worker can stop.
            if task is None:
                break

            # A task consists of a chunk id, batch seed, and lines
            chunk, seed, batch = task

            try:
                # Set random seed for this batch, so the worker and the order
                # in which batches are processed are no longer relevant
                random.seed(seed)

                for modifier in self.modifiers:
                    batch = list(modifier(batch))

                self.results.put((chunk, batch, None))
            except Exception as exc:
                self.results.put((chunk, None, exc))
        self.results.close()


class ModifierPool:
    """Pool of ModifierWorker that exposes `map()` to run a batch of sentences
    through a predefined list of modifiers. Similar to multiprocessing.Pool
    except that the `func` argument doesn't need to be passed for each call.
    """

    """Number of worker processes in the pool"""
    workers: int

    """Modifier list each worker applies to the batches"""
    modifiers: List[Modifier]

    """Queue for submitting chunks of work to the workers"""
    tasks: Queue

    """Queue for receiving chunks from the workers"""
    results: Queue

    messages: Queue

    log_worker: QueueListener

    def __init__(self, modifiers:List[Modifier], processes:int=0):
        self.modifiers = modifiers
        self.workers = processes if processes > 0 else min(os.cpu_count() or 1, 8)

    def __enter__(self) -> 'ModifierPool':
        self.tasks = Queue()
        self.results = Queue()
        
        self.messages = Queue()
        self.log_worker = QueueListener(self.messages, *logging.getLogger().handlers, respect_handler_level=True)
        self.log_worker.start()

        self.processes = [
            ModifierWorker(self.tasks, self.results, self.messages, self.modifiers, daemon=True)
            for _ in range(self.workers)
        ]

        for process in self.processes:
            process.start()

        return self

    def __exit__(self, *args):
        # Tell workers to stop
        for _ in self.processes:
            self.tasks.put(None)
        self.tasks.close()

        # Wait for workers to close down
        for process in self.processes:
            process.join()

        self.log_worker.stop()
        self.messages.close()

    def map(self, batch:List[str], chunksize:int=0) -> List[str]:
        if chunksize > 0:
            chunks, remainder = divmod(len(batch), chunksize)
        else:
            chunks = len(self.processes)
            chunksize, remainder = divmod(len(batch), chunks)

        # Shortcut for getting the slice of batch for each chunk
        chunk_slice = lambda chunk: slice(
            chunk * chunksize,
            chunk * chunksize + (chunksize if chunk < chunks else remainder)
        )

        # Submit tasks to workers
        for chunk in range(chunks + (1 if remainder > 0 else 0)):
            self.tasks.put((chunk, random.random(), batch[chunk_slice(chunk)]))

        # Placeholder for the returned chunks, in order
        chunk_results = [[]] * (chunks + (1 if remainder > 0 else 0))

        # Retrieve results from workers
        for _ in range(chunks + (1 if remainder > 0 else 0)):
            chunk, result, exc = self.results.get()
            if exc is not None:
                raise exc
            chunk_results[chunk] = result

        # Stitch the ordered result chunks back together into a single batch
        return list(chain(*chunk_results))


class ErzatsModifierPool:
    """Same as ModifierPool, but does all the work on the main thread."""
    def __init__(self, modifiers:List[Modifier], processes:int=0):
        self.modifiers = modifiers

    def __enter__(self) -> 'ErzatsModifierPool':
        return self

    def __exit__(self, *args):
        pass

    def map(self, batch:List[str], chunksize:int=0) -> List[str]:
        if chunksize > 0:
            chunks, remainder = divmod(len(batch), chunksize)
        else:
            raise ValueError("Need a chunksize > 0")

        # Shortcut for getting the slice of batch for each chunk
        chunk_slice = lambda chunk: slice(
            chunk * chunksize,
            chunk * chunksize + (chunksize if chunk < chunks else remainder)
        )

        tasks = []

        # Submit tasks to workers
        for chunk in range(chunks + (1 if remainder > 0 else 0)):
            tasks.append((chunk, random.random(), batch[chunk_slice(chunk)]))

        # Placeholder for the returned chunks, in order
        chunk_results = [[]] * (chunks + (1 if remainder > 0 else 0))

        random_state = random.getstate()

        for chunk, seed, batch in tasks:
            # Set random seed for this batch, so the worker and the order
            # in which batches are processed are no longer relevant
            random.seed(seed)

            for modifier in self.modifiers:
                batch = list(modifier(batch))

            chunk_results[chunk] = batch

        random.setstate(random_state)
        
        # Stitch the ordered result chunks back together into a single batch
        return list(chain(*chunk_results))


def make_modifier_pool(modifiers:List[Modifier], processes:int) -> Union[ModifierPool, ErzatsModifierPool]:
    if processes == 0:
        return ErzatsModifierPool(modifiers, processes)
    else:
        return ModifierPool(modifiers, processes)
