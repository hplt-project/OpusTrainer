#!/usr/bin/env python3
import heapq
import os
import subprocess
from argparse import ArgumentParser, FileType
from dataclasses import dataclass
from itertools import islice, chain
from operator import itemgetter
from queue import Queue
from random import Random
from shutil import which
from struct import Struct
from tempfile import mkstemp
from threading import Thread
from typing import TypeVar, Iterator, Iterable, List, Optional, Tuple, Callable


# Buffer size for reading files. Bufsize that Python assigns is generally too small?
BUFSIZE=2**16

# Prefer pigz if available, but fall back to calling gzip
PATH_TO_GZIP = which("pigz") or which("gzip")

HEADER = Struct('@fI') # f for random float, I for line length


@dataclass(frozen=True)
class SortTask:
	"""Job that describes to shuffle a chunk to the shuffle_chunk_worker thread.
	Passing along the seed created by the main thread because those
	random.random() calls are predictable. The order in which Shuffling tasks
	are picked up and finished may not be."""
	fileno: int
	chunk: List[Tuple[float,bytes]]

	def __call__(self) -> None:
		with os.fdopen(self.fileno, 'wb', buffering=BUFSIZE) as fh:
			self.chunk.sort(key=itemgetter(0))
			for rand, line in self.chunk:
				fh.write(HEADER.pack(rand, len(line)))
				fh.write(line)


def task_worker(queue:"Queue[Optional[Callable[[],None]]]") -> None:
	"""Worker thread that takes a queue of filenames and seeds, and shuffles them
	in memory. Put a None in the queue to make it stop."""
	while True:
		task = queue.get()
		if task is None:
			break
		task()


def iter_shuffled_file(filename:str) -> Iterable[Tuple[float,bytes]]:
	with open(filename, 'rb', buffering=BUFSIZE) as fh:
		while True:
			header = fh.read(HEADER.size)
			if header == b'':
				break
			random, length = HEADER.unpack(header)
			yield random, fh.read(length)


def shuffle(fin: Iterable[bytes], lines:int, *, seed:Optional[int]=None, threads:int=1, tmpdir:Optional[str]=None) -> Iterable[bytes]:
	"""Shuffle a list by reading it into a bunch of files (of `lines` length)
	and shuffling all of these with `threads` in-memory sorters."""
	random = Random(seed)

	chunks: List[str] = []

	try:
		if threads > 0:
			# Limiting queue to 1 pending chunk otherwise we'll run out of memory quickly.
			queue: "Queue[Optional[SortTask]]" = Queue(maxsize=threads)

			# Prepare shuffle workers to start shuffling chunks as soon as we've
			# finished writing them.
			sorters = [
				Thread(target=task_worker, args=[queue])
				for _ in range(threads)
			]

			try:
				for sorter in sorters:
					sorter.start()

				# Split the input file into separate temporary chunks
				line_it = iter(fin)
				while True:
					chunk = [(random.random(), line) for line in islice(line_it, lines)]
					if not chunk:
						break

					fileno, filename = mkstemp(dir=tmpdir)
					# Remember the chunk's filename for later
					chunks.append(filename)
					# And immediately start shuffling & writing that chunk in another thread
					# so we can use this thread to continue ingesting chunks
					queue.put(SortTask(fileno, chunk))
			finally:
				# Tell sorters that they can stop waiting
				for _ in sorters:
					queue.put(None)

				# Wait for them to finish shuffling the last files
				for sorter in sorters:
					sorter.join()
		else:
			line_it = iter(fin)
			while True:
				chunk = [(random.random(), line) for line in islice(line_it, lines)]
				if not chunk:
					break

				fileno, filename = mkstemp(dir=tmpdir)
				chunks.append(filename)

				task = SortTask(fileno, chunk)
				task()				

		# Open all chunks. We'll be reading the next line from a random one of them.
		chunk_fds = [iter_shuffled_file(filename) for filename in chunks]

		# Use heap merge to read the next smallest random element from chunk_fds
		# which are already sorted.
		for _, line in heapq.merge(*chunk_fds, key=itemgetter(0)):
			yield line

	finally:
		# Whatever happened, if a filename of a temporary file made it into the
		# `chunks` list, we are responsible for cleaning it up.
		for filename in chunks:
			os.unlink(filename)


class Reader(Iterable[bytes]):
	"""Lazily opens a file only once you start trying to read it. Also magically
	reads gzipped files."""
	def __init__(self, filename:str):
		self.filename = filename

	def _read_gzip(self, filename:str) -> Iterable[bytes]:
		"""Open gzipped files through gzip subprocess. It is faster than Python's
		gzip submodule, and you get a bit of multiprocessing for free as the
		external gzip process can decompress up to BUFSIZE while python is doing
		other things."""
		assert PATH_TO_GZIP is not None, 'No gzip executable found on system'
		child = subprocess.Popen([PATH_TO_GZIP, '-cd', filename], stdout=subprocess.PIPE, bufsize=BUFSIZE)
		assert child.stdout is not None
		yield from child.stdout
		if child.wait() != 0:
			raise RuntimeError(f'`{PATH_TO_GZIP} -cd {filename}` failed with return code {child.returncode}')

	def _read_plain(self, filename:str) -> Iterable[bytes]:
		with open(filename, 'rb') as fh:
			yield from fh

	def __iter__(self) -> Iterator[bytes]:
		if self.filename.endswith('.gz'):
			return iter(self._read_gzip(self.filename))
		else:
			return iter(self._read_plain(self.filename))


def main() -> None:
	parser = ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=1_000_000, help='number of lines per chunk. Note that these chunks are read into memory when being shuffled')
	parser.add_argument('--threads', '-j', type=int, default=0, help=f'number of concurrent shuffle threads. Defaults to none')
	parser.add_argument('--temporary-directory', '-T', type=str, help='temporary directory for shuffling batches')
	parser.add_argument('--no-shuffle', '-n', action="store_false", help='Do not shuffle, to be used for debugging', dest="shuffle")
	parser.add_argument('seed', type=int)
	parser.add_argument('output', type=FileType('wb', bufsize=BUFSIZE), default='-')
	parser.add_argument('files', nargs='+')

	args = parser.parse_args()

	# Read the lines
	it: Iterable[bytes] = chain.from_iterable(Reader(filename) for filename in args.files)

	# Shuffle the lines
	if args.shuffle:
		it = shuffle(it, lines=args.batch_size, seed=args.seed, threads=args.threads, tmpdir=args.temporary_directory)

	args.output.writelines(it)


if __name__ == '__main__':
	main()
