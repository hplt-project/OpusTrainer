#!/usr/bin/env python3
import os
import sys
import json
import time

from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from shutil import copyfileobj

root = os.path.dirname(os.path.abspath(__file__))

def benchmark(name, config, args):
	with NamedTemporaryFile('w') as config_file:
		json.dump(config, config_file)
		config_file.flush()

		start_time = time.monotonic_ns()

		child = Popen([sys.executable, '-m', 'opustrainer', '--workers', '4', '-c', config_file.name, *args],
			stdout=sys.stderr,
			stderr=sys.stderr)

		if child.wait() != 0:
			raise RuntimeError(f'Child exited with exit code {child.returncode}')

		run_time = time.monotonic_ns() - start_time

		return {
      "name": name,
      "unit": "ns",
      "value": run_time
    }

with NamedTemporaryFile('w') as testdata:
	with open(os.path.join(root, 'test-data/clean.enzh.10')) as fh:
		for _ in range(10_000):
			fh.seek(0)
			copyfileobj(fh, testdata)

	testdata.flush()

	base = {
		'datasets': {
			'clean': testdata.name
		},
		'stages': [
			'start'
		],
		'start': [
			'clean 1',
			'until clean 1'
		],
		'modifiers': [],
		'seed': 1111,
		'trainer': 'dd if=/dev/stdin of=/dev/null'
	}

	scenarios = [
		{
			"name": "Baseline",
			"args": [],
			"config": {
				**base
			}
		},
		{
			"name": "Tags",
			"args": [],
			"config": {
				**base,
				"modifiers": [
					{
						"Tags": 0.1,
						"custom_detok_trg": "zh"
					}
				]
			}
		},
		{
			"name": "Tags with SPM",
			"args": [],
			"config": {
				**base,
				"modifiers": [
					{
						"Tags": 0.1,
						"custom_detok_trg": "zh",
						"spm_vocab_src": os.path.join(root, "test-data/vocab.zhen.spm"),
						"spm_vocab_trg": os.path.join(root, "test-data/vocab.zhen.spm")
					}
				]
			}
		},
	]

	json.dump([
		benchmark(**scenario)
		for scenario in scenarios
	], sys.stdout)
