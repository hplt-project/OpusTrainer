#!/usr/bin/env bash
opustrainer-train -c test_enzh_config.yml --sync -n -d -b 1 -B 1 -j 1

# All arguments after config are to just make example single threaded and not shuffled
