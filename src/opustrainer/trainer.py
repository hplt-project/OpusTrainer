#!/usr/bin/env python3
"""A translation model trainer. It feeds marian different sets of datasets with different thresholds
for different stages of the training. Data is uncompressed and TSV formatted src\ttrg
"""
import os
import sys
import signal
import argparse
import random
import subprocess
import shlex
import time

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union, Type, TextIO, cast, Iterable, Iterable, TypeVar, get_type_hints, get_args, get_origin
from tempfile import TemporaryFile
from itertools import islice
from pathlib import Path

import yaml

from opustrainer.modifiers import Modifier
from opustrainer.modifiers.merge import MergeModifier
from opustrainer.modifiers.noise import NoiseModifier
from opustrainer.modifiers.prefix import PrefixModifier
from opustrainer.modifiers.surface import UpperCaseModifier, TitleCaseModifier
from opustrainer.modifiers.placeholders import PlaceholderTagModifier
from opustrainer.modifiers.typos import TypoModifier
from opustrainer.modifiers.retokenize import RetokenizeModifier
from opustrainer.modifiers.pool import make_modifier_pool
from opustrainer import logger

def ignore_sigint():
    """Used as pre-exec hook for the trainer program as to ignore ctrl-c. We'll
    deal with ctrl-c in the python program, and then be very friendly about
    stopping the trainer.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


# Available batch modifiers
# TODO: Import these lazy, on demand?
MODIFIERS = {
    'Merge': MergeModifier,
    'Noise': NoiseModifier,
    'UpperCase': UpperCaseModifier,
    'TitleCase': TitleCaseModifier,
    'Tags': PlaceholderTagModifier,
    'Typos': TypoModifier,
    'Prefix': PrefixModifier,
    'Retokenize': RetokenizeModifier,
}

@dataclass(frozen=True)
class Dataset:
    name: str
    files: List[str]


@dataclass(frozen=True)
class DatasetState:
    seed: int
    line: int
    epoch: int


@dataclass(frozen=True)
class Stage:
    name: str
    datasets: List[Tuple[Dataset, float]]
    until_dataset: str
    until_epoch: Optional[int]
    modifiers: Optional[List[Modifier]]


@dataclass(frozen=True)
class Curriculum:
    seed: int
    datasets: Dict[str,Dataset]
    stages: Dict[str,Stage]
    modifiers: List[Modifier]
    stages_order: List[str]

    # Optionally, check how many fields each row of the datasets should have
    # Too many should select the N first fields. Too few should drop the row.
    num_fields: Optional[int]

    def __post_init__(self):
        if len(self.stages) != len(frozenset(self.stages)):
            raise ValueError('stages can only occur once')

        if not (frozenset(self.stages) <= frozenset(self.stages.keys())):
            raise ValueError('each stage has to be defined')

    def next_stage(self, stage:Stage) -> Optional[Stage]:
        """Helper to get the next stage given the current stage."""
        index = self.stages_order.index(stage.name)
        if index + 1 < len(self.stages_order):
            return self.stages[self.stages_order[index + 1]]
        else:
            return None


@dataclass(frozen=True)
class EpochTrackerState:
    epoch: int
    line: int


@dataclass(frozen=True)
class TrainerState:
    stage: str
    random_state: Any # whatever the type is returned by random.getstate(), which I think is implementation specific.
    epoch_tracker_state: EpochTrackerState
    datasets: Dict[str,DatasetState]


class DatasetReader:
    """Repeats, shuffles and reads a dataset ad infinitum."""
    dataset: Dataset
    seed: int
    line: int
    epoch: int
    shuffle: bool
    num_fields: Optional[int]

    tmpdir: Optional[str]

    _fh: Optional[TextIO] = None
    _next_line: str

    def __init__(self, dataset:Dataset, seed:int, tmpdir:Optional[str]=None, shuffle:bool=True,
                 num_fields:Optional[int]=None):
        """
        Parameters
        ----------
        dataset : Dataset
            Description of the dataset and its files
        seed : int
            Seed number for the random number generator that shuffles the data internally
        tmpdir : str, optional
            Path to directory in which the temporary shuffled dataset is written (default is `tempfile.gettempdir()`)
        shuffle : bool
            Indicates whether shuffling should happen. Enabled by default.
        num_fields: int, optional
            Optionally specify the number of fields each line should have. Trim to the required number if they are
            more than the necessary fields, or remove lines that don't have the required number of fields.
        """
        self.dataset = dataset
        self.seed = seed
        self.tmpdir = tmpdir
        self.epoch = 0
        self.line = 0
        self.shuffle = shuffle
        self.num_fields = num_fields

    def state(self) -> DatasetState:
        return DatasetState(self.seed, self.line, self.epoch)

    def restore(self, state:DatasetState) -> 'DatasetReader':
        self.close()

        self.seed = state.seed
        self.epoch = state.epoch

        # Skip forward
        for _ in range(state.line):
            next(self)

        return self

    def close(self):
        if self._fh:
            self._fh.close()

    def _open(self):
        logger.log(f"Reading {self.dataset.name} for epoch {self.epoch}")
        # Open temporary file which will contain shuffled version of `cat self.files`
        fh = TemporaryFile(mode='w+', encoding='utf-8', dir=self.tmpdir)

        # Shuffle data to the temporary file.
        # TODO: With the reimplementation of shuffle.py, it is technically
        # feasible to just write to a named pipe (or even stdout) instead of
        # a temporary file, and let the trainer read directly from that. Not 
        # sure if that has any performance or stability benefits/drawbacks.
        subprocess.check_call([sys.executable,
            '-m', 'opustrainer.shuffle',
            *(['--temporary-directory', self.tmpdir] if self.tmpdir else []),
            *([] if self.shuffle else ['--no-shuffle']),
            str(self.seed),
            f'/dev/fd/{fh.fileno()}',
            *self.dataset.files
        ], pass_fds=(fh.fileno(),))

        # Replace open file handle with this new file
        self._fh = cast(TextIO, fh) # TODO: Not sure why TemporaryFile is an
                                    # IO[str] according to typing, but seems
                                    # to implement TextIO.
        self._fh.seek(0)
        self.line = 0

        # Buffer the first line, also asserting that we're not reading an empty file.
        try:
            self._read_line()
        except StopIteration:
            raise RuntimeError('reading from empty shuffled file')

    def _read_line(self) -> None:
        try:
            # Try to find the next non-empty line
            while True:
                self._next_line = self._fh.readline() # type: ignore # _fh can't be none.

                # Empty return, not even a line ending, means EOF
                if self._next_line == '':
                    raise StopIteration

                # We can't call the function inside the string format prior to python 3.12
                # so we will have it here instead so that we can refer to it when logging if necessary.
                original_line: str = self._next_line.rstrip('\r\n')

                # Assert that the line is well formed, meaning non of the fields is the empty string
                # If not, try to get a new line from the corpus
                fields: List[str] = original_line.split('\t')
                if any(field == '' for field in fields):
                    logger.log_once(f"[Trainer] Empty field in {self.dataset.name} line: \"{original_line}\", skipping...", loglevel="WARNING")
                    continue

                # Try to see if we have the right number of fields and remove lines
                # that don't have all the fields or remove extra fields.
                if self.num_fields is not None:
                    if len(fields) == self.num_fields:
                        return
                    elif len(fields) > self.num_fields:
                        self._next_line = '\t'.join(fields[:self.num_fields]) + '\n'
                        return
                    elif len(fields) < self.num_fields:
                        logger.log_once(f"[Trainer] Expected {self.num_fields} fields in {self.dataset.name} line: \"{original_line}\" but only got {len(fields)}, skipping...", loglevel="WARNING")
                        continue

                return
        except StopIteration:
            self._fh.close() # type: ignore # _fh can't be none.
            self.seed += 1
            self.epoch += 1

    def __iter__(self):
        return self

    def __next__(self) -> str:
        if not self._fh or self._fh.closed:
            self._open() # TODO: do we want to do this lazy? Yes, restore()
                         # might be called twice right now and shuffling is
                         # expensive.

        assert self._fh is not None
        line = self._next_line
        self.line += 1

        # Read next line into memory and test we're not at EOF
        self._read_line()

        return line


@dataclass(frozen=True)
class ShuffledFile:
    seed: int
    proc: subprocess.Popen
    file: TextIO


class AsyncDatasetReader(DatasetReader):
    _pending: Optional[ShuffledFile]

    def __init__(self, *args, **kwargs):
        self._pending = None
        super().__init__(*args, **kwargs)

    def _open_async(self, seed:int):
        # Open temporary file which will contain shuffled version of `cat self.files`
        fh = TemporaryFile(mode='w+', encoding='utf-8', dir=self.tmpdir)

        self._pending = ShuffledFile(
            seed=seed,
            file=cast(TextIO, fh),
            proc=subprocess.Popen([sys.executable,
                '-m', 'opustrainer.shuffle',
                *(['--temporary-directory', self.tmpdir] if self.tmpdir else []),
                *([] if self.shuffle else ['--no-shuffle']),
                str(seed),
                f'/dev/fd/{fh.fileno()}',
                *self.dataset.files
            ], pass_fds=(fh.fileno(),))
        )

    def _kill_async(self):
        if self._pending is None:
            return

        self._pending.proc.kill()
        self._pending.proc.wait()
        self._pending.file.close()
        self._pending = None

    def _open(self):
        logger.log(f"Reading {self.dataset.name} for epoch {self.epoch}")

        # First time self._pending is None, but all subsequent calls to _open
        # should have self._pending be set.
        if self._pending is None:
            self._open_async(self.seed)

        # Assume shuffling has started
        assert self._pending is not None
        assert self._pending.seed == self.seed

        # Wait for that to finish (hopefully it already has since it was likely
        # started last iteration)
        self._pending.proc.wait()
        assert self._pending.proc.returncode == 0

        # Swap out the current _fh for the newly prepared one
        assert self._fh is None or self._fh.closed
        self._fh = self._pending.file
        self._pending = None

        # Make sure we start reading from the start again
        self._fh.seek(0)
        self.line = 0

        # Buffer the first line, also asserting that we're not reading an empty file.
        try:
            self._read_line()
        except StopIteration:
            raise RuntimeError('reading from empty shuffled file')

        # Start shuffling next
        self._open_async(self.seed + 1)

    def restore(self, state:DatasetState) -> 'AsyncDatasetReader':
        # Note: super().restore() will call close(), which will stop any
        # running shuffling that is probably no longer relevant.
        # TODO: Once PEP 673 is available, we can remove this overload entirely.
        return cast('AsyncDatasetReader', super().restore(state))

    def close(self):
        self._kill_async()
        super().close()


class StateLoader:
    """Tool to read and write TrainerState objects to yaml. Uses unsafe yaml
    because `random.getstate()` basically returns a blob, and it is very
    particular about the data types of that structure. So we use the yaml loader
    that encodes the python data type as well (i.e. tuple vs list).
    """
    def load(self, fh:TextIO) -> TrainerState:
        ymldata = yaml.load(fh, Loader=yaml.Loader)
        if not isinstance(ymldata, dict):
            raise ValueError(f'Empty state file: {fh.name}')
        return TrainerState(
            stage=ymldata['stage'],
            random_state=ymldata['random_state'],
            epoch_tracker_state=ymldata['epoch_tracker_state'],
            datasets={
                dataset_name: DatasetState(int(seed), int(line), int(epoch))
                for dataset_name, [seed, line, epoch] in ymldata['datasets'].items()
            }
        )

    def dump(self, state:TrainerState, fh:TextIO) -> None:
        yaml.dump({
            'stage': state.stage,
            'random_state': state.random_state,
            'epoch_tracker_state': state.epoch_tracker_state,
            'datasets': {
                dataset_name: [state.seed, state.line, state.epoch] #TODO: why a tuple, why not a dict? Isn't a dict more forward compatible?
                for dataset_name, state in state.datasets.items()
            }
        }, fh, allow_unicode=True, sort_keys=False) #TODO: is safe_dump not sufficient?


class CurriculumLoaderError(ValueError):
    """Exception raised when the yaml data contains an invalid curriculum"""
    pass


def flatten(iterable):
    """Iterates recursively (depth-first) trough all items in a list with
    possibly nested lists"""
    for item in iterable:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item


class CurriculumV1Loader:
    def load(self, ymldata:dict, *, basepath:str='./') -> Curriculum:
        datasets = self._load_datasets(ymldata, basepath)
        stages_order = self._load_stage_order(ymldata)
        return Curriculum(
            seed=int(ymldata['seed']),
            datasets=datasets,
            stages_order=stages_order,
            stages=self._load_stages(ymldata, basepath, stages_order, datasets),
            modifiers=self._load_modifiers(ymldata, basepath),
            num_fields=int(ymldata['num_fields']) if 'num_fields' in ymldata else None
        )

    def _load_datasets(self, ymldata:dict, basepath:str) -> Dict[str,Dataset]:
        """Reads
        ```yml
        datasets:
          clean: path/to/clean.gz
        ```
        """
        return {
            name: Dataset(name, [os.path.join(basepath, filepath)])
            for name, filepath in ymldata['datasets'].items()
        }

    def _load_stage_order(self, ymldata:dict) -> List[str]:
        """Reads
        ```yaml
        stages:
          - stage1
          - stage2
        ```
        """
        return list(ymldata['stages'])

    def _load_stages(self, ymldata:dict, basepath:str, stages_order:List[str], datasets:Dict[str,Dataset]) -> Dict[str,Stage]:
        """Reads:
        ```yaml
        stagename:
          - dataset1 frac
          - dataset2 frac
          - until dataset3 epochs
        ```
        or the more verbose version
        ```yaml
        stagename:
          mix:
            - dataset1 frac
            - dataset2 frac
            - until dataset3 epochs
          modifiers:
            - Modifier: freq
        ```
        """
        return {
            stage_name: self._load_stage(ymldata, basepath, stage_name, datasets, int(ymldata['seed']))
            for stage_name in stages_order
        }

    def _load_stage(self, ymldata:dict, basepath:str, stage_name:str, available_datasets:Dict[str,Dataset], seed:int) -> Stage:
        datasets: List[Tuple[Dataset, float]] = []

        if isinstance(ymldata[stage_name], list):
            mix = ymldata[stage_name]
        else:
            mix = ymldata[stage_name].get('mix', [])

        if len(mix) == 0:
            raise CurriculumLoaderError(f"the dataset mix of stage '{stage_name}' is empty or missing its until clause")
        if not mix[-1].startswith('until '):
            raise CurriculumLoaderError(f"the last entry in {stage_name}'s dataset mix is not the until clause")

        for line in mix[:-1]:
            dataset_name, weight = line.split()
            try:
                datasets.append((available_datasets[dataset_name], float(weight)))
            except KeyError:
                raise CurriculumLoaderError(f"stage '{stage_name}' mentions unknown dataset '{dataset_name}'")
            except ValueError:
                raise CurriculumLoaderError(f"could not convert the weight '{weight}' of stage '{stage_name}' dataset '{dataset_name}' to float")

        try:
            _, until_dataset_name, max_epochs = mix[-1].split()
            until_epoch = int(max_epochs) if max_epochs != 'inf' else None
        except ValueError:
            raise CurriculumLoaderError(f"could not parse last line as `until <dataset_name> <number|'inf'>`: {mix[-1]}")
        if until_dataset_name not in {dataset.name for dataset, weight in datasets if weight > 0.0}:
            raise CurriculumLoaderError(f"until clause of stage '{stage_name}' watches dataset '{until_dataset_name}' but that dataset is not read during this stage")

        try:
            return Stage(
                name=stage_name,
                datasets=datasets,
                until_dataset=until_dataset_name,
                until_epoch=until_epoch,
                modifiers=self._load_modifiers(ymldata[stage_name], basepath) if isinstance(ymldata[stage_name], dict) and 'modifiers' in ymldata[stage_name] else None
            )
        except Exception as exc:
            raise CurriculumLoaderError(f"could not complete the parse of stage '{stage_name}': {exc!s}") from exc

    def _load_modifiers(self, ymldata:dict, basepath:str) -> List[Modifier]:
        """Reads
        ```yml
        modifiers:
          - UpperCase: 0.05
          - TitleCase: 0.05
          - Tags: 0.02
            num_tags: 6
            custom_detok_src: null
            custom_detok_trg: zh
        ```
        """
        modifiers = [
            self._load_modifier(modifier_entry, basepath)
            for modifier_entry in flatten(ymldata.get('modifiers', []))
        ]

        for modifier in modifiers:
            modifier.validate(modifiers)

        return modifiers

    def _load_modifier(self, modifier_entry:Dict[str, Any], basepath:str) -> Modifier:
        (name, probability), *config_pairs = modifier_entry.items()
        settings = {
            **dict(config_pairs),
            'probability': float(probability)
        }
        try:
            modifier = MODIFIERS[name]
        except KeyError:
            raise CurriculumLoaderError(f"unknown modifier '{name}'")
        try:
            args = self._match_modifier_signature(modifier, settings, basepath)
            return modifier(**args)
        except Exception as exc:
            raise CurriculumLoaderError(f"could not initialize modifier '{name}': {exc!s}") from exc

    def _match_modifier_signature(self, fn:Type[Modifier], settings:Dict[str, Any], basepath:str) -> Dict[str,Any]:
        """Casts `settings` values into the types expected by `fn`'s constructor. This mainly
        exists so that a modifier can define a `arg:Path` argument that is then correctly
        interpreted as a relative path."""
        hints = get_type_hints(fn.__init__)
        args = {}

        for name, value in settings.items():
            if name in hints:
                try:
                    value = self._dynamic_cast_parameter(hints[name], value, basepath)
                except Exception as exc:
                    raise CurriculumLoaderError(f"could not convert '{name}' to the correct type: {exc!s}") from exc
            args[name] = value
        
        return args

    T = TypeVar('T')

    def _dynamic_cast_parameter(self, typedef:Type[T], value:Any, basepath:str) -> Union[T, None, Path]:
        """Casts `value` into a type that matches `typedef`. Can deal with `Union`."""
        if get_origin(typedef) == Union:
            attempt_errors = ''
            for subtype in get_args(typedef):
                try:
                    return self._dynamic_cast_parameter(subtype, value, basepath)
                except Exception as err:
                    attempt_errors += f'could not cast to {subtype!r}: {err!s}\n'
            raise ValueError(attempt_errors)
        elif typedef == Path:
            return Path(basepath).joinpath(value)
        elif typedef == type(None) and value is None:
            return None
        else:
            return typedef(value)


class CurriculumLoader:
    """Reads curriculum yaml files. Wrapper that decides which reader to use
    based on a version number that may be in there.
    """

    IMPLEMENTATIONS={
        '1': CurriculumV1Loader,
    }

    def load(self, fh:Union[TextIO,str,dict], **kwargs) -> Curriculum:
        if isinstance(fh, dict):
            ymldata = fh
        else:
            ymldata = yaml.safe_load(fh)

        impl = self.IMPLEMENTATIONS[str(ymldata.get('version', '1'))]()
        return impl.load(ymldata, **kwargs)


class EpochTracker:
    """Utility to track how many epochs the reader has progressed since the
    tracker started tracking."""
    def __init__(self, reader:DatasetReader):
        self.reader = reader
        self.epoch_offset = reader.epoch
        self.line_offset = reader.line

    @property
    def epoch(self):
        epoch = self.reader.epoch - self.epoch_offset

        # ... but if the reader is behind on where it was, it hasn't completed
        # a full epoch yet.
        if self.reader.line < self.line_offset:
            epoch -= 1
        return epoch

    def restore(self, state:EpochTrackerState) -> 'EpochTracker':
        self.epoch_offset = state.epoch
        self.line_offset = state.line
        return self

    def state(self) -> EpochTrackerState:
        return EpochTrackerState(self.epoch_offset, self.line_offset)


class Trainer:
    """Writes lines to a trainer program according to the curriculum."""
    curriculum: Curriculum
    readers: Dict[str, DatasetReader]
    stage: Optional[Stage]
    epoch_tracker: EpochTracker

    # Path to write temporary shuffled files to
    tmpdir:Optional[str]
    # For debugging purposes, whether to shuffle or not
    shuffle:bool

    # Reader class to use (I.e. DatasetReader or AsyncDatasetReader)
    _reader_impl: Type[DatasetReader]

    def __init__(self, curriculum:Curriculum, *, reader:Type[DatasetReader] = DatasetReader, \
                 tmpdir:Optional[str]=None, shuffle:bool=True):
        self.curriculum = curriculum
        self.tmpdir = tmpdir
        self.shuffle = shuffle
        self._reader_impl = reader
        random.seed(self.curriculum.seed)
        first_stage_name = self.curriculum.stages_order[0]

        #TODO: make sure this doesn't do too much work in case we call
        # restore() manually anyway.
        self.restore(TrainerState(
            stage=first_stage_name,
            random_state=random.getstate(),
            epoch_tracker_state=EpochTrackerState(0, 0),
            datasets={
                dataset.name: DatasetState(seed=curriculum.seed, line=0, epoch=0)
                for dataset in curriculum.datasets.values()
            }
        ))

    def restore(self, state:TrainerState):
        random.setstate(state.random_state)
        self.stage = self.curriculum.stages[state.stage]
        self.readers = {
            dataset.name: self._reader_impl(dataset, self.curriculum.seed,
                tmpdir=self.tmpdir,
                shuffle=self.shuffle,
                num_fields=self.curriculum.num_fields
            ).restore(state.datasets[dataset.name])
            for dataset in self.curriculum.datasets.values()
        }
        self.epoch_tracker = EpochTracker(self.readers[self.stage.until_dataset]).restore(state.epoch_tracker_state)

    def state(self) -> TrainerState:
        return TrainerState(
            stage=self.stage.name if self.stage is not None else '',
            random_state=random.getstate(),
            epoch_tracker_state=self.epoch_tracker.state(),
            datasets={
                name: reader.state()
                for name, reader in self.readers.items()
            }
        )

    def close(self):
        for reader in self.readers.values():
            reader.close()
        self.readers = {}

    def next_stage(self) -> Optional[Stage]:
        """Move to the next stage. Will return this next stage or None if there is no next stage."""
        if self.stage is None:
            return None

        self.stage = self.curriculum.next_stage(self.stage)

        # If there is a next stage, also reset the epoch tracker to track the
        # `until` clause of that new stage.
        # TODO: when self.stage is None, should we delete the EpochTracker?
        if self.stage is not None:
            self.epoch_tracker = EpochTracker(self.readers[self.stage.until_dataset])

        return self.stage

    def log_stage_information(self):
        if not self.stage:
            return

        logger.log(f'Starting stage: "{self.stage.name}"')
        if self.stage.until_epoch:
            logger.log(
                f'Running until dataset "{self.stage.until_dataset}" for {self.stage.until_epoch} epoch(s)'
            )
        else:
            logger.log(f'Running until dataset "{self.stage.until_dataset}"')

        if self.curriculum.modifiers:
            logger.log("Using curriculum modifiers:")
            for modifier in self.curriculum.modifiers:
                name = type(modifier).__name__
                logger.log(f" - {name} ({modifier.probability})")

        if self.stage.modifiers:
            logger.log("Using stage modifiers:")
            for modifier in self.stage.modifiers:
                name = type(modifier).__name__
                logger.log(f" - {name} ({modifier.probability})")

    def run(self, *, batch_size:int=100, chunk_size:int=16, processes:int=0) -> Iterable[List[str]]:
        """Yield batches, moving through the stages of training as datasets are consumed."""
        while self.stage is not None:
            self.log_stage_information()

            # Stage level modifiers take precedence over global modifiers,
            # but you can combine them yourself using YAML references.
            if self.stage.modifiers is not None:
                modifiers = self.stage.modifiers
            else:
                modifiers = self.curriculum.modifiers

            with make_modifier_pool(modifiers, processes) as pool:
                while self.stage.until_epoch is None or self.epoch_tracker.epoch < self.stage.until_epoch:
                    batch: List[str] = []

                    # Read from each dataset according to its weight in this stage
                    # (They will reshuffle and repeat if necessary)
                    for dataset, weight in self.stage.datasets:
                        batch.extend(
                            line.rstrip('\r\n') for line in
                            islice(self.readers[dataset.name], 0, int(batch_size * weight))
                        )

                    # Apply any modifiers to random lines in the batch, or sentence
                    # (Multiple modifiers can be applied to the same line)
                    batch = pool.map(batch, chunk_size)

                    if self.shuffle:
                        random.shuffle(batch)

                    # Tell anyone whose listening that something interesting happened
                    # TODO: Yield something useful, e.g. progress.
                    yield [line + '\n' for line in batch]

            # Move onto next stage. May be `None`, which would end this generator
            self.next_stage()
        logger.log("Finished the last stage")

class StateTracker:
    """Wraps around the trainer.run() call to restore and dump state its."""
    path: str
    loader: StateLoader
    dump: bool
    restore: bool

    def __init__(self, path:str, *, loader:StateLoader=StateLoader(), restore:bool=True, dump:bool=True, timeout=60):
        """
        Parameters
        --–-------
        path : str
            Path to state file
        loader : type, optional
            Loader class for encoding/decoding the state file
        restore : bool, optional
            Whether to restore the state if the state file currently exists (default is True)
        dump : bool, optional
            Whether to dump the state to the file after training (default is True)
        timeout : int, optional
            Minimum number of seconds between state dumps
        """
        self.path = path
        self.loader = loader
        self.dump = dump
        self.restore = restore
        self.timeout = timeout
        self._last_dump = 0

    def _restore(self, trainer:Trainer):
        with open(self.path, 'r', encoding='utf-8') as fh:
            return trainer.restore(self.loader.load(fh))

    def _dump(self, trainer:Trainer):
        new_statefile = f"{self.path}.new"
        with open(new_statefile, 'w', encoding='utf-8') as fh:
            self.loader.dump(trainer.state(), fh)
        os.rename(new_statefile, self.path)
        self._last_dump = time.monotonic()

    def run(self, trainer:Trainer, *args, **kwargs):
        if self.restore and os.path.exists(self.path):
            self._restore(trainer)

        try:
            for batch in trainer.run(*args, **kwargs):
                # TODO: Replace this with something that listens to Marian, and
                # writes the state to disk after marian performed validation.
                if self.dump and time.monotonic() - self._last_dump > self.timeout:
                    self._dump(trainer)
                yield batch
        finally:
            # Dump on clean exit as well as on exception.
            if self.dump:
                self._dump(trainer)


def print_state(state:TrainerState) -> None:
    logger.log(f"At stage {state.stage}")
    for name, reader in state.datasets.items():
        logger.log(f"Dataset {name}: overall epochs {reader.epoch: 3d}.{reader.line:010d}")


def parse_args(args:List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feeds marian tsv data for training.", allow_abbrev=False)
    parser.add_argument("--config", '-c', required=True, type=str, help='YML configuration input.')
    parser.add_argument("--state", '-s', type=str, help='YML state file, defaults to ${CONFIG}.state.')
    parser.add_argument("--sync", action="store_true", help="Do not shuffle async")
    parser.add_argument("--temporary-directory", '-T', default=None, type=str, help='Temporary dir, used for shuffling and tracking state')
    parser.add_argument("--do-not-resume", '-d', action="store_true", help='Do not resume from the previous training state')
    parser.add_argument("--no-shuffle", '-n', action="store_false", help='Do not shuffle, for debugging', dest="shuffle")
    parser.add_argument("--batch-size", '-b', type=int, default=100, help='Batch size')
    parser.add_argument("--chunk-size", '-B', type=int, default=16, help='Chunk size of batches fed to modifiers')
    parser.add_argument("--workers", '-j', type=int, default=os.cpu_count() or 1, help='Number of workers')
    parser.add_argument("--log-level", type=str, default="INFO", help="Set log level. Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL. Default is INFO")
    parser.add_argument("--log-file", '-l', type=str, default=None, help="Target location for logging. Always logs to stderr and optionally to a file.")
    parser.add_argument("trainer", type=str, nargs=argparse.REMAINDER, help="Trainer program that gets fed the input. If empty it is read from config.")
    return parser.parse_args(args)


def main() -> None:
    args = parse_args(sys.argv[1:])

    logger.setup_logger(args.log_file, args.log_level)

    with open(args.config, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)

    curriculum = CurriculumLoader().load(config, basepath=os.path.dirname(args.config))

    # Quick cheap check that all files exist before we begin training
    for dataset in curriculum.datasets.values():
        missing_files = {file for file in dataset.files if not os.path.exists(file)}
        if missing_files:
            raise ValueError(f"Dataset '{dataset.name}' is missing files: {missing_files}")

    trainer = Trainer(curriculum,
        reader=DatasetReader if args.sync else AsyncDatasetReader,
        tmpdir=args.temporary_directory,
        shuffle=args.shuffle)

    state_tracker = StateTracker(args.state or f'{args.config}.state', restore=not args.do_not_resume)

    # Make trainer listen to `kill -SIGUSR1 $PID` to print dataset progress
    signal.signal(signal.SIGUSR1, lambda signum, handler: print_state(trainer.state()))

    model_trainer = subprocess.Popen(
        args.trainer or shlex.split(config['trainer']),
        stdin=subprocess.PIPE,
        encoding="utf-8",
        preexec_fn=ignore_sigint) # ignore_sigint makes marian ignore Ctrl-C. We'll stop it from here.

    assert model_trainer.stdin is not None

    # TODO: This logic looks complicated, should be able to do this simpler. Three scenarios:
    #   1. ctrl-c is pressed and trainer is told this is the end of the training data
    #   2. ctrl-c is pressed and trainer has much training data in its buffers, ctrl-c needs to be
    #      pressed again to tell trainer to really terminate. Just closing its stdin and waiting for
    #      it to notice takes too long
    #   3. trainer decides it has read enough and will train no longer. This is the BrokenPipeError
    #      scenario. We don't need to deal with multiple levels of terminating the trainer because
    #      the trainer is already dead at this point.
    try:
        try:
            for batch in state_tracker.run(trainer, batch_size=args.batch_size, chunk_size=args.chunk_size, processes=args.workers):
                model_trainer.stdin.writelines(batch)
        except KeyboardInterrupt:
            logger.log("Ctrl-c pressed, stopping training")

        # Levels of waiting for the trainer. This is reached either because we ran out of batches
        # or because ctrl-c was pressed. Pressing ctrl-c more advances to next level of aggressiveness.
        for stage in ['exit', 'terminate', 'kill']:
            try:
                if stage == 'exit':
                    model_trainer.stdin.close()
                elif stage == 'terminate':
                    model_trainer.terminate()
                else:
                    model_trainer.kill()
                
                logger.log(f"waiting for trainer to {stage}. Press ctrl-c to be more aggressive")
                sys.exit(model_trainer.wait()) # blocking
            except KeyboardInterrupt:
                continue
    except BrokenPipeError:
        # BrokenPipeError is thrown by writelines() or close() and indicates that the child trainer
        # process is no more. We can safely retrieve its return code and exit with that, it should
        # not block at this point.
        logger.log("trainer stopped reading input")
        sys.exit(model_trainer.wait())


if __name__ == '__main__':
    main()
