# OpusTrainer
The purpose of the trainer is to provide the user with a flexible way of scheduling various sources of input data, as well as augment the training data with tittle casing, all caps, etc. This is particularly useful when you have multiple data sources and you want to pretrain the model first on backtranslated data, gradually add other sources of data, and finally fine tune, all in one go.

Alternatively, this tool is particularly suited to training multilingual models, as it provides an easy way to define the desired mixture of datasets from different language sources.

## Installation
You've got two options: Install directly from PyPI:

```sh
pip install opustrainer
```

or clone this repository, and install it in editable mode so you can change the source, but still use all the commands:

```sh
git clone git@github.com:hplt-project/opustrainer.git
cd opustrainer
pip install -e .
```

## Usage
```bash
% ./trainer.py --help
usage: trainer.py [-h] --config CONFIG [--temporary-directory TEMPORARY_DIR] [--state STATE_FILE] [--do-not-resume] [--sync] [trainer-command [arguments]]

Feeds marian tsv data for training.

options:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        YML configuration input.
  --temporary-directory TEMPORARY_DIR, -t TEMPORARY_DIR
                        Temporary dir, used for shuffling and tracking state
  --state STATE_FILE    Path to trainer state file which stores how much of
                        each dataset has been read. Defaults to ${CONFIG}.state
  --sync                Do not shuffle in the background
  --do-not-resume, -d   Do not resume from the previous training state
  --no-shuffle, -n      Do not shuffle, for debugging
```
Once you fix the paths in the configuration file, `train_config.yml` you can run a test case by doing:
```bash
./trainer.py -c train_config.yml /path/to/marian -c marian_config --any --other --flags
```
You can check resulting mixed file in `/tmp/test`. If your neural network trainer doesn't support training from `stdin`, you can use this tool to generate a training dataset and then disable data reordering or shuffling at your trainer implementation, as your training input should be balanced.

At the start of the training all datasets are shuffled. Each time a dataset's end is reached, it is re-shuffled. Shuffling [in the system temp directory](https://docs.python.org/3.11/library/tempfile.html#tempfile.gettempdir) but can be repositioned using `--temporary-directory` or the `TMPDIR` environment variable. By default, the training state is kept in the same place as the configuration file. If training is interrupted, re-running the trainer should resume from where it was (depending on how much your neural network trainer has buffered, that part will be skipped).


## Configuration file
Define your training process via a configuration file. You define the datasets on top, the stages and then for each stage a mixing criteria and a stage termination criteria. An example configuration file is provided below. The path to the `trainer` is a path to any neural network trainer that supports having stdin as training input format.
```yml
# Datasets are already TSV files. We support reading gzip'd files, as well as multiple dataset file per name
datasets:
  clean: test/data/clean
  medium: test/data/medium
  dirty: test/data/dirty

stages:
  - start
  - mid
  - end

start:
  - clean 0.8
  - medium 0.2
  - dirty 0
  - until clean 2 # Until two epochs of clean

mid:
  - clean 0.6
  - medium 0.3
  - dirty 0.1
  - until medium 1

end:
  - clean 0.4
  - medium 0.3
  - dirty 0.3
  - until dirty 5 # use `inf` to mean until forever

modifiers:
- UpperCase: 0.05 # Apply uppercase randomly to 5% of sentences. See below
- TitleCase: 0.05

seed: 1111
trainer: /path/to/trainer/run.py
```

### Extended stage configuration
If you want to change which modifiers are used for a specific stage, you can the extended stage configuration format. If a `modifiers` is mentioned here, it will override the curriculum-wide defined `modifiers` for just this stage.

In the extended format, the list of datasets is defined in the `mix` key. You can optionally add a `modifiers` key. For example: 

```yaml
start:
  mix:
  - clean 0.8
  - medium 0.2
  - dirty 0
  - until clean 2 # Until two epochs of clean
  modifiers:
    - UpperCase: 0.05
    - TitleCase: 0.05
```

Note that you can use YAML references if you wish to extensively combine global and local modifiers.

### Modifiers
Modifiers are randomly applied to the sentences that go into the trainer. Each modifier has a probability associated with it that is the chance that a sentence is modified by the modifier. E.g. a modifier with a probability of 0.05 will affect about 1 in every 20 sentences.

Modifiers are applied one after another, in the order you defined them, all with their own probability regardless of the modifiers that got applied before it. E.g. if you have the following configuration:

```yaml
modifiers:
- UpperCase: 0.05
- TitleCase: 0.05
```

This means that 1 in 20 sentences will be uppercased, and 1 in 20 will be titlecased. And effectively `0.05 * 0.05` so 1 in 400 will first be uppercased and then titlecased.

#### UpperCase
Turns the entire source and target sentence to upper case, e.g. 'heLLo' becomes 'HELLO'.

```yaml
modifiers:
  - UpperCase: 0.05
```

#### TitleCase
Makes the first letter of every word uppercase, and the rest lowercase. Words are split by spaces. E.g. 'heLLo' becomes 'Hello'.

```yaml
modifiers:
  - TitleCase: 0.05
```

#### Typos
Introduce typos in the source side of the sentence pair.

The probability of the modifier itself is the chance a sentence is affected. The probabilities of each of the types of typos describes the chance a word is affected. Each type of typo occurs at most once in a sentence.

You can specify a probability for each modifier individually. If any of the typo classes is omitted, it has a probability of 0. Alternatively, you can omit all typo classes. Then all of them will have a default 10% probability.

```yaml
modifiers:
- Typos: 0.05
  char_swap:     0.1 # Swaps two random consecutive word characters in the string.
  missing_char:  0.1 # Skips a random word character in the string.
  extra_char:    0.1 # Adds an extra, keyboard-neighbor, letter next to a random word character.
  nearby_char:   0.1 # Replaces a random word character with keyboard-neighbor letter.
  similar_char:  0.1 # Replaces a random word character with another visually similar character.
  skipped_space: 0.1 # Skips a random space from the string.
  random_space:  0.1 # Adds a random space in the string.
  repeated_char: 0.1 # Repeats a random word character.
  unichar:       0.1 # Replaces a random consecutive repeated letter with a single letter. 
````

#### Tags
Adds a placeholder tag to the source sentence that can be used by the model to hint how it should translate that word. The word to hint is chosen at random from the target sentence. Only words with a 1-to-1 mapping between source and target are considered.

This modifier needs a third column in the training data with per-word alignment information.

```yaml
- Tags: 0.05
  custom_detok_src: null
  custom_detok_trg: zh
  template: "__source__ {src} __target__ {trg} __done__"
```

All options are optional.

You can specify custom detokenizer languages using `custom_detok_src` and `custom_detok_trg` if the dataset you're reading from has been tokenized by the Moses tokenizer. This can be helpful to do for languages that do not use spaces to delimit words. The default tokenisation strategy is splitting/joining by spaces.

The format for telling the translation model the intention to translate a word in a certain way can be controlled by `template`. Here `{src}` and `{trg}` are replaced by the selected words from the source and target side of the sentence pair.

#### Prefix
Prepends a random subsection of the target sentence before the source sentence. 

This is useful for teaching the model to force decode a specific string if the user is absolutely certain it has to appear in the output. For example `I like pie. Me gustan los pasteles.` becomes `__start__ los pasteles __end__ I like pie. Me gustan los pasteles.`

Note: The Prefix modifier must always be used as the last modifier, but ideally never used together with "Tags".

```yaml
modifiers:
 - Prefix: 0.5
   min_words: 2
   max_words: 5
   template: "__start__ {trg} __end__ "
```

### Tags
In order to train models supporting vocabulary "hints" we provide a tags system where we laverage word alignment information to provide "hints" for the translation model to what it should produce on the Target side. The work is very similar to [Dinu et al. 2019](https://aclanthology.org/P19-1294/). An example. Given an alignment augmented tsv training line:
```I like pies! \t Ich mag Kuchen! \t 0-0 1-1 2-2```
The machine translation traing system would see something like:
```I like <tag0> mag <tag0> pies! \t Ich mag Kuchen!```
Where the numeric ID of the tag is drawn from a random distribution from the total number of tags. The probability asigned to the `Tags` modifier determines how likely is it for a tag augmentation to appear on any given word.

Finally, if your corpus is augmented with alignment info, but you don't want to use any tags, just set the probability of this modifier to 0.

### Generating vocabulary and tags before training
In the future, this will be handled by a training Pipeline, but until then here's the basic scripts used

For producing alignment augmented corpus use this script:
```bash
#!/bin/bash -v

# Usage: ./align_corpus.sh source_corpus target_corpus src trg

# install fast align
mkdir -p bin

# download and compile fast_align
if [ ! -e bin/fast_align ]; then
    git clone https://github.com/clab/fast_align
    mkdir -p fast_align/build
    cd fast_align/build
    cmake ..
    make -j4
    cp fast_align atools ../../bin
    cd ../../
fi

# Prepare the corpus for fast align
test -s $2/corpus.tmp.${3}-${4}.falign ||  cat $1 | sed 's/\t/ ||| /' > $2/corpus.tmp.${3}-${4}.falign

# Align it
test -s $2/align.${3}-${4}.s2t  || bin/fast_align -vod  -i $2/corpus.tmp.${3}-${4}.falign > $2/align.${3}-${4}.s2t
test -s $2/align.${3}-${4}.t2s  || bin/fast_align -vodr -i $2/corpus.tmp.${3}-${4}.falign > $2/align.${3}-${4}.t2s

test -s $2/corpus.${3}-${4}.aln || bin/atools -i $2/align.${3}-${4}.s2t -j $2/align.${3}-${4}.t2s -c grow-diag-final-and > $2/corpus.${3}-${4}.aln
```

For creating vocabulary with tags support, use this script:
```bash
#!/usr/bin/env bash
#Usage ./vocab.sh en de path-to-corpora char-cov vocab_size

char_cov=${4:-'0.9995'} # Default char coverage
vocab_size=${5:-'32000'} # Default vocab size
# Set up some constants

# Language pairs
src=$1
trg=$2
prefix="--model_prefix=model.${src}-${trg}"

# Placeholders array
placeholders="--user_defined_symbols=__source__,__target__,__done__,__start__,__end__"

# Character coverage. CJK is recommended to have 0.9995, vocab languages proabbly you want 1.
char_cov="--character_coverage=${char_cov}"

# First clone and compile SPM
spm_exec="sentencepiece/build/src/spm_train"
if [ ! -e ${spm_exec} ]; then
    git clone https://github.com/google/sentencepiece.git
    cd sentencepiece
    mkdir build
    cd build
    cmake ..
    make -j4
    cd ..
    cd ..
    if [ ! -e ${spm_exec} ]; then
        echo "Failed to compile sentencepiece"
        exit 1
    fi
fi

$spm_exec --bos_id=-1 --eos_id=0 --unk_id=1 ${placeholders} ${char_cov} ${prefix} --vocab_size=${vocab_size} --input=${3} --input_sentence_size=20000000 --byte_fallback #--input_format=tsv seems broken
```

## Future work

- Terminology support (using a dictionary), where augmentation happens not by using alignment scores but by taking values from a dictionary.
- A one click run training

# Acknowledgements

This project has received funding from the European Union’s Horizon Europe research and innovation programme under grant agreement No 101070350 and from UK Research and Innovation (UKRI) under the UK government’s Horizon Europe funding guarantee [grant number 10052546]
