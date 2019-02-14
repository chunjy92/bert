#! /usr/vectorin/python3
# -*- coding: utf-8 -*-
import os

__author__ = 'Jayeol Chun'

# as empirically observed in train data
SENSES = [
  'Temporal.Asynchronous.Precedence',
  'Temporal.Asynchronous.Succession',
  'Temporal.Synchrony',
  'Contingency.Cause.Reason',
  'Contingency.Cause.Result',
  'Contingency.Condition',
  'Comparison.Contrast',
  'Comparison.Concession',
  'Expansion.Conjunction',
  'Expansion.Instantiation',
  'Expansion.Restatement',
  'Expansion.Alternative',
  'Expansion.Alternative.Chosen alternative',
  'Expansion.Exception',
  'EntRel',
  'Comparison',
  'Temporal',
  'Expansion',
  'Contingency.Cause',
  'Contingency',
  'Temporal.Asynchronous'
]


def to_level(sense, level=2):
  s_split = sense.split(".")
  s_join = ".".join(s_split[:level])
  return s_join

def get_senses_with_level(level=2, sense_path=None):
  """ Collect unique senses from PDTB

  Args:
    level: Sense level
    export: whether to save collected senses

  Returns:
    list of unique senses with appropriate level
  """
  ret = []

  if level==3:
    ret = SENSES
  elif level==2:
    if sense_path is not None and os.path.exists(sense_path):
      ret = []
      with open(sense_path, 'r') as f:
        for line in f.readlines():
          line = to_level(line.strip())
          ret.append(line)
      return ret

    ret = []
    for sense in SENSES:
      sense = to_level(sense, level)
      if sense not in ret:
        ret.append(sense)

  elif level==1:
    raise ValueError("1 Degree not accepted")

  if sense_path is not None and not os.path.exists(sense_path):
    with open(sense_path, "w") as f:
      for sense in ret:
        f.write(sense)
        f.write("\n")

  return ret
