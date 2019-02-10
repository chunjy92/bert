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


def to_degree(sense, degree=2):
  s_split = sense.split(".")
  s_join = ".".join(s_split[:degree])
  return s_join

def get_senses_with_degree(degree=2, export=True):
  """"""
  out_p = "./senses.txt"
  if degree==3:
    # return SENSES
    ret = SENSES
  elif degree==2:
    if os.path.exists(out_p):
      ret = []
      with open(out_p, 'r') as f:
        for line in f.readlines():
          line = to_degree(line.strip())
          ret.append(line)
      return ret

    ret = []
    for sense in SENSES:
      sense = to_degree(sense, degree)
      if sense not in ret:
        ret.append(sense)
    ret = list(set(ret))

  elif degree==1:
    raise ValueError("1 Degree not accepted")

  if export:
    with open(out_p, "w") as f:
      for sense in ret:
        f.write(sense)
        f.write("\n")

  return ret
