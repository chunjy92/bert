#! /usr/bin/python3
# -*- coding: utf-8 -*-
"""Originally to split explicit vs implicit
now used ONLY to collect statistics of dataset

Output format expected as input to md file
"""
import codecs
import json
import os
import time
from collections import Counter

from drs.senses import get_level, to_level

__author__ = 'Jayeol Chun'

# root data dir
DATA_PATH = "/home/b/jchun/Documents/Data/DiscourseRelation"


if __name__ == '__main__':
  begin = time.time()

  for dataset in ['dev', 'test', 'train']:
    rel_path = os.path.join(DATA_PATH, dataset, "relations.json")

    print("\n==================Processing {:5}==================="
          .format(dataset.upper()))
    pdtb = codecs.open(rel_path, encoding='utf-8')

    # lengths
    total_conns = 0
    arg1_len = arg2_len = conn_len = 0 # for avg
    max_arg1 = max_arg2 = max_conn = 0 # max
    min_arg1 = min_arg2 = 100000 # min

    # multi senses
    multi_senses = 0
    multi_sense_counter = Counter()

    # levels
    level_counter = Counter()

    # relation type_counter
    type_counter = Counter()

    for pdtb_line in pdtb:
      rel = json.loads(pdtb_line)

      # relation type_counter
      type_counter.update([rel['Type']])

      # connective lengths
      if rel['Type']=='Explicit':
        # connective lengths
        conn_l = len(rel['Connective']['RawText'].split())
        conn_len += conn_l
        max_conn = conn_l if conn_l > max_conn else max_conn
        total_conns += 1

      # arg1 and arg2 lengths
      arg1_l = len(rel['Arg1']['RawText'].split())
      arg2_l = len(rel['Arg2']['RawText'].split())

      # avg length
      arg1_len += arg1_l
      arg2_len += arg2_l

      # max length
      max_arg1 = arg1_l if arg1_l > max_arg1 else max_arg1
      max_arg2 = arg2_l if arg2_l > max_arg2 else max_arg2

      # min length
      min_arg1 = arg1_l if arg1_l < min_arg1 else min_arg1
      min_arg2 = arg2_l if arg2_l < min_arg2 else min_arg2

      # multi senses
      senses = rel['Sense']
      if len(senses) > 1:
        multi_senses += 1
        sorted_senses = sorted([to_level(s) for s in senses])
        key = sorted_senses[0] + " & " + sorted_senses[1]
        multi_sense_counter.update([key])

      # sense levels
      for sense in senses:
        level_counter.update([get_level(sense)])

    # total number of relations
    total_rels = sum(type_counter.values())
    print("\nNumber of Relations: {}".format(total_rels))

    # distribution of relation type_counter
    print("\nRelation type_counter:")
    for k, v in type_counter.most_common():
      print("* {:10}: {:3} => {:.2f}%".format(k, v, v*100/total_rels))

    # length stats
    print("\nAverage Lengths:")
    print("* Arg1: {:.4}\n* Arg2: {:.4}\n* Conn: {:.4}".format(
      arg1_len / total_rels, arg2_len / total_rels, conn_len / total_conns))

    print("\nMax Lengths:")
    print("* Arg1: {}\n* Arg2: {}\n* Conn: {}".format(
      max_arg1, max_arg2, max_conn))

    print("\nMin Lengths:")
    print("* Arg1: {}\n* Arg2: {}".format(min_arg1, min_arg2))

    # percentage of relations with multi senses
    print("\nNum Multi Sense: {} => {:.2f}%".format(
      multi_senses, multi_senses * 100 / total_rels))

    # distribution of senses by level
    print("\nNumber of Senses by Level:")
    total_levels = sum(level_counter.values())
    for k, v in level_counter.most_common():
      print("* {}th : {:3} => {:.2f}%".format(k, v, v * 100 / total_levels))

    # distribution of multi senses
    print("\nList of Multi Senses:")
    total_multi_senses = sum(multi_sense_counter.values())
    for k, v in multi_sense_counter.most_common():
      print("* {:46}: {:3} => {:.2f}%".format(k, v, v*100 / total_multi_senses))

  print("\nExecution Time: {:.4f}s".format(time.time() - begin))
