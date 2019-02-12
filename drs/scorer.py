"""Accuracy & F1 scorer for BERT outputs

Assumed either entirely implicit or explicit
"""
import sys

def accuracy(gold_list, auto_list):
    acc_bin = [gold == auto for gold, auto in zip(gold_list, auto_list)]
    acc = sum(acc_bin) / len(acc_bin)
    print('Accuracy: {:<13.5}'.format(acc), end='\n\n')
    return acc

def prf_for_one_tag(gold_list, auto_list, tag):
    tp, fp, fn = 0, 0, 0
    for gold, auto in zip(gold_list, auto_list):
        if tag==gold and tag==auto:
            tp += 1
        elif tag==gold:
            fn += 1
        elif tag==auto:
            fp += 1

    p = tp / (tp + fp) if tp + fp != 0 else 0.
    r = tp / (tp + fn) if tp + fn != 0 else 0.
    f = 2 * p * r / (p + r) if p + r != 0 else '-'
    print('{:35} precision {:<13.5}recall {:<13.5}F1 {:<13.5}'.format(
        tag, p, r, f))
    return tp, fp, fn

def prf(gold_list, auto_list):
    gold_tag_set = sorted(set(gold_list))

    total_tp, total_fp, total_fn = 0, 0, 0
    for tag in gold_tag_set:
        tp, fp, fn, = prf_for_one_tag(gold_list, auto_list, tag)
        total_tp += tp
        total_fp += fp
        total_fn += fn
    print()
    print("{:35} TP: {:<18} FN: {:<15} FP: {}".format(
        "Total Counts", total_tp, total_fn, total_fp))

    p = total_tp / (total_tp + total_fp) if total_tp + total_fp != 0 else 0.
    r = total_tp / (total_tp + total_fn) if total_tp + total_fn != 0 else 0.
    f = 2 * p * r / (p + r) if p + r != 0 else '-'
    print('{:35} precision {:<13.5}recall {:<13.5}F1 {:<13.5}'.format(
        'Micro-Average', p, r, f))
    print()
    return p,r,f

if __name__=='__main__':
    gold = sys.argv[1]
    auto = sys.argv[2]

    gold_list = [line.strip() for line in open(gold)]
    auto_list = [line.strip() for line in open(auto)]

    if 'implicit' in gold:
        task_type = "implicit"
    else:
        task_type = "explicit"

    print('='*90 + '\nEvaluation for {} discourse relations\n'.format(task_type))
    accuracy(gold_list, auto_list)
    prf(gold_list, auto_list)
