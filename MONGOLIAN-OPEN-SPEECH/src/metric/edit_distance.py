#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""Edit distance and WER computation.

Authors
 * Aku Rouhe 2020

@File       :   edit_distance.py
@Created by :   lx
@Create Time:   2021/9/26 16:21
@Description:   加载模型的一部分；比对模型
"""

import collections

EDIT_SYMBOLS = {
    "eq": "=",  # when tokens are equal
    "ins": "I",
    "del": "D",
    "sub": "S",
}


def op_table(a, b):
    """Table of edit operations between a and b.

    Solves for the table of edit operations, which is mainly used to
    compute word error rate. The table is of size ``[|a|+1, |b|+1]``,
    and each point ``(i, j)`` in the table has an edit operation. The
    edit operations can be deterministically followed backwards to
    find the shortest edit path to from ``a[:i-1] to b[:j-1]``. Indexes
    of zero (``i=0`` or ``j=0``) correspond to an empty sequence.

    The algorithm itself is well known, see

    `Levenshtein distance <https://en.wikipedia.org/wiki/Levenshtein_distance>`_

    Note that in some cases there are multiple valid edit operation
    paths which lead to the same edit distance minimum.

    Arguments
    ---------
    a : iterable
        Sequence for which the edit operations are solved.
    b : iterable
        Sequence for which the edit operations are solved.

    Returns
    -------
    list
        List of lists, Matrix, Table of edit operations.

    Example
    -------
    >>> ref = [1,2,3]
    >>> hyp = [1,2,4]
    >>> for row in op_table(ref, hyp):
    ...     print(row)
    ['=', 'I', 'I', 'I']
    ['D', '=', 'I', 'I']
    ['D', 'D', '=', 'I']
    ['D', 'D', 'D', 'S']
    """
    # For the dynamic programming algorithm, only two rows are really needed:
    # the one currently being filled in, and the previous one
    # The following is also the right initialization
    prev_row = [j for j in range(len(b) + 1)]
    curr_row = [0] * (len(b) + 1)  # Just init to zero
    # For the edit operation table we will need the whole matrix.
    # We will initialize the table with no-ops, so that we only need to change
    # where an edit is made.
    table = [
        [EDIT_SYMBOLS["eq"] for j in range(len(b) + 1)]
        for i in range(len(a) + 1)
    ]
    # We already know the operations on the first row and column:
    for i in range(len(a) + 1):
        table[i][0] = EDIT_SYMBOLS["del"]
    for j in range(len(b) + 1):
        table[0][j] = EDIT_SYMBOLS["ins"]
    table[0][0] = EDIT_SYMBOLS["eq"]
    # The rest of the table is filled in row-wise:
    for i, a_token in enumerate(a, start=1):
        curr_row[0] += 1  # This trick just deals with the first column.
        for j, b_token in enumerate(b, start=1):
            # The dynamic programming algorithm cost rules
            insertion_cost = curr_row[j - 1] + 1
            deletion_cost = prev_row[j] + 1
            substitution = 0 if a_token == b_token else 1
            substitution_cost = prev_row[j - 1] + substitution
            # Here copying the Kaldi compute-wer comparison order, which in
            # ties prefers:
            # insertion > deletion > substitution
            if (
                    substitution_cost < insertion_cost
                    and substitution_cost < deletion_cost
            ):
                curr_row[j] = substitution_cost
                # Again, note that if not substitution, the edit table already
                # has the correct no-op symbol.
                if substitution:
                    table[i][j] = EDIT_SYMBOLS["sub"]
            elif deletion_cost < insertion_cost:
                curr_row[j] = deletion_cost
                table[i][j] = EDIT_SYMBOLS["del"]
            else:
                curr_row[j] = insertion_cost
                table[i][j] = EDIT_SYMBOLS["ins"]
        # Move to the next row:
        prev_row[:] = curr_row[:]
    return table


def count_ops(table):
    """Count the edit operations in the shortest edit path in edit op table.

    Walks back an edit operations table produced by table(a, b) and
    counts the number of insertions, deletions, and substitutions in the
    shortest edit path. This information is typically used in speech
    recognition to report the number of different error types separately.

    Arguments
    ----------
    table : list
        Edit operations table from ``op_table(a, b)``.

    Returns
    -------
    collections.Counter
        The counts of the edit operations, with keys:

        * "insertions"
        * "deletions"
        * "substitutions"

        NOTE: not all of the keys might appear explicitly in the output,
        but for the missing keys collections. The counter will return 0.

    Example
    -------
    >>> table = [['I', 'I', 'I', 'I'],
    ...          ['D', '=', 'I', 'I'],
    ...          ['D', 'D', '=', 'I'],
    ...          ['D', 'D', 'D', 'S']]
    >>> print(count_ops(table))
    Counter({'substitutions': 1})
    """
    edits = collections.Counter()
    # Walk back the table, gather the ops.
    i = len(table) - 1
    j = len(table[0]) - 1
    while not (i == 0 and j == 0):
        if i == 0:
            edits["insertions"] += 1
            j -= 1
        elif j == 0:
            edits["deletions"] += 1
            i -= 1
        else:
            if table[i][j] == EDIT_SYMBOLS["ins"]:
                edits["insertions"] += 1
                j -= 1
            elif table[i][j] == EDIT_SYMBOLS["del"]:
                edits["deletions"] += 1
                i -= 1
            else:
                if table[i][j] == EDIT_SYMBOLS["sub"]:
                    edits["substitutions"] += 1
                i -= 1
                j -= 1
    return edits


def alignment(a, b, pad=0):
    """
    a: target
    b: aux
    Example
    -------
    """
    table = [
        [False for j in range(len(b))]
        for i in range(len(a))
    ]
    for i, a_token in enumerate(a):
        for j, b_token in enumerate(b):
            if a_token == b_token:
                table[i][j] = True
    # new_list = []
    # for idx in range(len(a)):
    #     t = b[table[idx].index(True)] if True in table[idx] else 0
    #     new_list.append(t)
    return [b[table[idx].index(True)] if True in table[idx] else pad for idx in range(len(a))]


def find_ops(table):
    """

    :param table: list
        Edit operations table from ``op_table(a, b)``.
    :return: list, (i, j)
    """
    edits = []
    # Walk back the table, gather the ops.
    i = len(table) - 1
    j = len(table[0]) - 1
    while not (i == 0 and j == 0):
        if i == 0:
            edits.append([i-1,j-1,"ins"])
            # edits["insertions"] += 1
            j -= 1
        elif j == 0:
            edits.append([i-1, j-1, "del"])
            # edits["deletions"] += 1
            i -= 1
        else:
            if table[i][j] == EDIT_SYMBOLS["ins"]:
                edits.append([i-1, j-1, "ins"])
                # edits["insertions"] += 1
                j -= 1
            elif table[i][j] == EDIT_SYMBOLS["del"]:
                edits.append([i-1, j-1, "del"])
                # edits["deletions"] += 1
                i -= 1
            else:
                if table[i][j] == EDIT_SYMBOLS["sub"]:
                    edits.append([i-1, j-1, "sub"])
                    # edits["substitutions"] += 1
                i -= 1
                j -= 1
    return edits


if __name__ == '__main__':
    a = [1, 2, 3, 4]  # [1, 2, 0, 3, 4]
    b = [1, 3, 5, 3, 4]
    tp1 = op_table(a, b)
    for i in tp1:
        print(i)  # 看对角线
    tp2 = count_ops(tp1)
    print(tp2)
    tp3 = find_ops(tp1)
    print(tp3)
    # c = alignment(a, b)
    # print(c)
