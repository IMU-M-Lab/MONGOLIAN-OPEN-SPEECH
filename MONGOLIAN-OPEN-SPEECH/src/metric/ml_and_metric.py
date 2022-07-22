'''
文件名<空格>文本
'''
from jiwer import cer, wer
import re

def M2L(latStr):
    tMgUnicode = [0x1820, 0x1821, 0x1822, 0x1823, 0x1824, 0x1825, 0x1826, 0x1827, 0x1828, 0x1829, 0x182a, 0x182b,
                  0x182c, 0x182d, 0x182e, 0x182f, 0x1830, 0x1831, 0x1832, 0x1833, 0x1834, 0x1835, 0x1836, 0x1837,
                  0x1838, 0x1839, 0x183a, 0x183b, 0x183c, 0x183d, 0x183e, 0x183f, 0x1840, 0x1841, 0x1842, 0x180a,
                  0x180b, 0x180c, 0x180d, 0x180e, 0x202f, 0x200c, 0x200d, 0x0020, 0x000d, 0x000a, 0x1801, 0x1802,
                  0x1803, 0x1804, 0x2014, 0x300a, 0x300b, 0x3014, 0x3015, 0x3008, 0x3009];
    tMgLDCodeNew = [97, 101, 105, 113, 118, 111, 117, 69, 110, 78, 98, 112, 104, 103, 109, 108, 115, 120, 116, 100,
                    99, 106, 121, 114, 119, 102, 107, 75, 67, 122, 72, 82, 76, 90, 81, 38, 39, 34, 96, 95, 45, 94, 42,
                    32, 13, 10, 92, 44, 46, 58, 36, 60, 62, 91, 93, 123, 125]

    monStr = [];
    for i in range(len(latStr)):
        flag = 0
        for j in range(57):
            if latStr[i] == chr(tMgUnicode[j]):
                monStr.append(chr(tMgLDCodeNew[j]))
                flag = 1
                break
        if flag == 0:
            monStr.append(latStr[i])
    return ''.join(monStr)


def sort_files(file1, file2):
    with open(file1, encoding='utf8') as f:
        recList = f.readlines()

    with open(file2, encoding='utf8') as f:
        refList = f.readlines()

    sort_ref = []
    sort_rec = []

    for rec in recList:
        for ref in refList:
            if ref.split(' ', maxsplit=1)[0] == rec.split(' ', maxsplit=1)[0]:
                sort_ref.append(ref)
                sort_rec.append(rec)

    fileObject = open(file1, "w", encoding='utf-8')
    for ip in sort_rec:
        fileObject.write(ip)
        # fileObject.write('\n')
    fileObject.close()

    fileObject = open(file2, "w", encoding='utf-8')
    for ip in sort_ref:
        fileObject.write(ip)
        # fileObject.write('\n')
    fileObject.close()


def ml_process(Sentences_processed):
    list = []
    Sentences_processed = Sentences_processed.replace('▁', ' ')
    Sentences_processed = Sentences_processed.replace('<unk>', ' ')
    # name = x.split()[0]
    latin = M2L(Sentences_processed).replace('`', '   `')
    latin = latin.replace('-', '   -')
    latin = re.sub("\.|\,|\!|\?|\:", " ", latin)
    latin = re.sub("\s+", " ", latin)
    list.append(latin)
    return list


def compute_cwer(rec, ref):
    # with open(file1, encoding='utf8') as f:
    #     est = f.readlines()
    #
    # with open(file2, encoding='utf8') as f:
    #     ref = f.readlines()
    return wer(ref, rec), cer(ref, rec)
