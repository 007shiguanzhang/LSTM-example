import codecs
import os
import json


class Loading(object):
    """
    loading dict from txt
    """
    def __init__(self):
        d = dict()
        path = os.getcwd() + "/dictionary"
        files = os.listdir(path)
        for file in files:
            with codecs.open((path+'/'+file), 'rb', encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.split(',')
                    tag, content = line[0], line[1]
                    if content:
                        flag = content[0]
                        d[flag] = d.get(flag, []) + [[content, tag]]

        self.d = d


def tag_dict(s):
    result = []
    i = 0
    label_dict = {'1': "BEH", '2': "CHE", '3': "DIS", '4': "TRE", '5': "BOD"}

    with open('json_Dict.json', 'r') as f:
        load_dict = json.load(f)
    while i < len(s):
        if s[i] not in load_dict:
            result.append('0')
            i = i + 1
        else:
            temp = load_dict[s[i]]
            content = [kk[0] for kk in temp]
            flag = False
            for index, k in enumerate(content):
                if k == s[i:i+len(k)]:
                    print("标记字符:", k)
                    temp[index][1] = temp[index][1][-1]  # 由于解码问题产生的
                    tag = label_dict[temp[index][1]]
                    if len(k) == 1:
                        result.append("S-"+tag)
                    elif len(k) == 2:
                        result.extend(["B-"+tag, "E-"+tag])
                    else:
                        result.extend(["B-"+tag, ("M-"+tag)*(len(k)-2), "E-"+tag])
                    i += len(k)
                    flag = True
            if not flag:
                result.append('0')
                i = i + 1
    return result


def fusing(Dict_tag, Lstm_tag):
    '''
    :param Dict_tag: 
    :param Lstm_tag: 
    :return: fused—tag
    '''
    if len(Dict_tag) != len(Lstm_tag):
        return 'error: the length not equal'
    else:
        length = len(Dict_tag)
    result = []
    for i in range(length):
        if Dict_tag[i] != 0:
            result.append(Dict_tag[i])
        else:
            result.append(Lstm_tag[i])
    return result

if __name__ == '__main__':
    dictionary = Loading()
    d = dictionary.d
    s = u'男、39岁，承德市双滦区人。主因上腹部、腰部疼痛1天入院。'
    result = tag_dict(s)
    print('input:', s, '\n', 'output:', result)

