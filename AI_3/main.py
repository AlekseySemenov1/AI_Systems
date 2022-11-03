import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt


def calc_info(x, n):
    big_t = len(x)
    p = {}
    for i in x[n]:
        if p.get(i) is None:
            p[i] = 1
            continue
        p[i] += 1
    s = 0
    for i in p:
        s += (p[i] / big_t * np.log2(p[i] / big_t))
    return -s


def calc_info_x(x, y, n):
    big_t = len(x)
    p = {}
    for i in range(0, big_t):
        if p.get(x[n][i]) is None:
            p[x[n][i]] = [[0, 'e'], [0, 'p']]
        if y[i] == 'e':
            p[x[n][i]][0] = [p[x[n][i]][0][0] + 1, 'e']
        else:
            p[x[n][i]][1] = [p[x[n][i]][1][0] + 1, 'p']
    s = 0
    for i in p:
        if p[i][0][0] != 0 and p[i][1][0] != 0:
            s1 = p[i][0][0] + p[i][1][0]
            s += s1 / big_t * -(p[i][0][0] / s1 * np.log2(p[i][0][0] / s1) + p[i][1][0] / s1 * np.log2(p[i][1][0] / s1))
    return s


def calc_split_info(x, n):
    big_t = len(x)
    p = {}
    for i in x[n]:
        if p.get(i) is None:
            p[i] = 1
            continue
        p[i] += 1
    s = 0
    for i in p:
        s += p[i] / big_t * np.log2(p[i] / big_t)
    return -s


def calc_gain_ratio(x, y, n):
    return (calc_info(x, n) - calc_info_x(x, y, n)) / calc_split_info(x, n)


def select_attributes(x):
    n = len(x.columns) - 1
    n1 = round(np.sqrt(n))
    attributes = []
    for i in range(n1):
        numb_attr = np.random.randint(0, n)
        while numb_attr in attributes:
            numb_attr = np.random.randint(0, n)
        attributes.append(numb_attr)
    return attributes


def gain_ratio_att_list(x, y):
    attributes = select_attributes(x)
    gain_ratio_att = []
    for i in attributes:
        gain_ratio_att.append([i, calc_gain_ratio(x, y, i)])
    return gain_ratio_att


def sort_gain(x):
    return x[1]


class DecisionTreeNode(object):

    def __init__(self, name, data, y, lvl):
        self.name = name
        self.data = data
        self.y = y
        self.child = []
        self.lvl = lvl
        self.class_count = [0, 0]

    def add_child(self, node):
        self.child.append(node)

    def add_class_count(self, x):
        self.class_count[x] += 1


def create_des_tree(x, y):
    start_node = DecisionTreeNode('Mushrooms', x, y, 0)
    start_node.add_class_count(0)
    start_node.add_class_count(1)
    gain_ratio_att = sorted(gain_ratio_att_list(x, y), key=sort_gain, reverse=True)
    queue = [start_node]
    while len(queue) > 0:
        cur_node = queue.pop(0)
        if cur_node.lvl < len(gain_ratio_att) and cur_node.class_count[0] != 0 and cur_node.class_count[1] != 0:
            attr = gain_ratio_att[cur_node.lvl][0]
            p = {}
            p1 = {}
            for i in range(0, len(cur_node.data)):
                if p.get(cur_node.data[attr][i]) is None:
                    p[cur_node.data[attr][i]] = pd.DataFrame(columns=x.columns)
                    p1[cur_node.data[attr][i]] = []
                p[cur_node.data[attr][i]] = pd.concat(
                    [p[cur_node.data[attr][i]], pd.DataFrame(cur_node.data.iloc[i]).T])
                p1[cur_node.data[attr][i]].append(cur_node.y[i])
            for i in p:
                new_node = DecisionTreeNode(i, p[i].reset_index(drop=True), p1[i], cur_node.lvl + 1)
                for j in range(0, len(new_node.data)):
                    if new_node.y[j] == 'e':
                        new_node.add_class_count(0)
                    else:
                        new_node.add_class_count(1)
                cur_node.add_child(new_node)
                queue.append(new_node)
            cur_node.data = []
        else:
            continue
    return start_node, gain_ratio_att


def predict_proba(des_tree, x, gain_ratio_list):
    y = []
    for i in range(0, len(x)):
        cur_node = des_tree
        skip_node_count = 0
        while len(cur_node.child) > 0:
            for j in range(0, len(cur_node.child)):
                if x[gain_ratio_list[cur_node.lvl][0]][i] == cur_node.child[j].name:
                    cur_node = cur_node.child[j]
                    skip_node_count = 0
                    break
                skip_node_count += 1
            if skip_node_count == len(cur_node.child):
                break
        if skip_node_count == len(cur_node.child) and len(cur_node.child) > 0:
            y.append([0, 0])
        else:
            if cur_node.class_count[0] >= cur_node.class_count[1]:
                y.append([1, 0])
            else:
                y.append([0, 1])

    return y


def predict(des_tree, x, gain_ratio_list):
    y = []
    for i in range(0, len(x)):
        cur_node = des_tree
        skip_node_count = 0
        while len(cur_node.child) > 0:
            for j in range(0, len(cur_node.child)):
                if x[gain_ratio_list[cur_node.lvl][0]][i] == cur_node.child[j].name:
                    cur_node = cur_node.child[j]
                    skip_node_count = 0
                    break
                skip_node_count += 1
            if skip_node_count == len(cur_node.child):
                break
        if skip_node_count == len(cur_node.child) and len(cur_node.child) > 0:
            y.append('Undefined')
        elif cur_node.class_count[0] >= cur_node.class_count[1]:
            y.append('e')
        else:
            y.append('p')
    return y


def check(y1, y2):
    t_p = 0
    f_p = 0
    t_n = 0
    f_n = 0
    for i in range(len(y1)):
        if y2[i] == 'Undefined':
            continue
        if y2[i] == y1[i] and y1[i] == 'p':
            t_n += 1
        elif y2[i] == y1[i]:
            t_p += 1
        elif y2[i] != y1[i] and y1[i] == 'p':
            f_n += 1
        else:
            f_p += 1
    return t_p / (t_p + f_n), t_p / (t_p + f_p), (t_p + t_n) / (t_p + t_n + f_n + f_p)


def roc_draw(y_test, pred_proba):
    y1 = []
    y2 = []
    for i in range(len(pred_proba)):
        if y_test[i] == 'p':
            y1.append(0)
        else:
            y1.append(1)
        y2.append(pred_proba[i][0])
    fpr, tpr, treshold = roc_curve(y1, y2)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange',
             label='ROC кривая (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Пример ROC-кривой')
    plt.legend(loc="lower right")
    plt.savefig("ROC.png")
    plt.show()


def pr_draw(y_test, proba):
    y1 = []
    y2 = []
    for i in range(len(proba)):
        if y_test[i] == 'p':
            y1.append(0)
        else:
            y1.append(1)
        if proba[i] == 'p':
            y2.append(0)
        else:
            y2.append(1)
    precision, recall, _ = precision_recall_curve(y1, y2)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.savefig('PR.jpg')
    plt.show()


def main():
    file = pd.read_csv("C:\\Users\\Asus\\PycharmProjects\\CII3\\agaricus-lepiota.data", header=None, delimiter=',')
    x = file.drop([22], axis=1)
    y = file[22]
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=2)
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    des_tree, gain_ratio_att = create_des_tree(x_train, y_train)
    pred_proba = predict_proba(des_tree, x_test, gain_ratio_att)
    pred = predict(des_tree, x_test, gain_ratio_att)
    print(y_test)
    print(pred)
    print(check(y_test, pred))
    roc_draw(y_test, pred_proba)
    pr_draw(y_test, pred)


main()
