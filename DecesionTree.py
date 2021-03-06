#coding:utf-8
import sys
from math import log
import operator
from numpy import mean


def get_labels(train_file):
    '''
    返回所有数据集labels（列表）
    :param train_file:
    :return:
    '''
    labels = []
    for line in open(train_file):
        label = line.strip().split(',')[-1]
        labels.append(label)
    return labels


def format_data(dataset_file):

    '''
    返回dataset(列表集合）和features（列表）
    '''
    j = 0
    dataset = []
    for line in open(dataset_file, 'rU'):
        line = line.strip()
        j += 1
        fea_and_label = line.split(',')
        dataset.append(
            [float(fea_and_label[i]) for i in range(len(fea_and_label) - 1)] + [fea_and_label[len(fea_and_label) - 1]])
    features = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas', 'X_Perimeter', 'Y_Perimeter', 'Sum_of_Luminosity', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity', 'Length_of_Conveyer', 'TypeOfSteel_A300', 'TypeOfSteel_A400', 'Steel_Plate_Thickness', 'Edges_Index', 'Empty_Index', 'Square_Index', 'Outside_X_Index', 'Edges_X_Index', 'Edges_Y_Index', 'Outside_Global_Index', 'LogOfAreas', 'Log_X_Index', 'Log_Y_Index', 'Orientation_Index', 'Luminosity_Index', 'SigmoidOfAreas']
    return dataset, features


def split_dataset(dataset, feature_index):
    '''
    按指定feature划分数据集，返回四个列表:
	@dataset_less:指定特征项的属性值＜=该特征项平均值的子数据集
	@dataset_greater:指定特征项的属性值＞该特征项平均值的子数据集
	@label_less:按指定特征项的属性值＜=该特征项平均值切割后子标签集
	@label_greater:按指定特征项的属性值＞该特征项平均值切割后子标签集
    :param dataset:
    :param feature_index:
    :param labels:
    :return:
    '''
    dataset_less = []
    dataset_greater = []
    label_less = []
    label_greater = []
    datasets = []
    for data in dataset:
        datasets.append(data[0:26])
    mean_value = mean(datasets, axis=0)[feature_index]
    for data in dataset:
        if data[feature_index]>mean_value:
            dataset_greater.append(data)
            label_greater.append(data[-1])
        else:
            dataset_less.append(data)
            label_less.append(data[-1])
    return dataset_less, dataset_greater, label_less, label_greater


def cal_entropy(dataset):
    '''
    计算数据集熵的大小
    :param dataset:
    :return:
    '''
    n = len(dataset)
    label_count = {}
    for data in dataset:
        label = data[-1]
        label_count[label] = label_count.get(label, 0) + 1
    entropy = 0
    for label in label_count:
        prob = float(label_count[label])/n
        entropy -= prob*log(prob, 2)
    # print "entropy:", entropy
    return entropy


def cal_info_gain(dataset, feature_index, base_entropy):
    '''
    计算制定特征对数据集的信息增益值
    g(D,F) = H(D) - H(D/F) = entropy(dataset) - sum{1,k}(len(sub_dataset)/len(dataset))*entropy(sub_dataset)
    @base_entropy = H(D)
    :param dataset:
    :param feature_index:
    :param base_entropy:
    :return:
    '''
    datasets = []
    for data in dataset:
        datasets.append(data[0:26])
    mean_value = mean(datasets, axis=0)[feature_index] #计算指定特征的所有数据集值的平均值
    dataset_less = []
    dataset_greater = []
    for data in dataset:
        if data[feature_index] > mean_value:
            dataset_greater.append(data)
        else:
            dataset_less.append(data)
    #条件熵 H(D/F)
    condition_entropy = float(len(dataset_less))/len(dataset)*cal_entropy(dataset_less) + float(len(dataset_greater))/len(dataset)*cal_entropy(dataset_greater)
    return base_entropy - condition_entropy


def cal_info_gain_ratio(dataset, feature_index):
    '''
    	计算信息增益比  gr(D,F) = g(D,F)/H(D)
    	'''
    base_entropy = cal_entropy(dataset)
    if base_entropy == 0:
        return 1
    info_gain = cal_info_gain(dataset, feature_index, base_entropy)
    info_gain_ratio = info_gain / base_entropy
    return info_gain_ratio


def choose_best_fea_to_split(dataset, features):
   '''
   根据每个特征的信息增益比大小，返回最佳划分数据集的特征索引
   :param dataset:
   :param features:
   :return:
   '''

   split_fea_index = -1
   max_info_gain_ratio =0.0
   for i in range(len(features)-1):
       info_gain_ratio = cal_info_gain_ratio(dataset, i)
       if info_gain_ratio > max_info_gain_ratio:
           max_info_gain_ratio = info_gain_ratio
           split_fea_index = i

   return split_fea_index


def most_occur_label(labels):
    '''
    返回数据集中出现此书最多的label
    :param labels:
    :return:
    '''
    label_count = {}
    for label in labels:
        if label not in label_count.keys():
            label_count[label] = 1
        else:
            label_count[label] += 1
    sorted_label_count = sorted(label_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_label_count[0][0]


def build_tree(dataset, labels, features):
    '''
    创建决策树
    :param dataset: 训练数据集
    :param labels: 数据集中包含的所有label
    :param features: 可进行划分的特征集
    :return:
    '''
    #若数据集为空，返回NULL
    if len(labels) == 0:
        return 'NULL'
    #若数据集中只有一种label，返回该label
    if len(labels) == len(labels[0]):
        return labels[0]
    #若没有可划分的特征集，则返回数据集中出现此书最多的label
    if len(features) == 0:
        return most_occur_label(labels)
    #若数据集趋于稳定，则返回数据集中出现次数最多的label
    if cal_entropy(dataset) == 0:
        return most_occur_label(labels)
    split_feature_index = choose_best_fea_to_split(dataset, features)
    split_feature = features[split_feature_index]
    decesion_tree = {split_feature:{}}
    #若划分特征的信息增益比小于阈值，则返回数据集中出现次数最多的label
    if cal_info_gain_ratio(dataset, split_feature_index) < 0.17:
        return most_occur_label(labels)
    del (features[split_feature_index])
    dataset_less, dataset_greater, labels_less, labels_greater = split_dataset(dataset, split_feature_index)
    decesion_tree[split_feature]['<='] = build_tree(dataset_less, labels_less, features)
    decesion_tree[split_feature]['>'] = build_tree(dataset_greater, labels_greater, features)
    return decesion_tree


def store_tree(decesion_tree, filename):
    '''
    把决策树以二进制格式写入文件
    :param decesion_tree:
    :param filename:
    :return:
    '''
    import pickle
    writer = open(filename, 'w')
    pickle.dump(decesion_tree, writer)
    writer.close()


def read_tree(filename):
    '''
    从文件中读取决策树，返回决策树
    :param filename:
    :return:
    '''
    import pickle
    reader = open(filename, 'rU')
    return pickle.load(reader)


def classify(decesion_tree, features, test_data, mean_values):
    '''
    对测试数据进行分类，decesion_tree:{'petal_length': {'<=': {'petal_width': {'<=': 'Iris-setosa', '>': {'sepal_width': {'<=': 'Iris-versicolor', '>': {'sepal_length': {'<=': 'Iris-setosa', '>': 'Iris-versicolor'}}}}}}, '>': 'Iris-virginica'}}
    :param decesion_tree:
    :param feature:
    :param test_data:
    :param mean_values:
    :return:
    '''
    first_fea = decesion_tree.keys()[0]
    fea_index = features.index(first_fea)
    if test_data[fea_index] <= mean_values[fea_index]:
        sub_tree = decesion_tree[first_fea]['<=']
        if type(sub_tree) == dict:
            return classify(sub_tree, features, test_data, mean_values)
        else:
            return sub_tree
    else:
        sub_tree = decesion_tree[first_fea]['>']
        if type(sub_tree) == dict:
            return classify(sub_tree, features, test_data, mean_values)
        else:
            return sub_tree


def get_means(train_dataset):
    '''
    获取训练数据集各个属性的数据平均值
    :param train_dataset:
    :return:
    '''
    dataset = []
    for data in train_dataset:
        dataset.append(data[0:26])
    mean_values = mean(dataset, axis=0)
    return mean_values


def run(train_file, test_file):
    '''主函数'''
    labels = get_labels(train_file)
    train_dataset, train_features = format_data(train_file)
    decesion_tree = build_tree(train_dataset, labels, train_features)
    # print 'decesion_tree:', decesion_tree
    store_tree(decesion_tree, 'decesiontree')
    mean_values = get_means(train_dataset)
    test_dataset, test_features = format_data(test_file)
    n = len(test_dataset)
    correct = 0
    print decesion_tree
    for test_data in test_dataset:
        label = classify(decesion_tree, test_features,test_data,mean_values)
        if label == test_data[-1]:
            correct += 1
    print '准确率：', correct/float(n)



if __name__ == '__main__':
    run('train2.csv', 'test.csv')
