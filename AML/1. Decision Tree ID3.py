import math
import csv


def entropy(data):
    labels = [row[-1] for row in data]
    label_counts = {}
    for label in labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    entropy = 0
    for count in label_counts.values():
        prob = count / len(data)
        entropy -= prob * math.log2(prob)
    return entropy


def split_data(data, attribute_index, attribute_value):
    new_data = []
    for row in data:
        if row[attribute_index] == attribute_value:
            new_row = row[:attribute_index] + row[attribute_index+1:]
            new_data.append(new_row)
    return new_data


def select_best_attribute(data, attributes):
    base_entropy = entropy(data)
    best_info_gain = 0
    best_attribute_index = -1
    for i in range(len(attributes)):
        attribute_values = set([row[i] for row in data])
        new_entropy = 0
        for value in attribute_values:
            subset = split_data(data, i, value)
            prob = len(subset) / len(data)
            new_entropy += prob * entropy(subset)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_attribute_index = i
    return best_attribute_index


def majority_vote(labels):
    label_counts = {}
    for label in labels:
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1
    max_count = 0
    majority_label = None
    for label, count in label_counts.items():
        if count > max_count:
            max_count = count
            majority_label = label
    return majority_label


def build_tree(data, attributes):
    labels = [row[-1] for row in data]
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    if len(attributes) == 0:
        return majority_vote(labels)
    best_attribute_index = select_best_attribute(data, attributes)
    best_attribute = attributes[best_attribute_index]
    tree = {best_attribute: {}}
    attribute_values = set([row[best_attribute_index] for row in data])
    for value in attribute_values:
        subset = split_data(data, best_attribute_index, value)
        sub_attributes = attributes[:best_attribute_index] + \
            attributes[best_attribute_index+1:]
        tree[best_attribute][value] = build_tree(subset, sub_attributes)
    return tree


def print_tree(tree, indent=''):
    if isinstance(tree, str):
        print(tree)
        return
    attribute = list(tree.keys())[0]
    subtree = tree[attribute]
    print("\n"+indent + attribute + ":")
    for value, sub_tree in subtree.items():
        print(indent + "  " + value + " -> ", end='')
        print_tree(sub_tree, indent + "    ")


def predict(tree, instance):
    if isinstance(tree, str):
        return tree
    attribute = list(tree.keys())[0]
    attribute_value = instance[attribute]
    if attribute_value not in tree[attribute]:
        return None
    subtree = tree[attribute][attribute_value]
    return predict(subtree, instance)


data = []
with open('computer.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for lines in csvFile:
        data.append(lines)

attributes = data[0]
data = data[1:]


tree = build_tree(data, attributes)


print("Decision Tree:")
print_tree(tree)
print()


instance = {"age": "youth", "income": "medium",
            "student": "yes", "credit_rating": "fair"}
prediction = predict(tree, instance)

print("Prediction for instance:", instance)
print("Predicted label:", prediction)
