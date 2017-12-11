#!/usr/bin/python3
# ################################################################
# K-Nearest Neighbour Test Running Time: 2325.7810401916504
# Output has been written in the file, output.txt
# ################################################################
# For k-nearest, you still need to perform "training", which
# essentially stores the training data. This data is used during
# testing. Since it checks 1000 images with 40000 images, it takes
# a long time to run during testing.
# As for the value of K, we are using 35.
# ################################################################
#
#
# Model: adaboost
# Adaboost Training Running Time: 68.51164317131042
# Adaboost Test Accuracy: 0.324
# Adaboost Test Running Time: 0.1714920997619629
# Output has been written in the file, output.txt
# ################################################################
# For adaboost, we are creating 20 decision stumps. Each stump
# randomly picks 2 pixels in the image and randomly picks an
# opeartion such as "less than equal to" or "greater than equal to".
# Because of it is only only looking at 2 random pixels, it does not
# perform very well, with an accuracy of 32.4%.
# ################################################################
#
#
# Model: nnet
# Neural Network Training Running Time: 291.90286898612976
# Neural Network Test Accuracy: 68.29268292682927
# Neural Network Test Running Time: 0.29344725608825684
# Output has been written in the file, output.txt
# ################################################################
# For Neural Networks, we have 1 hidden layer with 50 nodes.
# For the hidden layer we are using a sigmoid activation function.
# Fo the output layer we are using a softmax activation function.
# ################################################################

import sys
import numpy as np
import time
import pickle
import os
from operator import itemgetter
from random import uniform, choice, randint
from operator import itemgetter, le, ge, lt, gt
from collections import Counter



class test:
    def __init__(self, test_filepath, model, model_filepath="model_file.txt", output_filepath="output.txt"):
        self.test_filepath = test_filepath
        self.model_filepath = model_filepath
        self .output_filepath = output_filepath
        self.model = model
        self.img_name = []
        self.img_orientation = []
        self.img_labels = []
        self.img_data = []
        self.img_output = []

        self.config = {}
        self.parameters = {}

        self.load_model()
        self.loadData()

    def loadData(self):
        f = open(self.test_filepath)
        line = f.readline()
        count = 0
        while line and count != -1000:
            count += 1
            data = line.split(" ")
            self.img_name.append(data[0])
            self.img_labels.append(data[1])
            if data[1] == '0':
                self.img_orientation.append([1, 0, 0, 0])
            elif data[1] == '90':
                self.img_orientation.append([0, 1, 0, 0])
            elif data[1] == '180':
                self.img_orientation.append([0, 0, 1, 0])
            elif data[1] == '270':
                self.img_orientation.append([0, 0, 0, 1])

            self.img_data.append([int(x) for x in data[2:]])
            line = f.readline()

        self.img_name = np.asarray(self.img_name)
        self.img_orientation = np.asarray(self.img_orientation)
        self.img_data = np.asarray(self.img_data)

    def toString(self):
        print("Testing File:", self.test_filepath)
        print("Model File:", self.model_filepath)
        print("Model:", self.model)
        print("\n")

    def head(self):
        print("Head:")
        print("\t\tName:", self.img_name[0])
        print("\t\tOrientation:", self.img_orientation[0])
        print("\t\tData:", self.img_data[0])

    def print_data(self):
        print("Data:")
        for i in range(len(self.img_name)):
            print("\t\tName:", self.img_name[i])
            print("\t\tOrientation:", self.img_orientation[i])
            print("\t\tData:", self.img_data[i])

    def print_data_shape(self):
        print("Shape:")
        print("\t\tLength of Orientation list:", self.img_orientation.shape)
        print("\t\tLength of each element in Data list:", self.img_data.shape)

    def print_model(self):
        print("Parameters for ", self.model, ":")
        print("\t\tConfiguration: ", self.config)
        print("\t\tParameters: ", self.parameters)

    def load_model(self):
        if os.path.isfile(self.model_filepath):
            pickled_config = pickle.load(open(self.model_filepath, "rb"))
            self.config = pickled_config[self.model]["config"]
            self.parameters = pickled_config[self.model]["parameters"]
        else:
            print("Model file not found. Please train model or correctly load model file.")

    def save_output(self):
        filehandle = open(self.output_filepath, "w")
        for line in self.img_output:
            filehandle.write(" ".join(str(x) for x in line) + '\n')
        # filehandle.writelines(self.img_output)
        print("Output has been written in the file, {0}".format(self.output_filepath))

    def nearest(self):
        k = self.config["k"]

        def toString():
            print("Configurations for K-Nearest")
            print("\t\tValue of K: ", k)

        def euclidean_Distance(x, y):
            return np.sqrt(np.sum(np.power([m - n for m, n in zip(x, y)], 2)))

        # toString()
        X = self.parameters["X"]
        y = self.parameters["y"]
        accuracy = 0
        for i in range(self.img_data.shape[0]):
            dist = []

            for j in range(X.shape[0]):
                dist.append([y[j], euclidean_Distance(self.img_data[i], X[j])])
            dist = sorted(dist, key=itemgetter(1))

            ans = [0, 0, 0, 0]
            for j in range(k):
                ans[np.argmax(dist[j][0])] += 1

            if np.argmax(self.img_orientation[i]) == np.argmax(ans):
                accuracy += 1

            if np.argmax(ans) == 0:
                self.img_output.append((self.img_name[i], "0"))
            elif np.argmax(ans) == 1:
                self.img_output.append((self.img_name[i], "90"))
            elif np.argmax(ans) == 2:
                self.img_output.append((self.img_name[i], "180"))
            elif np.argmax(ans) == 3:
                self.img_output.append((self.img_name[i], "270"))

        print("K-Nearest Neighbour Test Accuracy: {0}".format(100 * accuracy / self.img_data.shape[0]))
        print("K-Nearest Neighbour Test Running Time: {0}".format(time.time() - start))

    def adaboost(self):

        def predict_label(stump, image):
            pixel1, pixel2 = stump['pixels']
            if stump['op'](image[pixel1], image[pixel2]):
                return stump['label']
            else:
                return stump['alt']

        def use_stumps(stumps):
            predictions = []
            images = self.img_data.tolist()
            for image in images:
                weighted_prediction = {'0': 0, '90': 0, '180': 0, '270': 0}
                for stump_id, stump in stumps.items():
                    weighted_prediction[predict_label(stump=stump, image=image)] += stump['weight']
                predictions.append(max(weighted_prediction.items(), key=lambda x: x[1])[0])
            return predictions

        predicted_labels = use_stumps(self.parameters["stumps"])
        label_pairs = list(zip(predicted_labels, self.img_labels))
        self.img_output = list(zip(self.img_name, predicted_labels))
        print("Adaboost Test Accuracy: {0}".format(round(sum([label[0] == label[1] for label in label_pairs])/len(predicted_labels), 3)))
        print("Adaboost Test Running Time: {0}".format(time.time() - start))

    def nnet(self):
        iterations = self.config["iterations"]
        alpha = self.config["learning_rate"]

        hiddenneurons = self.config["hidden_neurons"]

        weight_hidden = self.parameters["weight_hidden"]
        bias_hidden = self.parameters["bias_hidden"]
        weight_output = self.parameters["weight_output"]
        bias_output = self.parameters["bias_output"]

        accuracy = 0

        def toString():
            print("Configurations for Neural Net")
            print("\t\tIterations: ", iterations)
            print("\t\tAlpha: ", alpha)
            print("\t\tHidden Neurons: ", hiddenneurons)

        def sigmoid(x):
            return np.divide(1, (1 + np.exp(-x)))

        def softmax(x):
            max_per_row = np.amax(x, axis=1)
            max_per_row = max_per_row.reshape((max_per_row.shape[0], 1))
            y = x - max_per_row
            probs = np.exp(y) + 0.001
            probs_sum = np.sum(probs, axis=1, keepdims=True)
            probs_sum = np.reshape(probs_sum, (probs_sum.shape[0], 1))
            return probs / probs_sum

        Z1 = sigmoid((np.dot(self.img_data, weight_hidden)) + bias_hidden)
        ans = softmax((np.dot(Z1, weight_output)) + bias_output)
        for j in range(self.img_data.shape[0]):
            if np.argmax(self.img_orientation[j]) == np.argmax(ans[j]):
                accuracy += 1

            if np.argmax(ans[j]) == 0:
                self.img_output.append((self.img_name[j], "0"))
            elif np.argmax(ans[j]) == 1:
                self.img_output.append((self.img_name[j], "90"))
            elif np.argmax(ans[j]) == 2:
                self.img_output.append((self.img_name[j], "180"))
            elif np.argmax(ans[j]) == 3:
                self.img_output.append((self.img_name[j], "270"))

        print("Neural Network Test Accuracy: {0}".format(100 * accuracy / self.img_data.shape[0]))
        print("Neural Network Test Running Time: {0}".format(time.time() - start))

    def test(self):
        if self.model == 'nearest':
            self.nearest()

        if self.model == 'adaboost':
            self.adaboost()

        if self.model == 'nnet':
            self.nnet()

        self.save_output()


class train:
    def __init__(self, train_filepath, model_filepath,  config, model="model_file.txt"):
        self.train_filepath = train_filepath
        self.model_filepath = model_filepath
        self.model = model
        self.img_name = []
        self.img_orientation = []
        self.img_labels = []
        self.img_data = []
        self.config = config["config_" + self.model]
        self.parameters = {}

        self.loadData()

    def toString(self):
        print("Training File:", self.train_filepath)
        print("Model File:", self.model_filepath)
        print("Model:", self.model)
        print("\n")

    def head(self):
        print("Head:")
        print("\t\tName:", self.img_name[0])
        print("\t\tOrientation:", self.img_orientation[0])
        print("\t\tData:", self.img_data[0])

    def print_data(self):
        print("Data:")
        for i in range(len(self.img_name)):
            print("\t\tName:", self.img_name[i])
            print("\t\tOrientation:", self.img_orientation[i])
            print("\t\tData:", self.img_data[i])

    def print_data_shape(self):
        print("Shape:")
        print("\t\tLength of Orientation list:", self.img_orientation.shape)
        print("\t\tLength of each element in Data list:", self.img_data.shape)

    def loadData(self):
        f = open(self.train_filepath)
        line = f.readline()
        count = 0
        while line and count != -1000:
            count += 1
            data = line.split(" ")
            self.img_name.append(data[0])
            self.img_labels.append(data[1])
            if data[1] == '0':
                self.img_orientation.append([1, 0, 0, 0])
            elif data[1] == '90':
                self.img_orientation.append([0, 1, 0, 0])
            elif data[1] == '180':
                self.img_orientation.append([0, 0, 1, 0])
            elif data[1] == '270':
                self.img_orientation.append([0, 0, 0, 1])

            self.img_data.append([int(x) for x in data[2:]])
            line = f.readline()

        self.img_name = np.asarray(self.img_name)
        self.img_orientation = np.asarray(self.img_orientation)
        self.img_data = np.asarray(self.img_data)

    def save_model(self):
        if os.path.isfile(self.model_filepath):
            pickled_config = pickle.load(open(self.model_filepath, "rb"))

            if self.model not in pickled_config:
                pickled_config[self.model] = {}

            pickled_config[self.model]["config"] = self.config
            pickled_config[self.model]["parameters"] = self.parameters

        else:
            pickled_config = {
                self.model: {
                    "config": self.config,
                    "parameters": self.parameters
                }
            }
        pickle.dump(pickled_config, open(self.model_filepath, "wb"))

    def nearest(self):
        k = self.config["k"]

        def toString():
            print("Configurations for K-Nearest")
            print("\t\tValue of K: ", k)

        self.parameters["X"] = self.img_data
        self.parameters["y"] = self.img_orientation

        print("K-Nearest Neighbour Training Running Time: {0}".format(time.time() - start))

    def adaboost(self):
        stump_cnt = self.config["stumps"]

        def get_pixel_pair(start=0, end=191):
            """
            return random pixel pair
            """
            return (randint(start, end), randint(start, end))

        def generate_decision_stumps(num_stumps):
            """
            generate pool of decision stumps
            """
            decision_stumps = {}
            orientations = ('0', '90')
            alt = ('180', '270')
            ops = (le, ge)

            for stump in range(num_stumps):
                decision_stumps[stump] = {'pixels': get_pixel_pair(),
                                          'label': choice(orientations),
                                          'alt': choice(alt),
                                          'op': choice(ops),
                                          'weight': 0,
                                          'used': False,
                                          'id': stump}
            return decision_stumps

        def calc_error(predicted, actual):
            label_pairs = list(zip(predicted, actual))
            return round(sum([label[0] != label[1] for label in label_pairs]) / len(predicted), 3)

        def update_incorrect_indices(predicted, actual, incorrect_indices):
            new_indices = set([i for i, item in enumerate(list(zip(predicted, actual))) if item[0] == item[1]])
            return incorrect_indices - new_indices

        def update_weights(curr_weights, error, incorrect_indices):
            # new_weights = list(map(lambda x: x*(error/(1-error)), curr_weights))
            new_weights = [w * (error / (1 - error)) if i not in incorrect_indices else w for i, w in
                           enumerate(curr_weights)]
            norm_const = sum(new_weights) + 0.1
            return list(map(lambda x: x / norm_const, new_weights))

        def eval_stump(predicted, actual, weights, incorrect_indices):
            label_pairs = list(zip(predicted, actual))
            return sum([weights[ind] for ind in incorrect_indices if label_pairs[ind][0] == label_pairs[ind][1]])

        def predict_labels(stump):
            predictions = []
            images = self.img_data.tolist()
            for image in images:
                pixel1, pixel2 = stump['pixels']
                if stump['op'](image[pixel1], image[pixel2]):
                    predictions.append(stump['label'])
                else:
                    predictions.append(stump['alt'])
            return predictions

        def build_boosted_stumps(num_stumps):
            stumps = generate_decision_stumps(num_stumps)
            weights = [1 / self.img_data.shape[0]] * self.img_data.shape[0]
            incorrect_indices = set([i for i in range(self.img_data.shape[0])])

            for i in range(num_stumps):
                best_stump = (0, 0, 0, [])
                for stump_id, stump in stumps.items():
                    if stump['used']:
                        continue

                    predicted_labels = predict_labels(stump)
                    error = calc_error(predicted=predicted_labels, actual=self.img_labels)
                    stump['weight'] = np.log((1 - error) / error + 1)
                    stump_value = eval_stump(predicted=predicted_labels,
                                             actual=self.img_labels,
                                             weights=weights,
                                             incorrect_indices=incorrect_indices)

                    if stump_value >= best_stump[1]:
                        best_stump = (stump['id'], stump_value, error, predicted_labels)
                stumps[best_stump[0]]['used'] = True
                weights = update_weights(curr_weights=weights,
                                         error=best_stump[2],
                                         incorrect_indices=incorrect_indices)
                incorrect_indices = update_incorrect_indices(predicted=best_stump[3],
                                                             actual=self.img_labels,
                                                             incorrect_indices=incorrect_indices)
            self.parameters["stumps"] = stumps

        build_boosted_stumps(stump_cnt)
        print("Adaboost Training Running Time: {0}".format(time.time() - start))

    def nnet(self):
        iterations = self.config["iterations"]
        alpha = self.config["learning_rate"]

        inputNeurons = self.img_data.shape[1]
        hiddenneurons = self.config["hidden_neurons"]
        outputNeurons = self.img_orientation.shape[1]

        weight_hidden = np.random.uniform(size=(inputNeurons, hiddenneurons))
        bias_hidden = np.random.uniform(size=(1, hiddenneurons))
        weight_output = np.random.uniform(size=(hiddenneurons, outputNeurons))
        bias_output = np.random.uniform(size=(1, outputNeurons))

        def toString():
            print("Configurations for Neural Net")
            print("\t\tIterations: ", iterations)
            print("\t\tAlpha: ", alpha)
            print("\t\tHidden Neurons: ", hiddenneurons)

        def sigmoid(x):
            return np.divide(1, (1 + np.exp(-x)))

        def d_sigmoid(x):
            sig = sigmoid(x)
            return sig * (1 - sig)

        def softmax(x):

            max_per_row = np.amax(x, axis=1)
            max_per_row = max_per_row.reshape((max_per_row.shape[0], 1))
            y = x - max_per_row
            probs = np.exp(y) + 0.001
            probs_sum = np.sum(probs, axis=1, keepdims=True)
            probs_sum = np.reshape(probs_sum, (probs_sum.shape[0], 1))
            return probs / probs_sum

        def d_softmax(x):
            sftmax = softmax(x)
            return sftmax * (1 - sftmax)

        toString()
        for i in range(iterations):
            Z1 = sigmoid((np.dot(self.img_data, weight_hidden)) + bias_hidden)
            Z2 = softmax((np.dot(Z1, weight_output)) + bias_output)

            err = self.img_orientation - Z2

            d_output = err * d_softmax(Z2)
            e_hidden = np.dot(d_output, np.transpose(weight_output))

            d_hidden = e_hidden * d_sigmoid(Z1)

            weight_output += np.dot(np.transpose(Z1), d_output) * alpha
            bias_output += np.sum(d_output, axis=0, keepdims=True) * alpha

            weight_hidden += np.dot(np.transpose(self.img_data), d_hidden) * alpha
            bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * alpha

        self.parameters["weight_hidden"] = weight_hidden
        self.parameters["bias_hidden"] = bias_hidden
        self.parameters["weight_output"] = weight_output
        self.parameters["bias_output"] = bias_output

        print("Neural Network Training Running Time: {0}".format(time.time() - start))

    def train(self):
        if self.model == 'nearest':
            self.nearest()

        if self.model == 'adaboost':
            self.adaboost()

        if self.model == 'nnet':
            self.nnet()

        self.save_model()


if __name__ == '__main__':
    start = time.time()
    config = {
        "config_nearest": {
            "k": 35
        },

        "config_adaboost": {
            "stumps": 20
        },

        "config_nnet": {
            "iterations": 1000,
            "hidden_neurons": 50,
            "learning_rate": 0.0001
        }
    }
    if sys.argv[4] == "best":
        sys.argv[4] = "nnet"

    if sys.argv[1] == "train":
        classifier = train(train_filepath=sys.argv[2], model=sys.argv[4], config=config,  model_filepath=sys.argv[3])
        classifier.toString()
        classifier.train()

    if sys.argv[1] == "test":
        classifier = test(test_filepath=sys.argv[2], model=sys.argv[4], model_filepath=sys.argv[3], output_filepath="output.txt")
        classifier.toString()
        classifier.test()

    print("Total Time: {0}".format(time.time() - start))
