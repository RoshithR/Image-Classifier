import sys
import numpy as np
import time
import matplotlib.pyplot as mpt
import pickle
import os
from operator import itemgetter


class test:
    def __init__(self, test_filepath, model, model_filepath="model_file.txt"):
        self.test_filepath = test_filepath
        self.model_filepath = model_filepath
        self.model = model
        self.img_name = []
        self.img_orientation = []
        self.img_data = []

        self.config = {}
        self.parameters = {}

        self.load_model()
        self.loadData()

    def loadData(self):
        f = open(self.test_filepath)
        line = f.readline()
        count = 0
        while line and count != 1000:
            count += 1
            data = line.split(" ")
            self.img_name.append(data[0])
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

        print("K-Nearest Neighbour Test Accuracy: ", (100 * accuracy / self.img_data.shape[0]))
        print("K-Nearest Neighbour Test Running Time: {0}".format(time.time() - start))

    def adaboost(self):
        print("Adaboost yet to be implemented. Come back Later")

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

        Z1 = sigmoid((np.dot(self.img_data, weight_hidden)) + bias_hidden)
        ans = sigmoid((np.dot(Z1, weight_output)) + bias_output)
        for j in range(self.img_data.shape[0]):
            if np.argmax(self.img_orientation[j]) == np.argmax(ans[j]):
                accuracy += 1
        print("Neural Network Test Accuracy: ", (100 * accuracy / self.img_data.shape[0]))
        print("Neural Network Test Running Time: {0}".format(time.time() - start))

    def test(self):
        if self.model == 'nearest':
            self.nearest()

        if self.model == 'adaboost':
            self.adaboost()

        if self.model == 'nnet':
            self.nnet()


class train:
    def __init__(self, train_filepath, model_filepath,  config, model="model_file.txt"):
        self.train_filepath = train_filepath
        self.model_filepath = model_filepath
        self.model = model
        self.img_name = []
        self.img_orientation = []
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
        while line and count != 1000:
            count += 1
            data = line.split(" ")
            self.img_name.append(data[0])
            if data[1] == '0':
                self.img_orientation.append([1, 0, 0, 0])
            elif data[1] == '90':
                self.img_orientation.append([0, 1, 0, 0])
            elif data[1] == '180':
                self.img_orientation.append([0, 0, 1, 0])
            elif data[1] == '270':
                self.img_orientation.append([0, 0, 0, 1])

            # if data[1] == '0':
            #     self.img_orientation.append([1, -1, -1, -1])
            # elif data[1] == '90':
            #     self.img_orientation.append([-1, 1, -1, -1])
            # elif data[1] == '180':
            #     self.img_orientation.append([-1, -1, 1, -1])
            # elif data[1] == '270':
            #     self.img_orientation.append([-1, -1, -1, 1])

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

        def euclidean_Distance(x, y):
            return np.sqrt(np.sum(np.power([m - n for m, n in zip(x, y)], 2)))

        def learn():
            accuracy = 0
            for i in range(self.img_data.shape[0]):
                dist = []

                for j in range(self.img_data.shape[0]):
                    dist.append([self.img_orientation[j], euclidean_Distance(self.img_data[i], self.img_data[j])])
                dist = sorted(dist, key=itemgetter(1))

                ans = [0, 0, 0, 0]
                for j in range(k):
                    ans[np.argmax(dist[j][0])] += 1

                if np.argmax(self.img_orientation[i]) == np.argmax(ans):
                    accuracy += 1

            print("K-Nearest Neighbour Training Accuracy: ", (100 * accuracy / self.img_data.shape[0]))

        if self.img_data.shape[0] < 1000:
            learn()
        print("K-Nearest Neighbour Training Running Time: {0}".format(time.time() - start))

        self.parameters["X"] = self.img_data
        self.parameters["y"] = self.img_orientation

    def adaboost(self):
        D = np.ones(shape=self.img_data.shape[0]) / self.img_data.shape[0]

        def toString():
            print("Configurations for Adaboost")
            print("\t\tInitial D: ", D)
            print("\t\tShape of D: ", D.shape)
            print("\t\tIterations: ", self.config["iterations"])

        def h(x):
            ans_b = []
            b_orientation = {}
            accuracy = 0
            for i in range(x.shape[0]):
                d = x[i]
                b = d[2::3].reshape((8, 8))

                b_orientation["left"] = np.sum(b[:, 0])
                b_orientation["right"] = np.sum(b[:, -1])
                b_orientation["top"] = np.sum(b[0, :])
                b_orientation["bottom"] = np.sum(b[-1, :])

                Max = max(b_orientation, key=b_orientation.get)
                if Max == 'top':
                    ans_b.append(1)
                else:
                    ans_b.append[-1]

            if np.argmax(self.img_orientation[i]) == np.argmax(ans_b[i]):
                    accuracy += 1

            ans_g = []
            g_orientation = {}

            for i in range(x.shape[0]):
                d = x[i]
                g = d[1::3].reshape((8, 8))

                g_orientation["left"] = np.sum(g[:, 0])
                g_orientation["right"] = np.sum(g[:, -1])
                g_orientation["top"] = np.sum(g[0, :])
                g_orientation["bottom"] = np.sum(g[-1, :])

                Max = max(g_orientation, key=g_orientation.get)
                if Max == 'bottom':
                    ans_g.append(D[i] * 1)
                else:
                    ans_g.append[D[i] * -1]


        #
        # def euclidean_Distance(x, y):
        #     return np.sqrt(np.sum(np.power([m - n for m, n in zip(x, y)], 2)))
        #
        # def h2(x):
        #     hx = []
        #     accuracy = 0
        #     for i in range(x.shape[0]):
        #         dist = []
        #         for j in range(1000):
        #             k = np.random.randint(self.img_data.shape[0])
        #             dist.append([self.img_orientation[k], euclidean_Distance(x[i],  self.img_data[k])])
        #
        #         dist = sorted(dist, key=itemgetter(1))
        #
        #         ans = [0, 0, 0, 0]
        #         for j in range(len(dist)):
        #             ans[np.argmax(dist[j][0])] += 1
        #
        #         if np.argmax(self.img_orientation[i]) == np.argmax(ans):
        #             accuracy += 1
        #     print("Adaboost Training Accuracy: ", (100 * accuracy / self.img_data.shape[0]))
        #


        # toString()


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

        error = np.zeros(shape=iterations)

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

        # toString()
        for i in range(iterations):
            Z1 = sigmoid((np.dot(self.img_data, weight_hidden)) + bias_hidden)
            Z2 = sigmoid((np.dot(Z1, weight_output)) + bias_output)

            err = self.img_orientation - Z2

            d_output = err * d_sigmoid(Z2)
            e_hidden = np.dot(d_output, np.transpose(weight_output))

            d_hidden = e_hidden * d_sigmoid(Z1)

            weight_output += np.dot(np.transpose(Z1), d_output) * alpha
            bias_output += np.sum(d_output, axis=0, keepdims=True) * alpha

            weight_hidden += np.dot(np.transpose(self.img_data), d_hidden) * alpha
            bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * alpha

            error[i] = np.sum(np.sqrt(sum(np.power(err, 2))))

            # Z1 = sigmoid((np.dot(self.img_data, weight_hidden)) + bias_hidden)
            # Z2 = (np.dot(Z1, weight_output)) + bias_output
            # probs = np.exp(Z2)
            # probs /= np.sum(probs, axis=1, keepdims=True)
            #
            # # err = np.sum(-np.log(probs[range(self.img_data.shape[0]), self.img_orientation]))
            # print(self.img_orientation * np.log(probs))
            # err = - np.sum(self.img_orientation * np.log(probs))
            #
            #
            # d_output = Z2 - self.img_orientation
            # # d_output = probs
            # # d_output[range(self.img_data.shape[0], self.img_orientation)] -= 1
            # e_hidden = np.dot(d_output, np.transpose(weight_output))
            #
            # d_hidden = e_hidden * d_sigmoid(Z1)
            #
            # weight_output += np.dot(np.transpose(Z1), d_output) * alpha
            # bias_output += np.sum(d_output, axis=0, keepdims=True) * alpha
            #
            # weight_hidden += np.dot(np.transpose(self.img_data), d_hidden) * alpha
            # bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * alpha
            #
            # error[i] = err

        Z1 = sigmoid((np.dot(self.img_data, weight_hidden)) + bias_hidden)
        ans = sigmoid((np.dot(Z1, weight_output)) + bias_output)
        accuracy = 0
        for j in range(self.img_data.shape[0]):
            if np.argmax(self.img_orientation[j]) == np.argmax(ans[j]):
                accuracy += 1
        print("Neural Network Training Accuracy: ", (100 * accuracy / self.img_data.shape[0]))

        self.parameters["weight_hidden"] = weight_hidden
        self.parameters["bias_hidden"] = bias_hidden
        self.parameters["weight_output"] = weight_output
        self.parameters["bias_output"] = bias_output

        print("Neural Network Training Running Time: {0}".format(time.time() - start))
        # mpt.plot(range(iterations), error)
        # mpt.show()

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
            "iterations": 1000
        },

        "config_nnet": {
            "iterations": 1000,
            "hidden_neurons": 5,
            "learning_rate": 0.0001
        }
    }

    # test test-data.txt model-file.txt nnet
    if sys.argv[1] == "train":
        classifier = train(train_filepath=sys.argv[2], model=sys.argv[4], config=config,  model_filepath=sys.argv[3])
        classifier.toString()
        classifier.train()

        # pickled_config = pickle.load(open(sys.argv[3], "rb"))
        # config = pickled_config[sys.argv[4]]["config"]
        # parameters = pickled_config[sys.argv[4]]["parameters"]
        # print("Parameters for ", sys.argv[4], ":")
        # print("\t\tConfiguration: ", config)
        # print("\t\tParameters: ", parameters)

    if sys.argv[1] == "test":
        classifier = test(test_filepath=sys.argv[2], model=sys.argv[4], model_filepath=sys.argv[3])
        classifier.toString()
        # classifier.print_model()
        classifier.test()

    print("Total Time: {0}".format(time.time() - start))
