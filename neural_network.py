from typing import List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
import statistics 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression




def data_split(split_ratio):
    scaler=MinMaxScaler(feature_range=(0.1,0.9))
    data = pd.read_csv('a1_data.txt', delimiter = "\t")
    data.columns = ['a','b','c','d','e']
    data_scaled=scaler.fit_transform(data)
    input_train,input_test,output_train,output_test=train_test_split(data_scaled[:,0:4], data_scaled[:,4], test_size=split_ratio, random_state=0)
    return input_train,output_train,input_test,output_test



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_der(x):
    return x * (1 - x)
### used the pow beacuse of the round option 
def square_error(y, y_p):
    return round(pow((y - y_p), 2).item(), 10)


def error(y, y_p):
    return abs(round((y - y_p).item(), 10))

### this data structure as there is main layer class and each of the input, hidden and output layers inherit from it with the needed modification
class Layer:
    def __init__(self, nodes, layer_type):
        self.activations = np.zeros(nodes) # g(h)
        self.layer_type: str = layer_type
        self.nodes: int = nodes


class InputLayer(Layer):
    ### the use of super is good when inhertin to prevent the class calling 
    def __init__(self, nodes):
        super().__init__(nodes, 'input')

    
class HiddenLayer(Layer):
    
    def __init__(self, nodes):
        super().__init__(nodes, 'hidden')
        self.thresholds = np.random.rand(nodes)
        self.thresholds_inc = np.zeros(nodes)
        self.deltas = np.zeros(nodes)



class OutputLayer(Layer):
    
    def __init__(self, nodes):
        super().__init__(nodes, 'output')
        self.thresholds = np.random.rand(nodes)
        self.thresholds_inc = np.zeros(nodes)
        self.deltas = np.zeros(nodes)




class NeuralNetwork:
    
    def __init__(self, nodes_input, nodes_output, hidden_layers, alfa=0.1):

        self.layers = self.create_layers(nodes_input, nodes_output, hidden_layers)
        self.weights = self.create_weights()
        self.weights_incr = self.create_weights_incr()

        self.input = np.zeros(nodes_input)
        self.output = np.zeros(nodes_output)
        self.label = np.zeros(nodes_output)
        self.alfa = alfa

    def create_layers(self, nodes_input, nodes_output, hidden_layers):
        layers = []
        input_layer = InputLayer(nodes=nodes_input)
        output_layer = OutputLayer(nodes=nodes_output)
        
        # append all layers 
        layers.append(input_layer)
        # in the hidden layer the number of nodes are equal to the number of input nodes
        for times in range(hidden_layers):
            layers.append(HiddenLayer(nodes=nodes_input))
        layers.append(output_layer)
        return layers

    def create_weights(self):
        layers = []
        for index in range(len(self.layers)-1): 
            layer = np.random.rand(self.layers[index].nodes, self.layers[index+1].nodes)
            layers.append(layer)
        return layers

    def create_weights_incr(self):
        layers = []
        for index in range(len(self.layers)-1): 
            layer = np.zeros((self.layers[index].nodes, self.layers[index+1].nodes))
            layers.append(layer)
        return layers

    

    
    def load_data(self, input, label=None):
        
        self.input = input
        #later for predicting condition when it only takes input
        if (label is not None):
            self.label = label

    def feed_forward(self):
        prev_layer = None
        weight_index = 0
        for layer in self.layers:
            
            if layer.layer_type == 'input':
                layer.activations = self.input
                prev_layer = layer
            else: 
                for index in range(layer.nodes):
                    
                    weights_j_neuron = np.array(self.weights[weight_index][:,index])
                    
                    raw_activation = np.dot(prev_layer.activations, weights_j_neuron) - layer.thresholds[index]
                    # apply sigmoid
                    layer.activations[index] = sigmoid(raw_activation)

                prev_layer = layer
                weight_index += 1

        self.output = self.layers[-1].activations

    def back_propagate(self):
        self.compute_deltas()
        self.update_weights_thresholds()


    def compute_deltas(self):
        prev_layer = None
        weight_index = len(self.weights) - 1
        for layer in reversed(self.layers): 
            if layer.layer_type == 'output':
                layer.deltas = sigmoid_der(layer.activations)*(layer.activations - self.label)
                prev_layer = layer

            elif layer.layer_type == 'input':
                # no deltas for input layer
                pass

            else: 
                # hidden layer
                deltas = np.zeros(layer.nodes)

                for index in range(layer.nodes):
                    
                    weights_i_neuron = np.array(self.weights[weight_index][index,:])
                    
                    deltas[index] = np.dot(prev_layer.deltas, weights_i_neuron) * sigmoid_der(layer.activations[index])

                layer.deltas = deltas
                prev_layer = layer
                weight_index -= 1
    
    def update_weights_thresholds(self):
        weight_layers = range(len(self.weights))
        for weight_index in reversed(weight_layers): 
            index_layer = weight_index
            layer = self.layers[index_layer]
            prev_layer = self.layers[index_layer + 1]
            for index in range(layer.nodes):
                # m
                prev_weights_incr = self.weights_incr[weight_index][index,:]
                
                self.weights_incr[weight_index][index,:] = ( (-1)  * layer.activations[index] * prev_layer.deltas ) + ( self.alfa * prev_weights_incr )
            
            self.weights[weight_index] += self.weights_incr[weight_index] 
            prev_thresholds_inc = prev_layer.thresholds_inc
            prev_layer.thresholds_inc = (prev_layer.deltas) + (self.alfa * prev_thresholds_inc)
            prev_layer.thresholds += prev_layer.thresholds_inc

    def train(self, input, label):
        self.load_data(input, label)
        self.feed_forward()
        self.back_propagate()
        return error(self.output, label)

    
    def predict(self, input):
        self.load_data(input)
        self.feed_forward()
        return self.output
    
    def validate(self, input, label):
        self.load_data(input, label)
        self.feed_forward()
        return (error(self.output, label), self.output)
           


if __name__ == "__main__": 

    np.random.seed(10) 
    input,output,input2,output2 = data_split(0.2)
    

    Lr = 0.7
    epoch=300
    hidden_layer=4
    

    nn = NeuralNetwork(4, 1, hidden_layer, Lr)
    last_loss = []
    for i in range(epoch):
        
        loss = []
        for i in range(len(input)):
            step_loss = nn.train(input[i], output[i])
            loss.append(step_loss)
        last_loss.append(loss[-1])
    print("Training Done")
    print("=================")
    #for element in last_loss:
    #    print("epoch loss", element)
    print("mean loss", statistics.mean(last_loss))
    print("NN Predicted values")
    NNpredicted=[]
    for i in range(len(input2)):
        #print(nn.predict(input2[i]))
        NNpredicted.append(np.array(nn.predict(input2[i])))
    for i in range(len(NNpredicted)):
        print(NNpredicted[i])

    print("real values")
    for i in range(len(output2)):
        print(output2[i])
    
## Cross validation part
    kf = KFold(n_splits=4)
    kf.get_n_splits(input)
    
    count=1
    for train_index, test_index in kf.split(input): 
        
        X_train, X_test = input[train_index], input[test_index]
        y_train, y_test = output[train_index], output[test_index]

        nn = NeuralNetwork(4, 1, hidden_layer, Lr)
        last_loss = []
        for i in range(epoch):
            loss = []
            for i in range(len(X_train)):
                step_loss = nn.train(X_train[i], y_train[i])
                loss.append(step_loss)
            last_loss.append(loss[-1])
        print("Training Done for CV number ",count)
        print("=================")
        #for element in last_loss:
        #    print("epoch loss", element)
        print("mean loss of CV",count)
        print(statistics.mean(last_loss))
        print("=================")
        print("mean error with testing for CV", count)
        valid=[]
        for i in range(len(X_test)):
            v,o = nn.validate(X_test[i], y_test[i])
            valid.append(v)
        print(statistics.mean(valid))
        print("=================")
        count+=1

    
    plt.plot(output2, output2, color='red', linewidth=0.2)
    plt.scatter(output2,output2, marker='*', label="Real")
    
    plt.scatter(NNpredicted, output2, marker='^', label="NN")
    plt.legend()
    plt.xlabel('Predicted values')
    plt.ylabel('Real values')
    plt.savefig('predicted_NN_vs_real3.png')

    MLR = LinearRegression()


    MLR.fit(input, output)

    # test
    predicted_MLR = MLR.predict(input2)

    # error
    
    plt.plot(output2, output2, color='black', linewidth=0.2)
    #plt.scatter(output2, output2, marker='s', label="Real")
    plt.scatter(predicted_MLR, output2, marker='^', label="MLR")
    plt.legend()
    plt.xlabel('Predicted values')
    plt.ylabel('Real values')
    plt.savefig('MLR and Real 3.png')



# %%
