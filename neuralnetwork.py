import numpy
import scipy
import matplotlib.pyplot as plt
import scipy.misc
import glob

class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #노드 개수 설정
        
        self.lr = learningrate
        #학습률
        
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        #input 에서 hidden으로의 weight
        
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        #hidden 에서 output으로의 weight
        
        self.activation_function = lambda x: scipy.special.expit(x)
        #활성화 함수(다음 레이어로 전달되기위한 조건)
        
    def train(self, inputs_list, targets_list):
        
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        #전치행렬로 만든다
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        #hidden으로 들어오는 input
        hidden_outputs = self.activation_function(hidden_inputs)
        #hidden에서 나오는 output
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        #final로 들어오는 input
        final_outputs = self.activation_function(final_inputs)
        #final에서 나오는 output
        
        output_errors = targets- final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        #계산된 결과
        
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0-final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)), numpy.transpose(inputs))
        #weight의 조정
        
    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

input_nodes=784
hidden_nodes=200
output_nodes=10
learning_rate=0.1
n=neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

print('open')
training_data_file=open("mnist_train.csv", 'r')
training_data_list=training_data_file.readlines()
training_data_file.close()

print('close')

epochs = 5
for e in range(epochs):
    for train_record in training_data_list:
        all_values = train_record.split(',')
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])]=0.99
        n.train(inputs, targets)
    print(e)

test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for record in test_data_list:
    all_values = record.split(',')
    
    correct_label = int(all_values[0])
    print (correct_label, "correct label")
    
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #값이 0에서 255여서 0과 1사이의 값으로만 만들기위해서 필요하다
    
    outputs = n.query(inputs)
    
    label = numpy.argmax(outputs)
    
    print (label, "network's answer")
    
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        
scorecard_array = numpy.asarray(scorecard)
print ("Performance =", scorecard_array.sum() / scorecard_array.size)

own_dataset = []

for image_file_name in glob.glob('neural/?.png'):
     label = int(image_file_name[-5:-4])
     print (label)
     img_array = scipy.misc.imread(image_file_name, flatten = True)
     img_data = 255.0 - img_array.reshape(784)
     #MNIST에서는 255이 검은색이고 0이 흰색을 나타내서 이런식으로 뺴줘야된다
     img_data = (img_data / 255.0 * 0.99) + 0.01
     print (numpy.min(img_data))
     print (numpy.max(img_data))
     
     record = numpy.append(label, img_data)
     own_dataset.append(record)
     
number = 3

for item in range(number):
    plt.imshow(own_dataset[item][1:].reshape(28,28), cmap = 'Greys', interpolation = 'None')
    correct_label = own_dataset[item][0]
    inputs = own_dataset[item][1:]

    outputs = n.query(inputs)
    print(outputs)

    label = numpy.argmax(outputs)
    print("network thinks ", label)

    if (label == correct_label):
        print("match")
    else:
        print("no match")
