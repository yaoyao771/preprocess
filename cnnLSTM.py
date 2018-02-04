import lasagne
import theano
import time

import numpy as np
import cPickle as cp
import theano.tensor as T
from sliding_window import sliding_window

# Hardcoded number of sensor channels employed in the OPPORTUNITY challenge
NB_SENSOR_CHANNELS = 113

# Hardcoded number of classes in the gesture recognition problem
NUM_CLASSES = 17

# Hardcoded length of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_LENGTH = 24

# Length of the input sequence after convolutional operations
FINAL_SEQUENCE_LENGTH = 8

# Hardcoded step of the sliding window mechanism employed to segment the data
SLIDING_WINDOW_STEP = 12

# Batch Size
# BATCH_SIZE = 100

# Number filters convolutional layers
NUM_FILTERS = 64

# Size filters convolutional layers
FILTER_SIZE = 5

# Number of unit in the long short-term recurrent layers
NUM_UNITS_LSTM = 128

NUM_EPOCHS = 500


def load_dataset(filename):
    f = file(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test


print("Loading data...")
X_train, y_train, X_test, y_test = load_dataset('data/oppChallenge_locomotion.data')

print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))

if 1 in y_train:
    print("1 in y_train")
else:
    print("1 not in y_train")

assert NB_SENSOR_CHANNELS == X_train.shape[1]


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


# Sensor data is segmented using a sliding window mechanism
X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape, y_test.shape))
print("y_test[100:300] is {} ".format(y_test[100:300]))

# Data is reshaped since the input of the network is a 4 dimension tensor
X_test = X_test.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))

# process the train data
X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LENGTH, SLIDING_WINDOW_STEP)
print(" ..after sliding window (train): inputs {0}, targets {1}".format(X_train.shape, y_train.shape))
# reshape the X_train to suit the input
X_train = X_train.reshape((-1, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))

examLen = int(X_train.shape[0] * 0.7)
X_val = X_train[examLen:]
y_val = y_train[examLen:]

X_train = X_train[:examLen]
y_train = y_train[:examLen]

print("\n\nX_val.shape is {0},\ty_val.shape is {1}".format(X_val.shape, y_val.shape))
print("\nX_train.shape is {0},\ty_train.shape is {1}\n".format(X_train.shape, y_train.shape))

# import pickle
# f=open('train_x.txt','wb')
# pickle.dump(X_train,f)
# f.close()

# for lr in [0.005,0.001]:
#     LEARNING_RATE=lr
#     for bs in [108,256]:
#         BATCH_SIZE=bs
LEARNING_RATE = 0.001
BATCH_SIZE = 108

net = {}
net['input'] = lasagne.layers.InputLayer((BATCH_SIZE, 1, SLIDING_WINDOW_LENGTH, NB_SENSOR_CHANNELS))
net['conv1/5x1'] = lasagne.layers.Conv2DLayer(net['input'], NUM_FILTERS, (FILTER_SIZE, 1))
net['conv2/5x1'] = lasagne.layers.Conv2DLayer(net['conv1/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
net['conv3/5x1'] = lasagne.layers.Conv2DLayer(net['conv2/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
net['conv4/5x1'] = lasagne.layers.Conv2DLayer(net['conv3/5x1'], NUM_FILTERS, (FILTER_SIZE, 1))
net['shuff'] = lasagne.layers.DimshuffleLayer(net['conv4/5x1'], (0, 2, 1, 3))
net['lstm1'] = lasagne.layers.LSTMLayer(net['shuff'], NUM_UNITS_LSTM)
net['lstm2'] = lasagne.layers.LSTMLayer(net['lstm1'], NUM_UNITS_LSTM)
# In order to connect a recurrent layer to a dense layer, it is necessary to flatten the first two dimensions
# to cause each time step of each sequence to be processed independently (see Lasagne docs for further information)
net['shp1'] = lasagne.layers.ReshapeLayer(net['lstm2'], (-1, NUM_UNITS_LSTM))
net['prob1'] = lasagne.layers.DenseLayer(net['shp1'], 1024, nonlinearity=lasagne.nonlinearities.rectify)
net['prob2'] = lasagne.layers.DenseLayer(net['prob1'], 128, nonlinearity=lasagne.nonlinearities.rectify)

net['output'] = lasagne.layers.DenseLayer(net['prob2'], NUM_CLASSES, nonlinearity=lasagne.nonlinearities.softmax)
# Tensors reshaped back to the original shape
# net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'], (-1, FINAL_SEQUENCE_LENGTH, NUM_CLASSES))
# Last sample in the sequence is considered
# net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)

# Tensors reshaped back to the original shape
# net['shp2'] = lasagne.layers.ReshapeLayer(net['prob'], (BATCH_SIZE, FINAL_SEQUENCE_LENGTH, NUM_CLASSES))
# Last sample in the sequence is considered
# net['output'] = lasagne.layers.SliceLayer(net['shp2'], -1, 1)
#
# The model is populated with the weights of the pretrained network
# all_params_values = cp.load(open('weights/DeepConvLSTM_oppChallenge_gestures.pkl'))
# lasagne.layers.set_all_param_values(net['output'], all_params_values)

# get_feature_vector=lasagne.layers.get_output(net['lstm2'])

# Compilation of theano functions
# Obtaining the probability distribution over classes
prediction = lasagne.layers.get_output(net['output'])

print("\nnet['input'].output_shape is {}\n".format(net['input'].output_shape))

print("\nnet['conv4/5x1'].output_shape is {}\n".format(net['conv4/5x1'].output_shape))

print("\nnet['prob2'].output_shape is {}\n".format(net['prob2'].output_shape))

print("net['output'].output_shape is {}".format(net['output'].output_shape))
# net['output'].output_shape is (100, 18)
#
target_values = T.ivector('target_output')

loss = lasagne.objectives.categorical_crossentropy(prediction, target_values)
loss = loss.mean()

# create parameter update expressions
params = lasagne.layers.get_all_params(net['output'], trainable=True)
updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=LEARNING_RATE,
                                            momentum=0.9)
train_fn = theano.function([net['input'].input_var, target_values], loss, updates=updates, on_unused_input='warn')

val_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
val_loss = lasagne.objectives.categorical_crossentropy(val_prediction, target_values)
val_loss = val_loss.mean()

val_acc = T.mean(T.eq(T.argmax(val_prediction, axis=1), target_values),
                 dtype=theano.config.floatX)

val_fn = theano.function([net['input'].input_var, target_values], [val_loss, val_acc], on_unused_input='warn')


# Returning the predicted output for the given minibatch
# test_feature_vector = theano.function([net['input'].input_var],[get_feature_vector])

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# Classification of the testing data
print("Processing {0} instances in mini-batches of {1}".format(X_train.shape[0], BATCH_SIZE))

train_count = 0
for epoch in range(NUM_EPOCHS):
    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    print("the epoch of {} is starting..".format(epoch))
    for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
        inputs, targets = batch
        # print("the inputs.shape is {0}\ttargets.shape is {1}".format(inputs.shape,targets.shape))
        # the inputs.shape is (100, 1, 24, 113)
        # targets.shape is (100,)
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # And a full pass over the validation data:
    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc
        val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, NUM_EPOCHS, time.time() - start_time))
    validation_loss = val_err / val_batches
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(validation_loss))
    print("  validation accuracy:\t\t{:.2f} %".format(
        val_acc / val_batches * 100))

test_prediction = lasagne.layers.get_output(net['output'], deterministic=True)
test_fn = theano.function([net['input'].input_var], [T.argmax(test_prediction, axis=1)], on_unused_input='warn')

test_pred = np.empty((0))
test_true = np.empty((0))
# test_feature = np.empty((0))
# test_feature=np.zeros((100,8,128))
start_time = time.time()
iterate_count = 0
np.savez('model_cnn_gesture.npz', *lasagne.layers.get_all_param_values(net['output']))
for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=False):
    iterate_count = iterate_count + 1
    inputs, targets = batch
    y_pred, = test_fn(inputs)
    # print("y_pred's shape is {}".format(y_pred.shape))
    # y_pred shape is(100,)
    # y_feature,=test_feature_vector(inputs)
    # print("y_feature's shape is {}".format(y_feature.shape))
    # y_feature's shape is (100, 8, 128)
    test_pred = np.append(test_pred, y_pred, axis=0)
    test_true = np.append(test_true, targets, axis=0)
    # test_feature=np.append(test_feature,y_feature,axis=0)
    if iterate_count % 10 == 0:
        print("...\titerate count is {}".format(iterate_count))
        print("now test_pred's shape is {}".format(test_pred.shape))

# save test_true to the file
# file=open('train_true.txt','wb')
# pickle.dump(test_true,file)
# file.close()

# save test_feature to the file
# file_feature=open('train_feature.txt','wb')
# pickle.dump(test_feature,file_feature)
# file_feature.close()

print("... part of test_true:{}".format(test_true[:10]))
print("... part of test_pred:{}".format(test_pred[:10]))
# Results presentation
print("||Results||")
print("\tTook {:.3f}s.".format(time.time() - start_time))
import sklearn.metrics as metrics

f1Score = metrics.f1_score(test_true, test_pred, average='weighted')
accuracyScore = metrics.accuracy_score(test_true, test_pred)
precisionScore = metrics.precision_score(test_true, test_pred, average='weighted')
recallScore = metrics.recall_score(test_true, test_pred, average='weighted')
print("\tTest fscore:\t{:.4f} ".format(f1Score))
with open('cnn_train_gesture.txt', 'a') as f:
    f.write('\n-------\n-lr:' + bytes(LEARNING_RATE) + '-batch_size:' + bytes(BATCH_SIZE) + '-epoch:' + bytes(
        NUM_EPOCHS) + '\n-f1_score is :' + bytes(f1Score) + '-the precision:' + bytes(
        precisionScore) + 'the rescall:' + bytes(recallScore) + 'the accuracy:' + bytes(accuracyScore) + '\r\n')

