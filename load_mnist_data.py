import numpy as np
from settings import *

class CustomDataLoader:

    def __init__(self, data_dir=None, image_size=None, label_num=None, seed=None):
        self.data_dir = data_dir if data_dir is not None else DATA_DIR_PATH
        self.image_size = image_size if image_size is not None else IMAGE_SIZE
        self.label_num = label_num if label_num is not None else LABEL_NUM
        self.seed = seed if seed is not None else SEED
        self.pixel_size = self.image_size * self.image_size
        self.train_data = np.loadtxt(os.path.join(self.data_dir, TRAIN_DATA), delimiter=',', skiprows=1)
        self.test_data = np.loadtxt(os.path.join(self.data_dir, TEST_DATA), delimiter=',', skiprows=1)

    def one_hot_encode(self, y):
        '''
        Creates one-hot vector representation for each label.
        Subtracting these labels from the output of the neural network is more subtle
        ---
        label:  0  in one-hot representation:  [1 0 0 0 0 0 0 0 0 0]
        label:  1  in one-hot representation:  [0 1 0 0 0 0 0 0 0 0]
        label:  2  in one-hot representation:  [0 0 1 0 0 0 0 0 0 0]
        label:  3  in one-hot representation:  [0 0 0 1 0 0 0 0 0 0]
        label:  4  in one-hot representation:  [0 0 0 0 1 0 0 0 0 0]
        label:  5  in one-hot representation:  [0 0 0 0 0 1 0 0 0 0]
        label:  6  in one-hot representation:  [0 0 0 0 0 0 1 0 0 0]
        label:  7  in one-hot representation:  [0 0 0 0 0 0 0 1 0 0]
        label:  8  in one-hot representation:  [0 0 0 0 0 0 0 0 1 0]
        label:  9  in one-hot representation:  [0 0 0 0 0 0 0 0 0 1]
        ---
        Params:
        ---
        y: input of labels

        Returns:
        ---
        One-hot encoded vectors for labels
        '''
        encoded = np.zeros((10, 1))
        encoded[y] = 1.0
        return encoded


    def preprocess_data(self):
        '''
        First images and labels are extracted from each train and test data.
        Then train_images & train_labels are randomly shuffled.
        Finally, X and y values of each data are reshaped into list and one-hot
        encoded vectors respectively.
        Returns:
        ---
        Tuple of zipped objects for tuples of training_data and testing_data
        '''
        train_images = np.asfarray(self.train_data[:, 1:]) * SCALE_FACTOR
        test_images = np.asfarray(self.test_data[:, 1:]) * SCALE_FACTOR

        train_labels = np.asfarray(self.train_data[:, :1])
        test_labels = np.asfarray(self.test_data[:, :1])

        np.random.seed(self.seed)
        shuffle_index = np.random.permutation(train_images.shape[0])
        train_images, train_labels = train_images[shuffle_index], train_labels[shuffle_index]

        # Transforming X values into a list of pixel_size i.e. 28*28=784 if image_size=28
        X_train = [np.reshape(x, (self.pixel_size, 1)) for x in train_images]
        X_test = [np.reshape(x, (self.pixel_size, 1)) for x in test_images]

        # Transforming y values into their one-hot encoded vector form
        y_train = [self.one_hot_encode(y) for y in train_labels.astype(int)]
        y_test = [self.one_hot_encode(y) for y in test_labels.astype(int)]

        # Zipping the training data and testing data
        training_data = zip(X_train, y_train)
        testing_data = zip(X_test, y_test)

        return (training_data, testing_data)


# loader = CustomDataLoader()
# train_data, test_data = loader.preprocess_data()

# print(list(train_data)[1][:2])