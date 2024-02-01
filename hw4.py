import numpy as np

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []
    
    def hFunc(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.theta)))
    
    def jFunc(self, X, y):
        h = self.hFunc(X)
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           
        ###########################################################################
        bias = np.ones((X.shape[0], 1))
        XWithBias = np.concatenate((bias, X), axis=1)
        XSize = XWithBias.shape[1]
        self.theta = np.random.rand(XSize)
        difference = np.inf

        for i in range(self.n_iter):
            self.thetas.append(self.theta)
            self.Js.append(self.jFunc(XWithBias, y))
            if i > 0:
                difference = self.Js[i-1] - self.Js[i]
            if difference < self.eps:
                break
            self.theta = self.theta - self.eta * np.dot(XWithBias.T, self.hFunc(XWithBias) - y) / XWithBias.shape[0]
            
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        ###########################################################################                                         #
        # TODO: Implement the function.   
        ###########################################################################
        bias = np.ones((X.shape[0], 1))
        XWithBias = np.concatenate((bias, X), axis=1)
        
        for i in range(XWithBias.shape[0]):
            if self.hFunc(XWithBias[i]) >= 0.5:
                preds.append(1)
            else:
                preds.append(0)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    Xy = np.concatenate((X, y.reshape(-1, 1)), axis=1)
    np.random.shuffle(Xy)
    X = Xy[:, :-1]
    y = Xy[:, -1]
    X = np.array_split(X, folds)
    y = np.array_split(y, folds)
    accuracy = []

    for i in range(folds):
        Xtrain = np.concatenate(X[:i] + X[i+1:], axis=0)
        ytrain = np.concatenate(y[:i] + y[i+1:], axis=0)
        Xtest = X[i]
        ytest = y[i]
        algo.fit(Xtrain, ytrain)
        preds = algo.predict(Xtest)
        accuracy.append(np.mean(preds == ytest))

    cv_accuracy = np.mean(accuracy)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy

def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # Calculate normal desnity function for a given data, mean and standrad deviation.
    m1 = 1 / (np.sqrt(2 * np.pi) * sigma)
    m2 = np.exp(-0.5 * np.square((data - mu) / sigma))
    p = m1 * m2
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = []
        self.weights = []
        self.mus = []
        self.sigmas = []
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        splittedData = np.array_split(data, self.k)

        for i in range(self.k):
            self.weights.append(1 / self.k)
            self.mus.append(np.mean(splittedData[i]))
            self.sigmas.append(np.std(splittedData[i]))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for i in range(data.shape[0]):
            for j in range(self.k):
                self.responsibilities[i][j] = self.weights[j] * norm_pdf(data[i], self.mus[j], self.sigmas[j])
            self.responsibilities[i] /= np.sum(self.responsibilities[i])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for i in range(self.k):
            self.weights[i] = np.mean(self.responsibilities[:, i])
            self.mus[i] = np.sum(self.responsibilities[:, i] * data) / np.sum(self.responsibilities[:, i])
            self.sigmas[i] = np.sqrt(np.sum(self.responsibilities[:, i] * (data - self.mus[i]) ** 2) / np.sum(self.responsibilities[:, i]))
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        cost = []
        prev_cost = np.zeros(self.k)
        diff = np.infty

        self.init_params(data)
        for iteration in range(self.n_iter):
            if diff <= self.eps:
                break
            self.responsibilities = np.zeros((data.shape[0], self.k))
            self.expectation(data)
            self.maximization(data)
            for i in range(self.k):
                current_cost = self.cost(data, i)
                cost.append(current_cost)

            diff = np.max([abs(np.array(current_cost) - np.array(prev_cost))])
            prev_cost = current_cost
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas
    
    def cost(self, data, i):
        return -np.sum(np.log(np.sum(self.responsibilities[i])))
    

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for i in range(len(weights)):
        pdf += weights[i] * norm_pdf(data, mus[i], sigmas[i])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = np.zeros(2)
        self.EM1 = [EM(k=self.k) for i in range(2)]
        self.EM2 = [EM(k=self.k) for i in range(2)]

    def get_prior(self, X, y):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        self.prior[0] = (len(X[np.where(y[:,-1] == 0)])) / len(X)
        self.prior[1] = (len(X[np.where(y[:,-1] == 1)])) / len(X)

            
    def get_instance_likelihood(self, X, j):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood_1 = 0
        likelihood_2 = 0
        for i in range(self.k):
            likelihood_1 += norm_pdf(X[0], self.EM1[j].mus[i] , self.EM1[j].sigmas[i]) * self.EM1[j].weights[i]
            likelihood_2 += norm_pdf(X[1], self.EM2[j].mus[i] , self.EM2[j].sigmas[i]) * self.EM2[j].weights[i]
            
        return likelihood_1 * likelihood_2
    
    def get_instance_posterior(self, x, i):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        return self.prior[i] * self.get_instance_likelihood(x, i)

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # calculate prior
        EM_fit1 = lambda i, label : self.EM1[i].fit(X[np.where(y[:,-1] == i)][:,label])
        EM_fit2 = lambda i, label : self.EM2[i].fit(X[np.where(y[:,-1] == i)][:,label])
        
        y = np.column_stack([np.zeros_like(y) ,y])
        self.get_prior(X,y)
        
        EM_fit1(0, 0)
        EM_fit2(0, 1)
        EM_fit1(1, 0)
        EM_fit2(1, 1)

        
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        prediction = [] 
        for i in range(len(X)):
            predict_0 = self.get_instance_posterior(X[i], 0)
            predict_1 = self.get_instance_posterior(X[i], 1)
            if(predict_0 > predict_1):
                prediction.append(0)
            else:
                prediction.append(1)
            
        return np.asarray(prediction)

def calc_accuracy(X, y, model):
    acc = model.predict(X) - y
    class_unique, class_counts = np.unique(acc, return_counts=True)
    index = np.where(class_unique == 0)[0]
    return (class_counts[index] / len(y)) * 100


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    logistic_regression_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_regression_model.fit(x_train, y_train)
    naive_bayes_model = NaiveBayesGaussian(k=k)
    naive_bayes_model.fit(x_train, y_train)
    lor_train_acc = calc_accuracy(x_train, y_train, logistic_regression_model)[0]
    bayes_train_acc = calc_accuracy(x_train, y_train, naive_bayes_model)[0]
    lor_test_acc = calc_accuracy(x_test, y_test, logistic_regression_model)[0]
    bayes_test_acc = calc_accuracy(x_test, y_test, naive_bayes_model)[0]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.

    1. In this homework we explored two types of models: Naive Bayes using EM, and Logistic regression.  
       1. Generate a dataset (`dataset_a`), in 3 dimensions (3 features), with 2 classes, using **only** Multivariate-Gaussians (as many as you want) such that **Naive Bayes will work better on it when compared to Logisitc Regression**.
       2. Generate another dataset (`dataset_b`), in 3 dimensions (3 features), with 2 classes, using **only** Multivariate-Gaussians (as many as you want) such that **Logistic Regression will work better on it when compared to Naive Bayes**.
   
    2. Visualize the datasets.  
      You can choose one of two options for the visualization:
        1. Plot three 2d graphs of all the features against each other (feature1 vs feature2, feature1 vs feature3, feature2 vs feature3).
        2. Plot one 3d graph.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    # dataset a
    mean1 = [0, 0, 0]
    cov1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mean2 = [1, 1, 1]
    cov2 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    dataset_a_features = np.concatenate((multivariate_normal.rvs(mean1, cov1, 100), multivariate_normal.rvs(mean2, cov2, 100)))
    dataset_a_labels = np.concatenate((np.zeros(100), np.ones(100)))
    # dataset b
    mean1 = [0, 0, 0]
    cov1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mean2 = [0, 0, 0]
    cov2 = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    dataset_b_features = np.concatenate((multivariate_normal.rvs(mean1, cov1, 100), multivariate_normal.rvs(mean2, cov2, 100)))
    dataset_b_labels = np.concatenate((np.zeros(100), np.ones(100)))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }