import numpy as np

# ============ Activation Functions ============
def ReLU(h):
    return np.maximum(h, 0)

def sigmoid(h):
    return 1 / (1 + np.exp(-h))

def linear(h):
    return h

activation_function_dict = {'ReLU' : ReLU,
                            'sigmoid' : sigmoid,
                            'linear' : linear}

class FeedForwardNeuralNetwork:

    def fit(self, X, y, n_hidden, f1 = 'ReLU', f2 = 'linear', loss = 'RSS', lr = 1e-5, n_iter = 1e3, seed = None):

        # initialize attributes
        self.X = X
        self.y = y.reshape(len(y), -1)
        self.N = self.X.shape[0]
        self.D_X = self.X.shape[1]
        self.D_y = self.y.shape[1]
        self.D_h = n_hidden
        self.f1, self.f2 = f1, f2
        self.loss = loss
        self.lr = lr
        self.n_iter = int(n_iter)
        self.seed = seed

        # set seed
        np.random.seed(self.seed)

        # initialize weights
        self.W1 = np.random.randn(self.D_h, self.D_X) / 5
        self.c1 = np.random.randn(self.D_h, 1) / 5
        self.W2 = np.random.randn(self.D_y, self.D_h) / 5
        self.c2 = np.random.randn(self.D_y, 1) / 5

        # initialize outputs
        self.h1 = np.dot(self.W1, self.X.T) + self.c1
        self.z1 = activation_function_dict[self.f1](self.h1)
        self.h2 = np.dot(self.W2, self.z1) + self.c2
        self.yhat = activation_function_dict[self.f2](self.h2)

        # fit the weights
        for iteration in range(self.n_iter):

            # initialize derivatives
            dL_dW2 = 0
            dL_dc2 = 0
            dL_dW1 = 0
            dL_dc1 = 0

            for n in range(self.N):

                # calculate dL_dyhat based on loss
                if loss == 'RSS':
                    dL_dyhat = -2*(self.y[n] - self.yhat[:, n]).T
                elif loss == 'log':
                    dL_dyhat = (-(self.y[n] / self.yhat[:, n]) + (1 - self.y[n]) / (1 - self.yhat[:, n])).T
                
                # layer 2
                # calculate dyhat_dh2
                if self.f2 == 'linear':
                    dyhat_dh2 = np.eye(self.D_y)
                elif self.f2 == 'sigmoid':
                    dyhat_dh2 = np.diag(sigmoid(self.h2[:, n])*(1 - sigmoid(self.h2[:, n])))
                
                
                dh2_dc2 = np.eye(self.D_y)
                dh2_dW2 = np.zeros((self.D_y, self.D_y, self.D_h))

                for i in range(self.D_y):
                    dh2_dW2[i] = self.z1[:, n]

                dh2_dz1 = self.W2

                # layer 1
                if self.f1 == 'ReLU':
                    dz1_dh1 = 1*np.diag(self.h1[:, n] > 0)
                elif self.f1 == 'linear':
                    dz1_dh1 = np.eye(self.D_h)
                
                dh1_dc1 = np.eye(self.D_h)

                dh1_dW1 = np.zeros((self.D_h, self.D_h, self.D_X))

                for i in range(self.D_h):
                    dh1_dW1[i] = self.X[n]
                
                # update derivatives
                dL_dh2 = dL_dyhat @ dyhat_dh2
                dL_dW2 += dL_dh2 @ dh2_dW2
                dL_dc2 += dL_dh2 @ dh2_dc2
                dL_dh1 = dL_dh2 @ dh2_dz1 @ dz1_dh1
                dL_dW1 += dL_dh1 @ dh1_dW1
                dL_dc1 += dL_dh1 @ dh1_dc1

            # update weights
            self.W1 -= self.lr * dL_dW1
            self.c1 -= self.lr * dL_dc1.reshape(-1,1)
            self.W2 -= self.lr * dL_dW2
            self.c2 -= self.lr * dL_dc2.reshape(-1,1)

            # update outputs
            self.h1 = np.dot(self.W1, self.X.T) + self.c1
            self.z1 = activation_function_dict[self.f1](self.h1)
            self.h2 = np.dot(self.W2, self.z1) + self.c2
            self.yhat = activation_function_dict[self.f2](self.h2)

    def predict(self, X_test):

        self.h1 = np.dot(self.W1, X_test.T) + self.c1
        self.z1 = activation_function_dict[self.f1](self.h1)
        self.h2 = np.dot(self.W2, self.z1) + self.c2
        self.yhat = activation_function_dict[self.f2](self.h2)
        return self.yhat

class FeedForwardMatrix:
    
    def fit(self, X, y, n_hidden, f1 = 'ReLU', f2 = 'linear', loss = 'RSS', lr = 1e-5, n_iter = 5e3, seed = None):

        # initialize attributes
        self.X = X
        self.y = y.reshape(len(y), -1)
        self.N = self.X.shape[0]
        self.D_X = self.X.shape[1]
        self.D_y = self.y.shape[1]
        self.Xt = self.X.T
        self.yt = self.y.T
        self.D_h = n_hidden
        self.f1, self.f2 = f1, f2
        self.loss = loss
        self.lr = lr
        self.n_iter = int(n_iter)
        self.seed = seed

        # initialize weights
        np.random.seed(self.seed)
        self.W1 = np.random.randn(self.D_h, self.D_X)/5
        self.c1 = np.random.randn(self.D_h, 1)/5
        self.W2 = np.random.randn(self.D_y, self.D_h)/5
        self.c2 = np.random.randn(self.D_y, 1)/5

        # initialize outputs
        self.H1 = (self.W1 @ self.Xt) + self.c1
        self.Z1 = activation_function_dict[self.f1](self.H1)
        self.H2 = (self.W2 @ self.Z1) + self.c2
        self.yhat = activation_function_dict[self.f2](self.H2)

        # fit the weights
        for iteration in range(self.n_iter):

            if self.loss == 'RSS':
                self.dL_dyhat = -(self.yt - self.yhat)
            elif self.loss == 'log':
                self.dL_dyhat = (-(self.yt/self.yhat) + (1 - self.yt)/(1 - self.yhat))
            
            if self.f2 == 'linear':
                self.dyhat_dh2 = np.ones((self.D_y, self.N))
            elif self.f2 == 'sigmoid':
                self.dyhat_dh2 = sigmoid(self.H2) * (1 - sigmoid(self.H2))
            
            self.dL_dH2 = self.dL_dyhat * self.dyhat_dh2

            self.dL_dc2 = np.sum(self.dL_dH2, 1)

            self.dL_dW2 = np.tensordot(self.dL_dH2, self.Z1, (1,1))

            self.dL_dZ1 = np.tensordot(self.W2, self.dL_dH2, (0,0))

            if self.f1 == 'ReLU':
                self.dL_dH1 = self.dL_dZ1 * np.maximum(self.H1, 0)
            elif self.f1 == 'linear':
                self.dL_dH1 = self.dL_dZ1
            
            self.dL_dc1 = np.sum(self.dL_dH1, 1)

            self.dL_dW1 = np.tensordot(self.dL_dH1, self.Xt, (1,1))

            # update the weights
            self.W1 -= self.lr * self.dL_dW1
            self.c1 -= self.lr * self.dL_dc1.reshape(-1,1)
            self.W2 -= self.lr * self.dL_dW2
            self.c2 -= self.lr * self.dL_dc2.reshape(-1,1)

            # update the outputs
            self.H1 = (self.W1 @ self.Xt) + self.c1
            self.Z1 = activation_function_dict[self.f1](self.H1)
            self.H2 = (self.W2 @ self.Z1) + self.c2
            self.yhat = activation_function_dict[self.f2](self.H2)
    
    def predict(self, X_test):
        X_testT = X_test.T
        self.h1 = (self.W1 @ X_testT) + self.c1
        self.z1 = activation_function_dict[self.f1](self.h1)
        self.h2 = (self.W2 @ self.z1) + self.c2
        self.yhat = activation_function_dict[self.f2](self.h2)
        return self.yhat