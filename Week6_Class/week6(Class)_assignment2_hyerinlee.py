import numpy as np

#시그모이드 함수
def sigmoid(x):
        return 1 / (1+np.exp(-x))

#편미분 함수
def numerical_derivative(f, x):
    delta_x = 1e-4 # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index        
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x) # f(x+delta_x)

        x[idx] = tmp_val - delta_x 
        fx2 = f(x) # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val 
        it.iternext()   

    return grad

#========================================================================================#

class LogisticRegression_cls:
 
    
    def __init__(self, x_data, y_data, learning_rate=0.01):
        self._X_train = x_data
        self._y_train = y_data
        self._learning_rate = learning_rate
        self._W = np.random.rand(1,1)
        self._b = np.random.rand(1)

    #손실 값 계산 함수
    def error_val(self):
        delta = 1e-7
        
        z = np.dot(self._X_train, self._W) + self._b
        y = sigmoid(z)
        
        return -np.sum(self._y_train*np.log(y + delta) + (1 - self._y_train)*np.log((1-y)+delta))
    
             
    #예측 함수
    def predict(self, X):
        result=[]
        for x in X:
            z=np.dot(x, self._W) + self._b
            y=sigmoid(z)

            if y > 0.5:
                result.append(1)
            else:
                result.append(0)

        return result
    
    #학습 함수
    def train(self): 

        f = lambda x : self.error_val()
        print("Initial error value = ", self.error_val(), "Initial W = ", self._W, "\n", ", b = ", self._b )
        
        for step in  range(10001):  

            self._W -= self._learning_rate * numerical_derivative(f, self._W)

            self._b -= self._learning_rate * numerical_derivative(f, self._b)
            
            if (step % 400 == 0):
                print("step = ", step, "error value = ", self.error_val(), "W = ", self._W, ", b = ",self._b )