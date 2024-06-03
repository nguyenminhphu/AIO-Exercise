import math
import random
def exercise1(tp,fp,fn):
    if type(tp) != int: # check if tp is an integer
        print("tp must be an int")
        return
    elif type(fp) != int: # check if fp is an integer
        print("fp must be an int")
        return
    elif type(fn) != int: # check if fn is an integer
        print("fn must be an int")
        return
    elif tp <= 0 or fp <= 0 or fn <= 0: # check if tp and fp are greater than 0
        print("tp and fp and fn must be greater than 0")
        return
    precision = tp/(tp+fp) # calculate precision
    recall = tp/(tp+fn) # calculate recall
    f1 = 2*(precision*recall)/(precision+recall) # calculate f1 score
    print("Precision is: ",precision)
    print("Recall is: ",recall)
    print("f1-score is: ",f1)
    return
# Example to verify the function: (uncomment the lines to run the function)
# exercise1(2,3,4)
# exercise1('a', 3 , 4)
# exercise1(2, 'a', 4)
# exercise1(2, 3, 'a')
# exercise1(2, 3, 0)
# exercise1(2.1, 3, 0)


# Exercise 2:
def sigmoid_function(x):
    return 1/(1+math.exp(-x))

def ReLU_function(x):
    return x*(x>0)

def ElU_function(a,x):
    return x*(x>0) + a*(math.exp(x)-1)*(x<=0)

def excerise2(type_function, x):
    if is_number(x) == False:
        print("x must be a number")
        return
    if type_function == 'sigmoid':
        return sigmoid_function(x)
    elif type_function == 'ReLU':
        return ReLU_function(x)
    elif type_function == 'ElU':
        return ElU_function(1,x)
    else:
        print(f"{type_function} is not supported")
        return

def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


# Exercise 3:
def compute_loss(loss_name, y_hat, y): # compute loss for each iteration
    if loss_name == 'MAE':
        return abs(y_hat - y)
    elif loss_name == 'MSE':
        return (y_hat - y)**2
    elif loss_name == 'RMSE':
        return math.sqrt((y_hat-y)**2)

def compute_final_loss(loss_name, y_hat, y): # compute final loss
    n = len(y_hat)
    total_loss = 0
    if loss_name == 'MAE':
        total_loss = sum([abs(y_hat[i] - y[i]) for i in range(n)]) / n
    elif loss_name == 'MSE':
        total_loss = sum([(y_hat[i] - y[i])**2 for i in range(n)]) / n
    elif loss_name == 'RMSE':
        total_loss = math.sqrt(sum([(y_hat[i] - y[i])**2 for i in range(n)]) / n)
    return total_loss
def exercise3():
    n = input("Input number of samples ( integer number ) which are generated: ") # get number of samples
    if not n.isnumeric(): # check if n is a number
        print("n must be an integer")
        return
    n = int(n)
    loss_name = input("Input loss name: ") # get loss name
    loss_lst = ['MSE', 'MAE', 'RMSE'] # Define list of lost names
    if loss_name not in loss_lst: # check if loss name is in the list
        print(f"{loss_name} is not supported")
        return

    y = [random.uniform(0,10) for _ in range(n)] # list of y values
    y_hat = [random.uniform(0,10) for _ in range(n)] # list of y_hat values
    for i in range(n):
        print(f"loss name: {loss_name}, sample: {i}, pred: {y_hat[i]}, target: {y[i]}, loss:"
            f" {compute_loss(loss_name, y_hat[i], y[i])}")

    final_loss = compute_final_loss(loss_name, y_hat, y)
    print(f"final {loss_name}: {final_loss}")

# Example to verify the function:
# exercise3() # uncomment this line to run the function
# Exercise 4:
def approx_sin(x, n):
    print(sum([((-1)**i)*(x**(2*i+1))/math.factorial(2*i+1) for i in range(n)]))
def approx_cos(x, n):
    print(sum([((-1)**i)*(x**(2*i))/math.factorial(2*i) for i in range(n)]))

def approx_sinh(x, n):
    print(sum([(x**(2*i+1))/math.factorial(2*i+1) for i in range(n)]))
def approx_cosh(x, n):
    print(sum([(x**(2*i))/math.factorial(2*i) for i in range(n)]))

# Example to verify the function: (uncomment the lines to run the functions)
# approx_sin(3.14, 10)
# approx_cos(3.14, 10)
# approx_sinh(3.14, 10)
# approx_cosh(3.14, 10)

# Exercise 5: (Mean Difference of nth Root Error)
def md_nre_single_sample(y,y_hat,n,p):
    print(y**(1/n)-y_hat**(1/n))
    return
# Example to verify the function: (uncomment the lines to run the function)
# md_nre_single_sample(100,99.5,2,1)
# md_nre_single_sample(50,49.5,2,1)
# md_nre_single_sample(20,19.5,2,1)
# md_nre_single_sample(0.6,0.1,2,1)









