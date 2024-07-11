import torch
import torch.nn as nn

# Question 1
# Initialize the Softmax class
class Softmax(nn.Module):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x):
        exp_x = torch.exp(x)
        sum_exp_x = exp_x.sum(dim=0, keepdim=True)
        return exp_x / sum_exp_x

# Initialize the SoftmaxStable class
class SoftmaxStable(nn.Module):
    def __init__(self):
        super(SoftmaxStable, self).__init__()

    def forward(self, x):
        x_max = torch.max(x, dim=0, keepdim=True).values
        exp_x = torch.exp(x - x_max)
        sum_exp_x = exp_x.sum(dim=0, keepdim=True)
        return exp_x / sum_exp_x


data = torch.Tensor([1, 2, 3])
softmax = Softmax()
output = softmax(data)
print(output)  # tensor([0.0900, 0.2447, 0.6652])

# Question 2:
# 2a)
class Student:
    def __init__(self, name, yob, grade):
        self.name = name
        self.yob = yob
        self.grade = grade

    def describe(self):
        print(f"Student - Name: {self.name} - YoB: {self.yob} - Grade: {self.grade}")

class Teacher:
    def __init__(self, name, yob, subject):
        self.name = name
        self.yob = yob
        self.subject = subject

    def describe(self):
        print(f"Teacher - Name: {self.name} - YoB: {self.yob} - Subject: {self.subject}")

class Doctor:
    def __init__(self, name, yob, department):
        self.name = name
        self.yob = yob
        self.department = department

    def describe(self):
        print(f"Doctor - Name: {self.name} - YoB: {self.yob} - Department: {self.department}")

# 2b)
class Ward:
    def __init__(self, name):
        self.name = name
        self.people = []

    def add_people(self, person):
        self.people.append(person)

    def describe(self):
        print(f"Ward name: {self.name}")
        for person in self.people:
            person.describe()

# 2c)
def count_doctor(ward):
    count = 0
    for person in ward.people:
        if isinstance(person, Doctor):
            count += 1
    return count

Ward.count_doctor = count_doctor # Add the function to the Ward class

# 2d)
def sort_age(ward):
    ward.people.sort(key=lambda x: x.yob)
    for person in ward.people:
        person.describe()

Ward.sort_age = sort_age # Add the function to the Ward class
# 2e)
def compute_aaverage(ward):
    teachers = [person for person in ward.people if isinstance(person, Teacher)]
    total = 0
    count = 0
    for teacher in teachers:
        total += teacher.yob
        count += 1
    return total / count

Ward.compute_aaverage = compute_aaverage # Add the function to the Ward class


# Question 3:
class MyStack:
    def __init__(self, capacity):
        self.stack = []
        self.capacity = capacity
    def push(self, item):
        if len(self.stack) >= self.capacity:
            print("Stack is full")
        else:
            self.stack.append(item)

    def pop(self):
        if len(self.stack) == 0:
            print("Stack is empty")
        else:
            return self.stack.pop()

    def top(self):
        if len(self.stack) == 0:
            print("Stack is empty")
        else:
            return self.stack[-1]

    def is_empty(self):
        return len(self.stack) == 0

    def is_full(self):
        return len(self.stack) == self.capacity

stack1 = MyStack(capacity=5)

stack1.push(1)
stack1.push(2)
print(stack1.is_full())  # Output: False
print(stack1.top())  # Output: 2
print(stack1.pop())  # Output: 2
print(stack1.top())  # Output: 1
print(stack1.pop())  # Output: 1
print(stack1.is_empty())  # Output: True

# Question 4:
class MyQueue:
    def __init__(self, capacity):
        self.queue = []
        self.capacity = capacity

    def enqueue(self, item):
        if len(self.queue) >= self.capacity:
            print("Queue is full")
        else:
            self.queue.append(item)

    def dequeue(self):
        if len(self.queue) == 0:
            print("Queue is empty")
        else:
            return self.queue.pop(0)

    def front(self):
        if len(self.queue) == 0:
            print("Queue is empty")
        else:
            return self.queue[0]

    def is_empty(self):
        return len(self.queue) == 0

    def is_full(self):
        return len(self.queue) == self.capacity


queue1 = MyQueue(capacity=5)
queue1.enqueue(1)
queue1.enqueue(2)

print(queue1.is_full())  # Output: False
print(queue1.front())    # Output: 1
print(queue1.dequeue())  # Output: 1
print(queue1.front())    # Output: 2
print(queue1.dequeue())  # Output: 2
print(queue1.is_empty()) # Output: True






