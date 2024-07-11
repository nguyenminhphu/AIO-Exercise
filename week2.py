
# Question 1:
def question1(num_list,k):
    '''
    Description: This function is used for finding maximum value of a sliding window of size k in a list of numbers
    Approach: We will use two arrays left and right to store the maximum value of the left and right side of the window
    and then we will find the maximum value of the window by comparing the maximum value of left and right side of the window
    :param k:
    :param num_lst:
    :return: list of maximum values
    '''
    if not num_list or k == 0: # If num_list is empty or k is 0, then return empty list
        return [] # Return empty list

    n = len(num_list) # Get the length of num_list
    if k == 1: # If k is 1, then return the num_list
        return num_list # Return num_list

    left = [0] * n
    right = [0] * n

    left[0] = num_list[0]
    right[n - 1] = num_list[n - 1]

    for i in range(1, n): # Loop through the num_list
        left[i] = num_list[i] if i % k == 0 else max(left[i - 1], num_list[i]) # Get the maximum value of left

        j = n - 1 - i
        right[j] = num_list[j] if (j + 1) % k == 0 else max(right[j + 1], num_list[j]) # Get the maximum value of right

    result = []
    for i in range(n - k + 1):
        result.append(max(right[i], left[i + k - 1]))

    return result
# Uncomment below line to run the function
# num_list = [3, 4, 5, 1, -44 , 5 ,10, 12 ,33, 1]
# k=3
# print(question1(num_list, k))
# print(question1([3 , 4 , 5 , 1 , -44], 3))

# Question 2:
def count_chars(s):
    """
    Description: This function is used for counting the number of characters in a string
    Approach: We will use a dictionary to store the count of each character in the string
    :param s:
    :return:
    """
    character_dict = {}
    # Iterate over each character in the word
    for char in s:
        # If the character is already in the dictionary, increment its count
        if char in character_dict:
            character_dict[char] += 1
        # If the character is not in the dictionary, add it with a count of 1
        else:
            character_dict[char] = 1

    # Return the dictionary containing the character counts
    return character_dict

# Examples (uncomment to run)
# string1 = "Happiness"
# print(count_chars(string1))
#
# string2 = "smiles"
# print(count_chars(string2))
# print(count_chars("Baby"))
# Question 3:
    # Initialize an empty dictionary to hold the count of each word
def count_words(file_path):
    """
    Description: This function is used for counting the number of words in a file
    Approach: We will use a dictionary to store the count of each word in the file
    :param file_path:
    :return: counter
    """
    counter = {}

    # Open and read the file
    with open(file_path, 'r') as file:
        for line in file:
            # Split the line into words
            words = line.split()
            for word in words:
                # Convert the word to lowercase
                word = word.lower()
                # If the word is already in the dictionary, increment its count
                if word in counter:
                    counter[word] += 1
                # If the word is not in the dictionary, add it with a count of 1
                else:
                    counter[word] = 1

    # Return the dictionary containing the word counts
    return counter


# Example (uncomment to run)
# file_path = "P1_data.txt"
# result = count_words(file_path)
# print(result["man"])
# print(result["who"])

# Question 4: Levenshtein Distance
def levenshtein_distance(s, t):
    """
    Description: This function is used for calculating the Levenshtein distance between two strings
    Approach: We will use dynamic programming to calculate the Levenshtein distance between two strings
    :param s:
    :param t:
    :return: distance
    """
    m, n  = len(s), len(t)
    dp = [[0]*(n+1) for _ in range(m+1)]

    # initialised first row and column of the dp table
    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    # fill the dp table
    for i in range(1,m+1):
        for j in range(1,n+1):
            if s[i-1] == t[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j]+1, # deletion
                               dp[i][j-1]+1, # insertion
                               dp[i-1][j-1]+1) # substitution

    return dp[m][n]

# Example (uncomment to run)
# s = "kitten"
# t = "sitting"
# print(levenshtein_distance(s,t))
# print(levenshtein_distance("hi","hello"))
# print(levenshtein_distance("hola","hello"))

# Multiple choice test questions:
#5
def check_the_number(N):
    list_of_numbers = []
    results = ""

    for i in range(1, 5):
        list_of_numbers.append(i)  # Append i to the list_of_numbers

    if N in list_of_numbers:
        results = "True"
    else:
        results = "False"

    return results

# Examples (uncomment to run)
# N = 7
# assert check_the_number(N) == "False"
#
# N = 2
# results = check_the_number(N)
# print(results)  # Output: "True"


#6
def my_function(data, max_val, min_val):
    result = []
    for i in data:
        if i < min_val:
            result.append(min_val)
        elif i > max_val:
            result.append(max_val)
        else:
            result.append(i)
    return result

# Examples (uncomment to run)
# my_list = [5, 2, 5, 0, 1]
# max_val = 1
# min_val = 0
# assert my_function(max_val=max_val, min_val=min_val, data=my_list) == [1, 1, 1, 0, 1]
#
# my_list = [10, 2, 5, 0, 1]
# max_val = 2
# min_val = 1
# print(my_function(max_val=max_val, min_val=min_val, data=my_list))  # Output: [2, 2, 2, 1, 1]

#7
def my_function7(x, y):
    x.extend(y)
    return x

# Examples (uncomment to run)
# list_num1 = ['a', 2, 5]
# list_num2 = [1, 1]
# list_num3 = [0, 0]
#
# assert my_function7(list_num1, my_function7(list_num2, list_num3)) == ['a', 2, 5, 1, 1, 0, 0]
#
# list_num1 = [1, 2]
# list_num2 = [3, 4]
# list_num3 = [0, 0]
#
# print(my_function7(list_num1, my_function7(list_num2, list_num3)))  # Output: [1, 2, 3, 4, 0, 0]

#8
# print(min([1,2,3,-1]))

#9
# print(max([1,9,9,0]))

#10
def my_function(integers, number=1):
    return any(i == number for i in integers)

# Example (uncomment to run)
# my_list = [1, 3, 9, 4]
# assert My_function(my_list, -1) == False
#
# my_list = [1, 2, 3, 4]
# print(My_function(my_list, 2))  # Output: True

def my_function11(list_num):
    var = 0
    for i in list_num:
        var += i
    return var/len(list_num)

# Example (uncomment to run)
# assert my_function11([4,6,8]) == 6
# print(my_function11([0,1,2]))

#12
def my_function12(data):
    return [x for x in data if x%3 == 0]

# Example (uncomment to run)
# print(my_function12([3, 9,4,5]))
# print(my_function12([1,2,3,5,6]))

#12A
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# Example (uncomment to run)
# print(factorial(4))  # Output: 24

#14
def read_backward(x):
    return x[::-1]

# Example (uncomment to run)
# print(read_backward("I can do it"))
# print(read_backward("apricot"))

#15
def function_helper(x):
    return 'T' if x > 0 else 'N'
def my_function15(data):
    return [function_helper(x) for x in data]

# Example (uncomment to run)
# print(my_function15([10,0,-10,-1]))
# print(my_function15([2,3,5,-1]))

#16
def function_helper(x, data):
    for i in data:
        if x == i:
            return 0
    return 1

def my_function16(data):
    res = []
    for i in data:
        if function_helper(i, res):
            res.append(i)
    return res

# Example (uncomment to run)
# lst = [10, 10, 9, 7, 7]
# assert my_function16(lst) == [10, 9, 7]
#
# lst = [9, 9, 8, 1, 1]
# print(my_function16(lst))  # Output: [9, 8, 1]




