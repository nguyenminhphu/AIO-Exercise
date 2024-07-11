import numpy as np
import matplotlib.image as mpimg
import pandas as pd
# Question 1:
arr1 = np.arange(0, 10, 1)
print(arr1)


# Question 2:
# arr2 = np.ones((3, 3), dtype=bool)
# arr2 = np.ones((3, 3)) > 0
arr2 = np.full((3, 3), fill_value=True, dtype=bool)
print(arr2)

# QUestion 3:
arr3 = np.arange(0, 10)
print(arr3[arr3 % 2 == 1])

# Question 4:
arr4 = np.arange(0, 10)
arr4[arr4 % 2 == 1] = -1
print(arr4)

# Question 5:
arr5 = np.arange(10)
arr5 = arr5.reshape(2, -1)
print(arr5)

# Question 6:
arr61 = np.arange(10).reshape(2, -1)
arr62 = np.repeat(1, 10).reshape(2, -1)
arr6 = np.concatenate([arr61, arr62], axis=0)
print(arr6)

# Question 7:
arr71 = np.arange(10).reshape(2, -1)
arr72 = np.repeat(1, 10).reshape(2, -1)
arr7 = np.concatenate([arr71, arr72], axis=1)
print(arr7)

# Question 8:
arr8 = np.array([1, 2, 3])
print(np.repeat(arr8, 3))
print(np.tile(arr8, 3))

# Question 9:
arr9 = np.array([2,6,1,9,10,3,27])
index = np.where((arr9 >= 5) & (arr9 <= 10))
print("result: ", arr9[index])

# Question 10:
def maxx(x, y):
    if x >= y:
        return x
    else:
        return y

a10 = np.array([5, 7, 9, 8, 6, 4, 5])
b10 = np.array([6, 3, 4, 8, 9, 7, 1])

pair_max = np.vectorize(maxx, otypes=[float])
print(pair_max(a10, b10))

# Question 11:
a11 = np.array([5, 7, 9, 8, 6, 4, 5])
b11 = np.array([6, 3, 4, 8, 9, 7, 1])
print("Result: ", np.where(a11 < b11, b11, a11))

# Question 12:
img = mpimg.imread('dog.jpeg')
# Convert the image to grayscale using the Lightness method
gray_img_01 = np.mean([np.max(img, axis=2), np.min(img, axis=2)], axis=0)

# Print the value at (0, 0)
print(gray_img_01[0, 0])

# Question 13:
gray_img_02 = np.mean(img, axis=2)

# Print the value at (0, 0)
print(gray_img_02[0, 0])

# Question 14:
# Convert the image to grayscale using the Luminosity method
gray_img_03 = 0.21 * img[:, :, 0] + 0.72 * img[:, :, 1] + 0.07 * img[:, :, 2]

# Print the value at (0, 0)
print(gray_img_03[0, 0])

# Question 15:
# Load the dataset
df = pd.read_csv('advertising.csv')

# Convert the dataframe to a numpy array
data = df.to_numpy()

# Find the maximum value and its index in the "Sales" column
sales_column = data[:, -1]  # Assuming "Sales" is the last column
max_value = np.max(sales_column)
max_index = np.argmax(sales_column)

print("Max:", max_value)
print("Index:", max_index)

# Question 16:
tv_mean = df['TV'].mean()

print("Mean of TV column:", tv_mean)

# Question 17:
count_sales_20_or_more = df[df['Sales'] >= 20].shape[0]

print("Number of records with Sales >= 20:", count_sales_20_or_more)

# Question 18:
filtered_df = df[df['Sales'] >= 15]

# Calculate the mean of the "Radio" column for the filtered dataset
radio_mean_filtered = filtered_df['Radio'].mean()

print("Mean of Radio column for Sales >= 15:", radio_mean_filtered)

# Question 19:
# Calculate the mean of the "Newspaper" column
newspaper_mean = df['Newspaper'].mean()

# Filter the dataset for "Newspaper" > average "Newspaper"
filtered_df = df[df['Newspaper'] > newspaper_mean]

# Calculate the sum of the "Sales" column for the filtered dataset
sales_sum_filtered = filtered_df['Sales'].sum()

print("Sum of Sales for Newspaper > average:", sales_sum_filtered)

# Question 20:
# Calculate the average value of the "Sales" column
sales_mean = df['Sales'].mean()

# Create the 'scores' array
conditions = [
    df['Sales'] > sales_mean,
    df['Sales'] < sales_mean,
    df['Sales'] == sales_mean
]
choices = ['Good', 'Bad', 'Average']
scores = np.select(conditions, choices, default='Unknown')

# Print the result for scores[7:10]
print("Scores[7:10]:", scores[7:10])

# Question 21:
closest_to_mean = df['Sales'].iloc[(df['Sales'] - sales_mean).abs().argsort()[:1]].values[0]

# Create the 'scores' array
conditions = [
    df['Sales'] > closest_to_mean,
    df['Sales'] < closest_to_mean,
    df['Sales'] == closest_to_mean
]
choices = ['Good', 'Bad', 'Average']
scores = np.select(conditions, choices, default='Unknown')

# Print the result for scores[7:10]
print("Scores[7:10]:", scores[7:10])




#end



