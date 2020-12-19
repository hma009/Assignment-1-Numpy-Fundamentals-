#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[2]:


import numpy as np


# 2. Create a null vector of size 10 

# In[6]:


v=np.zeros(10)
v


# 3. Create a vector with values ranging from 10 to 49

# In[3]:


v1=np.arange(10,49)
v1


# 4. Find the shape of previous array in question 3

# In[5]:


s= np.shape(v1)
s


# 5. Print the type of the previous array in question 3

# In[8]:


print(type(v1))


# 6. Print the numpy version and the configuration
# 

# In[13]:


version=print(np.__version__)
confg= print(np.show_config())


# 7. Print the dimension of the array in question 3
# 

# In[15]:


dimension = print( np.ndim(v1))


# 8. Create a boolean array with all the True values

# In[17]:


a1=print(np.ones(5, dtype=bool))


# 9. Create a two dimensional array
# 
# 
# 

# In[29]:


a2 = np.ndarray(shape=(4,2), dtype= int )
a21 = np.arange(20).reshape(4,5)
print(a2)
print(a21)
print(np.shape(a2))


# 10. Create a three dimensional array
# 
# 

# In[34]:


a2 = np.ndarray(shape=(4,2,2), dtype= int )
a21 = np.arange(20).reshape(2,5,2)
print('Array a2')
print(a2)
print('Array a21')
print(a21)
print('shape of a2')
print(np.shape(a2))
print('shape of a21')
print(np.shape(a21))


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[35]:


vv= np.arange(6)
print(vv)
rv= np.flipud(vv)
print(rv)


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[38]:


n=np.zeros(10)
n[4]=1
n


# 13. Create a 3x3 identity matrix

# In[39]:


i =np.identity(3)
print(i)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[44]:


arr = np.array([1, 2, 3, 4, 5])
print(arr.dtype)
narr=arr.astype(float)
print(narr)


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[61]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  


arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])

np.multiply(arr1,arr2)


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[63]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],

            [7., 2., 12.]])
arr = arr1==arr2
arr


# 17. Extract all odd numbers from arr with values(0-9)

# In[65]:


ar= np.arange(10)
on=ar[ar%2==1]
on


# 18. Replace all odd numbers to -1 from previous array

# In[71]:


ar= np.arange(10)
ar[ar%2==1]=-1
ar


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[78]:


arr = np.arange(10)
print(arr)
arr[5:9]=12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[84]:


a=np.ones((7,7))
print(a)
a[1:6,1:6]=0
print(a)


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[87]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
print(arr2d)
arr2d[1,1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[89]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[:]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[98]:


x=np.arange(10).reshape(5,2)
print(x)
slice=x[:1]
slice


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[99]:


x=np.arange(10).reshape(5,2)
print(x)
slice=x[1:2]
slice


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[104]:


x=np.arange(10).reshape(2,5)
print(x)
slice=x[:,2:3]
slice


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[113]:


r=np.random.random((10,10))
print(r)
minimum=r.min()
print(minimum)
maximum=r.max()
print(maximum)


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[118]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
common=np.intersect1d(a,b)
common


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[129]:


a = np.array([1,2,3,2,3,4,3,4,5,6])
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(np.intersect1d(a,b))


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[133]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
data1= data[:]!=names[2]
data1


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[ ]:





# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[135]:


ax=np.arange(1,16).reshape(5,3)
ax


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[140]:


ax1=np.arange(1,17).reshape(2,2,4)
ax1


# 33. Swap axes of the array you created in Question 32

# In[143]:


ax1=np.arange(1,17).reshape(2,2,4)
np.swapaxes(ax1,0,1)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[159]:


a11=np.ndarray(10)
print(a11)
a12=np.sqrt(a11)
print (a12)

a12[a12<0.5]==0
a12


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[ ]:





# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[ ]:





# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[171]:


a = np.array([1,2,3,4,5])
b = np.array([5,6,7,8,9])
np.setdiff1d(a,b)


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[169]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
newColumn = numpy.array([[10,10,10]])


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[167]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[166]:


m=np.random.random((20))
print(m)

np.cumsum(m)

