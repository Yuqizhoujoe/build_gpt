# %%
''' Zip & Unzip examples'''

list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']
list3 = [True, False, True]

# Zip the lists
zipped = zip(list1, list2, list3)

print(list(zipped))

# Unzip the lists
# numbers, letters, booleans = zip(*zipped)

# print(numbers)   # Output: (1, 2, 3)
# print(letters)   # Output: ('a', 'b', 'c')
# print(booleans)  # Output: (True, False, True)

csv_data = [
      [1, 'Alice', 85],
      [2, 'Bob', 90],
      [3, 'Charlie', 78]
]

ids, names, scores = zip(*csv_data)

print(ids)    # Output: (1, 2, 3)
print(names)  # Output: ('Alice', 'Bob', 'Charlie')
print(scores) # Output: (85, 90, 78)

# %%
a = [1,2,3]
b = [4]
print(a + b)
# %%
words = ["hello", "world"]
for ch1, ch2 in zip(words[0], words[1]):
      print(ch1, ch2)
# %%
word = ".hello."
for ch1, ch2 in zip(word, word[1:]):
      print(ch1, ch2)
# %%
