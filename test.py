merged = {
    (101, 100): 1,
    (101, 101): 2,
    (101, 102): 3
}

# merged_list = [(count, key) for key, count in merged]
# merged_list.sort(key=lambda x: x[0], reverse=True)

merged_list = sorted([(count, key) for key, count in merged], reverse=True)
print(merged_list)