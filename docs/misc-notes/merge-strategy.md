# Merging Bodies

```python
bodies_to_bunch = dict()
bunches = set()

for i in range(shm.N):
    if i not in bodies_to_bunch:
        bunch = set([i])
    
    overlapping = np.argwhere(shm.interaction[i,] == OVERLAPPING).flatten()
    if overlapping.size:
        while overlapping.size:
            j = overlapping.pop(0)
```

1. Create a set of all bodies, `b`
2. Iterate over the items one at a time (pop)
    - Create a set of all bodies where `i < j`, named `other`
    - Create a "bunch" for the current item
        - Look for interactions where overlapping
            - Add `j` to `bunch`
            - Remove `j` from `bodies`
