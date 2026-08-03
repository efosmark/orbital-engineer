# Collision Grouping

## Finding Colliding Bodies

### Per Lane

**num_contacts_by_lane** _(N * num\_groups)_

| ID  | lane-0 | lane-1 | lane-2 | lane-ng |
| --- | ------ | ------ | ------ | ------- |
| 0   | 1      | 1      | 2      | ...     |
| 1   | 3      | 0      | 1      | ...     |
| 2   | 0      | 0      | 1      | ...     |
| N   | ...    | ...    | ...    | ...     |

**contacts_by_lane** _(N * N)_

- _Assume group_size = 3_
- Empty cells are set to NaN

| ID  |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |  N  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  0  |  3  |     |     |  4  |     |     |  7  |  9  |     |     |
|  1  |  4  |  5  |  6  |     |     |     |  7  |     |     |     |
|  2  |     |     |     |     |     |     |  8  |     |     |     |
|  N  |     |     |     |     |     |     |     |     |     |     |

### Reduced

**num_contacts** _(N)_

|  0  |  1  |  2  |  N  |
| --- | --- | --- | --- |
|  4  |  4  |  1  |     |

**contacts** _(N * N, sparse)_

- Empty cells are set to NaN

| ID  |  0  |  1  |  2  |  3  |  N  |
| --- | --- | --- | --- | --- | --- |
|  0  |  3  |  4  |  7  |  9  |     |
|  1  |  4  |  5  |  6  |  7  |     |
|  2  |  8  |     |     |     |     |
|  N  |     |     |     |     |     |
