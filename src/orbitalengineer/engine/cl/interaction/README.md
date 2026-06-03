# Interaction Prediction

## The Problem

Computing the next dt is performed by getting the minimum time until the next interaction across the whole node set. Because of this, even if multiple nodes have unbounded interaciton times, they are clamped to the smallest.

 1. Two nodes approach
 2. dt between them becomes small (dt_min)
 3. the min dt across all nodes is computed and used in the next step
 4. all nodes are incremented +dt_min, even if they are unbounded
    - requiring recomputation for N^2
    - rather than step forward +dt_base for the unbounded nodes and +dt_min for the ones that care

## Proposed Solution

- Nodes approach one-another
- Eventually, their TOI is below dt_base
- Nodes are grouped together on which nodes will interact with each-other within the remaining dt
- The smallest dt (dt_min) for each group is determiend
- The initial dt for each group member is set to the group's dt_min

## Interaction Times

| IDX   |  00  |  01  |  02  |  03  |  04  |  05  |  06  |  07  |  08  |  09  |
|-------|------|------|------|------|------|------|------|------|------|------|
| TOI   | 0.01 | 0.13 | 0.44 | 0.21 | 0.74 | 0.44 | 0.01 | 0.14 | 0.13 | 0.13 |
| GROUP |  00  |  01  |  02  |  03  |  04  |  02  |  00  |  01  |  01  |  01  |

### DISTINCT GROUPS

GROUPS: 00, 01, 02, 03, 04

### GROUP MEMBERS

|        |  [00]  |  [01]  |  [02]  |  [03]  |
|--------|--------|--------|--------|--------|
|  [00]  |   00   |   06   |        |        |
|  [01]  |   01   |   07   |   08   |   09   |
|  [02]  |   02   |   05   |        |        |
|  [03]  |   03   |        |        |        |
|  [04]  |   06   |        |        |        |
|  [05]  |        |        |        |        |

## GROUP SIZES

|  [00]  |  [01]  |  [02]  |  [03]  |  [04]  |
|--------|--------|--------|--------|--------|
|    2   |    4   |    2   |    1   |    1   |
