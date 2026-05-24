
# Pipeline

```plain

[step]

    -> [kick]
        -> [velocity ()]

    --> [drift]
        --> [position (status, velocity)]
            --> [edge_distance (radius, position)]

    --> [kick2]
        -> [velocity]
            -> [update-relative-velocity-along-normal]
                -> [compute-collision-groups]
                    -> [merge]
                -> [bounce]

    --> [interaction]
        --> [compute-time-of-impact]
            --> [minimum-node-dt]
                --> [global-minimum-dt]
            --> [minimum-group-dt]
    

```
