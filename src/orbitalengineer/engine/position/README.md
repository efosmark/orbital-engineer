# Position

A small OpenCL kernel that takes the velocity and dt step value and updates the position of each body.

Since this operation is O(N), it could be handled host-side. But having a distinct kernel allows it to avoid a copy operation.
