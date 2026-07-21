# Nudge

An OpenCL kernel to separate any overlapping bodies.

## Usage

- During initialization, in order to ensure no bodies that are randomly placed are taking up the same space.
- When changing the radial size of multiple bodies at once, this kernel can be used to ensure their new size does not overlap.
- It _can_ be used in the stepping process -- If `EPS_DIST` or `EPS_TIME` are too large, an overlap can occur, and this kernel can be used to rectify that. Though, it does not currently play nice with multiple overlaps at once. And is far less accurate than relying on a significantly small `EPS` value.
  - This was the original use. Hence why it is written for OpenCL and not a smaller numpy vector operation. Once proper swept detection was added (see: [interaction](../interaction/README.md)), this was no longer needed.

## TODO

- Improve the support for multiple bodies overlapping at once. For instance, if there is a large cluster of bodies, they should all nudge away from the center-of-mass.
- Ensure their new nudged spot is valid (e.g. not a new overlap). This can be faked by running the kernel several times, but that's not optimal.
