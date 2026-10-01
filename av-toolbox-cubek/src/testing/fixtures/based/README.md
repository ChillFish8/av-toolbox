# Fixtures

`fence_16` is a 96 by 64 crop of the luma of Big Buck Bunny, from `bbb_sunflower_1080p_30fps_normal.mp4`,
frame 13500 as ffms2 counts frames, with its top left sample at column 1600, row 900. It shows the sharp
diagonal edge of a fence post over blurred stalks.

The 8-bit luma is widened to 10 bits as `round(y * 1023 / 255)`, then to 16 bits as
`round(y * 65535 / 1023)`.

`lines_8` and `tiny_8` are crops of a synthetic plane of aliased lines.

Big Buck Bunny, (c) copyright 2008, Blender Foundation, www.bigbuckbunny.org, licensed under CC BY 3.0.
