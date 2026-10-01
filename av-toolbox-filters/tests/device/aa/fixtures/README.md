# Fixtures

`based_fence.luma` is a 192 by 128 crop of the luma of Big Buck Bunny, from
`bbb_sunflower_1080p_30fps_normal.mp4`, frame 13500 as ffms2 counts frames, with its top left sample at
column 1520, row 736. It shows the long edge of a sharpened fence post and grass stalks over a blurred
background.

The 8-bit luma is widened to 10 bits as `round(y * 1023 / 255)` and stored as two little-endian bytes per
sample.

Big Buck Bunny, (c) copyright 2008, Blender Foundation, www.bigbuckbunny.org, licensed under CC BY 3.0.
