This is a work in progress.

This repo is a challenge to try and implement a paper with as little outside help as possible.

I decided to implement the lens flare effect from [Physically-Based Real-Time
Lens Flare Rendering](https://resources.mpi-inf.mpg.de/lensflareRendering/).

## Development notes

![](screenshots/lenses.png)

This is what I should have done initially, but didn't.

I relied on the [description of realistic cameras from the Physically Based Rendering book (3rd edition)](https://www.pbr-book.org/3ed-2018/Camera_Models/Realistic_Cameras)
to get a model of the lens system. 
Should have made sure the derived data can work with the unmodified tracing algorithm and made a debug diagram as above.

Instead, I started off by translating the code from the supplementary material of the paper and threw everything at it,
trying to see what will stick.

Continued fiddling is starting to show some results that seem promising, but still incorrect, as can be seen below.

The part of the code that tries to limit the grid of traced rays to the dimensions that are certain to reach the sensor
needs some improvement. It currently does not take into consideration the possible angles of the rays. I believe this is
what causes the visual edges in the ghosts.

![](screenshots/wip01.png)

Adding a simple hot-reloading mechanism for the shaders was a good idea, even if it was a bit of a headache and took a
couple of hours. It sped up my iteration times quite a bit.

My next steps are to improve the code to shrink the grids as mentioned above, and adding some UI elements that will allow
me to play around with the `d1` parameter responsible for the anti-reflection coating of the lenses, to get some color 
into this.