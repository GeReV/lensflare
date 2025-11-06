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
