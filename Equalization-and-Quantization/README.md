# Equalization and Quantization
In this project I implemented histogram equalization and quantization of images.

## Table of contents
* [Histogram Equalization](#histogramequalization)
* [Quantization](#quantization)

## Histogram Equalization
The goal is to improve the image contrast and to make use of all gray levels.
The algorithm for the histogram equalization is as shown below:
1. Compute the image histogram.
2. Compute the cumulative histogram.
3. Normalize the cumulative histogram.
4. Multiply the normalized histogram by the maximal gray level value (Z − 1).
5. Verify that the minimal value is 0 and the the maximal is Z−1, otherwise, stretch the result linearly
in the range [0, Z − 1].
6. Round the values to get integers.
7. Map the intensity values of the image using the result of step 6.

For example:

Original image:

![Jerusalem](results/jerusalem_gray.png)

Original histogram:

![Jerusalem Histogram](results/jerusalem_hist.png)

Image after histogram equalization:

![Jerusalem](results/jerusalem_eq.png)

Histogram after equalization:

![Jerusalem](results/jerusalem_hist_eq.png)



## Quantization
The goal is to compress a range of values to a single quantum value.
The algorithm for the quantization is as shown below:

Let h be the histogram, let [Zi−1, Zi] be the i-th segment where Z0 = 0, Zk = 255.
Then, we would like to minimize the error over: <img src="https://render.githubusercontent.com/render/math?math=\sum_{i=0}^{k-1}\left(\sum_{g=\left\lfloor z_{i}\right\rfloor +1}^{\left\lfloor z_{i+1}\right\rfloor }(q_{i}-g)^{2}\cdot h(g)\right)">

The solution to the minimization problem is:
<img src="https://render.githubusercontent.com/render/math?math=q_{i}=\frac{\sum_{g=\left\lfloor z_{i}\right\rfloor +1}^{\left\lfloor z_{i+1}\right\rfloor }g\cdot h(g)}{\sum_{g=\left\lfloor z_{i}\right\rfloor +1}^{\left\lfloor z_{i+1}\right\rfloor }h(g)}">

where:
<img src="https://render.githubusercontent.com/render/math?math=z_{i}=\frac{q_{i-1}+q_{i}}{2}">

For example:

Original image:

![Monkey](monkey.jpg)

Image after quantization to 4 quant values:


![Monkey Quantization](results/monkey_quant.png)
