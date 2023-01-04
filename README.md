# Image Localization

Experimenting Template Matching using Genetic Algorithm.
To run, specify the image filename in either `one_scale.py` or `two_scale.py` and then run either of the file
It will take quiet a long time to run since the algorithm efficiency is exponential with image dimension (usually 2 to 3 minutes)

# User Manual

1. Clone this project:
```
git clone https://github.com/tringuyenminh2801/Image-Localization.git
```
2. Choose either [one_scale.py](/one_scale.py) or [two_scale.py](/two_scale.py) to run. One scale means that the algorithm will find the coordinates of the upper left point of the bounding box and the scale ratio of the bounding box compare to the original image. Two scale means the algorithm will find the coordinates of the upper left and the lower right point of the bounding box.

3. An image of the original image will first appear, press `q` to proceed to the template image.\\
Press `q` again, then the algorithm will run (the one scale will be much faster, it takes about 3 to 5 minutes). Then the Cosine Similarity value chart will appear to help you visualize the progress of finding the optimal image. \\
Press `q` again to see the original image with the bounding box.

# Final Result
## Input image and template
Image captured by short focal len
![alt](/assets/original.png)
\
Image captured by long focal len
![alt](/assets/template.png)


## One scale
![alt](/assets/one_result.png)
\
Cosine Similarity value
![alt](/assets/one_sim.png)

## Two scale
Result using [two_scale.py](/two_scale.py)
![alt](/assets/two_result.png)
\
Cosine Similarity value
![alt](/assets/two_sim.png)