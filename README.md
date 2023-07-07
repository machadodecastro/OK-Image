# ![logo](https://user-images.githubusercontent.com/5161201/194437918-22cab0f2-5a03-4650-90c4-69209e43220e.png)
<h1>OK Image</h1>

A free image preprocessing tool to apply Data Augmentation techniques in Computer Vision

<h2>Setup</h2>
<ul>
  <li>1 - Clone project: git clone https://github.com/machadodecastro/OK-Image/</li>
  <li>2 - cd OK-Image</li>
  <li>3 - Create virtual environment: virtualenv myenv</li>
  <li>4 - myenv\scripts\activate</li>
  <li>5 - pip install -r requirements.txt</li>
  <li>6 - python oki.py</li>
</ul>

<hr/>
Why many folders? Because I projected initially to save each transformed images into a separated folder. You can modify that and put into a single folder, editing the oki.py file.

<h2>Included Preprocessing OpenCV Operations</h2>
Below are listed the preprocessing operations included until the moment:

1. Grayscale
2. Normalization
3. Cropping
4. Rotation (45, 90, 180 and 270 degrees)
5. Resize
6. Enhance (Contrast + Brightness)
7. HSV Color
8. Difference of Gaussian
9. Negative Image
10. Contrast Stretching
11. Blurring (Gaussian Blur, Median Blur, Blur, Box Filter)
12. Gamma Correction
13. Drawing (Image, Text, Circle, Rectangle)
14. Shifting
15. Flipping
16. Luminosity
17. Mask (Circle, Rectangle)
18. Splitting RGB
19. Merging RGB
20. Edge Cascade
21. Dilation
22. Erosion
23. Contours
24. Concatenation (Vertical, Horizontal, Vertical and Horizontal)
25. Blending
26. Zooming
27. Bitwise (AND, OR, NOT, XOR)
28. Getting data (Pixels, Shape, Data Type)
29. Histogram (Grayscale, Color, Equalization)
30. Number of Image Contours
