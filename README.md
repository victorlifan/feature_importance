# Project Title
Feature Importance and Feature Selection with Statistical Methods

## by Fan Li

## Table of Contents
1. [Description](#description)
2. [Workflow](#Workflow)
	1. [Kmeans](#km)
	2. [Kmeans ++](#km+)
	3. [Applications](#app)
	4. [Advanced topic: RF + Kmeans](#rf+km)
	5. [Limitations](#mf)
3. [Dataset](#Dataset)
4. [About](#About)
5. [References](#ref)

<a name="description"></a>
## Description
For this project, I implemented the Kmeans as well as the kmeans++ algorithm from scratch. I used five data sets to showcase some applications and results of those algorithms. Further, after uncovering the drawbacks of Kmeans, I implemented a ‘Spectral clustering’ using the random forest (RF) technique paired with kmeans++ to overcome the ‘discontinuity of clusters’ issue. Lastly, I addressed some limitations and possible improvements for future research reference.


<a name="Workflow"></a>
## Workflow:
<a name="km"></a>
##### 1. Kmeans
> Procedure:

1. Initialize centroids꞉
Randomly initialize k number of data points from the original X data. The number of k depends on how many clusters we want to end up with.

2. Compute distance꞉
Here I used Euclidean distance to measure the distance from each of the remaining data points to each of the centroids we initialized in step 1, assigning each of the remaining data points to the ‘closest ’ centroids.

3. Update centroids꞉
Within each cluster, compute the average distance of all the data points to that centroid FEATURE WISE. This average distance will be the new centroids’ ‘coordinate’ in that cluster. Intuitively speaking, this means we are correcting the centroids to be the ‘center’ of that cluster. This means our final centroids will most likely not be members of the dataset. The reason we picked data points from the dataset as initial centroids is simply to assign a starting point.

4. Reassign data point
Finally, compute distance, reassign data points according to the new centroids we updated in step 3, update centroids. Iterate the above process until the centroids’ ‘coordinates’ don’t change any more.

ISSUES:
> Each rounds’ initial centroids are different due to the mechanism of random initialization. This goes without saying, but the algorithm produces slightly different final centroids, labels, and clusters. The good news is all the final norms are 0, which means those clusters make perfect sense in each of their own round’s ‘world’.


<a name="km+"></a>
##### 2. Kmeans ++

> Procedure:

1. Initialize centroids꞉
Randomly initialize k number of data points from the original X data. The number of k depends on how many clusters we want to end up with.
2. Compute distance꞉
Here I used Euclidean distance to measure the distance from each of the remaining data points to each of the centroids we initialized in step 1, assigning each of the remaining data points to the ‘closest ’ centroids.
3. Update centroids꞉
Within each cluster, compute the average distance of all the data points to that centroid FEATURE WISE. This average distance will be the new centroids’ ‘coordinate’ in that cluster. Intuitively speaking, this means we are correcting the centroids to be the ‘center’ of that cluster. This means our final centroids will most likely not be members of the dataset. The reason we picked data points from the dataset as initial centroids is simply to assign a starting point.

4. Reassign data point
Finally, compute distance, reassign data points according to the new centroids we updated in step 3, update centroids. Iterate the above process until the centroids’ ‘coordinates’ don’t change any more.

<a name="app"></a>
##### 3. Applications
*  Synthetic data set

![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/1.png)

* Multi-dimension data (Circle data 500*2)

![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/multi1.png)
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/multi2.png)

> As you can see, Kmeans performs poorly on disjointed and nested structures. To rescue, I will introduce spectral clustering by using RF and Kmeans together in the Advanced topic section.

* Breast cancer

    * Without scaling X꞉

![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/bc1.png)

    * With scaled X
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/bc2.png)

* Image compression
    * Grayscale
        * Original:
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/north-africa-1940s-grey.png)
        * Kmeans++ copresion:
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/gray_km.png)

    * Color
        * Original:
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/parrt-vancouver.jpg)
        * Kmeans++ copresion:
![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/color_km.jpg)

<a name="rf+km"></a>
##### 4. Advanced topic: RF + Kmeans

> Procedure:

1. RF ‘group’ similar data points.

2. Construct frequency (similarity) matrix

3. Feed similarity matrix to SpectralClustering

> Test on circle data (sklearn vs. RF+Kmeans)

![alt test](https://raw.githubusercontent.com/victorlifan/kmeans/main/img/vs.png)

<a name='mf'></a>
##### 5. Limitations
* Randomness in RF will sometimes result in unexpected cluster labels (accuracy is not as steady).

* Kmeans++ only considers picking the furthest point to its previous centroid. Take k=3 as an example, as a consequence, the 1st and 3rd controls can sometimes be quite close to each other. Is there a way to consider all the previous centroids and pick the furthest point from all the previous centroids? How do we even define the ‘minimum distance’ since each centroid has its own furthest points, one point can't be the furthest to multiple centroids?


<a name="Dataset"></a>
## Dataset

* `Synthetic data set`: A small synthetic data that has a shape of 16*1
* `Multi-dimension data (Circle data 500*2)`: from sklearn.datasets.make_circles
* `Breast cancer`: from sklearn.datasets.load_breast_cancer
* `north-africa-1940s-grey` and `parrt-vancouver.jpg`: from Professor [Terence Parr](https://en.wikipedia.org/wiki/Terence_Parr)

<a name="summary"></a>


<a name="About"></a>
## About
+ [`Jupyter Notebook file`](https://github.com/victorlifan/kmeans/blob/main/kmeans.ipynb): workspace where I performed and tested the works.
+ [`kmeans.py`](https://github.com/victorlifan/kmeans/blob/main/kmeans.py): modularized support functions
* [`kmeans.pdf`](https://github.com/victorlifan/kmeans/blob/main/kmeans.pdf): pdf presentation
+ [`img`](https://github.com/victorlifan/kmeans/tree/main/img): png files were used in this project.

<a name="Software"></a>
## Software used
+ Jupyter Notebook
+ Atom
+ Python 3.9
>   * Numpy
>   * Pandas
>   * Matplotlib
>   * Seaborn
>   * sklearn
>   * statistics
>   * PIL
>   * tqdm


<a name="ref"></a>
## References
* [K-Means Clustering: From A to Z](https://towardsdatascience.com/k-means-clustering-from-a-to-z-f6242a314e9a)
* [ML | K-means++ Algorithm](https://www.geeksforgeeks.org/ml-k-means-algorithm/)
* [Image Segmentation using K Means Clustering](https://www.geeksforgeeks.org/image-segmentation-using-k-means-clustering/)
* [Breiman's website](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm#prox)
