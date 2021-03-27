# Clustering Algorithms - KMeans, Spectral Clustering

## Description
Pure Implementations of the popular **Clustering algorithms** - Kmeans and the 
Normalized Spectral Clustering. 
Full demonstration of their utilization, 
and comparison between the two algorithm.

Coded with Python (Numpy, Matplotlib) and C (CAPI).

Was written as a final project for Software Project course.


  
## Demo:
  Visualization created by `$python main.py 5 100`:

  <img src="https://github.com/matanbt/final-project/blob/master/docs/demo.png" width='500px'>


## Files and Modules
 - **main.py:** TODO


## Usage
### Prior Setup:
   Running the project requires building `mykmeanssp` module from `kmeans.c`, 
   which can be done by the following command:
   
`$ python invoke build`


### Usage 1 - Full Demonstration:
 - #### Description:
   Generates data points, clusters it using both algorithms (i.e. Kmeans and Spectral Clustering), 
   creates 3 output files:
   - `data.txt` - The generated points and the corresponding center of each. 
   - `clusters.txt` - **First line** shows the K that was used. 
     
     The **next K lines** shows the indices clustered to each label by Spectral Clustering. 
     The **later k lines** shows the indices as above by Kmeans.
   - `clusters.pdf` - provides a visualization of the clustering, as well as results summary and score for each clustering.
 - #### Run Options:
    - `$ python main.py 0 0 --random` - casts *n*, *k* to generate data with, 
      then calculates *heuristic-k* using Eigengap Method and uses *this* k to run the clustering algorithms.
    - `$ python main.py {k} {n}` - when {k}, {n} to be replaced with integers indicates the desired values of k and n.
    generates data and clusters using the given k and n.
    - **With given data** - one can provide an object of `params`, 
      matrix of `points` , array of `centers` and run *Full Demo* using the following Python line: 
      
        `main.run_clustering(params, points, centers)`
      
    - **Note:** When not given, the dimension *d* will be cast from {2,3}
    

### Usage 2 - Handy Modules Utilization:
   - #### Description:
   - #### Run Options:
   - TODO

  

