# Clustering Algorithms - KMeans, Spectral Clustering

## Description
Pure Implementations of the popular **Clustering algorithms** - Kmeans and the 
Normalized Spectral Clustering. 
Full demonstration of their utilization, 
and comparison between the two algorithm.

Coded with Python (Numpy, Matplotlib) and C (CAPI).

Was written as a final project for Software Project course.


  
## Demo
Visualization created by `$ python main.py 5 100`:
  
  <img src="https://github.com/matanbt/final-project/blob/master/Docs/demo.png" width='500px'>


## Files and Modules
#### Core Modules and Files:
   * **config.py:** Program's configurations and constants.
   * **initialization.py:** Handles the input from the user, and Data generation.
   * **main.py:** The main module of the program. Glues everything together. Can also be used as a runnable script.
   * **kmeans.c:** CAPI extension of the KMeans algorithm implementation.
   * **kmeans_pp.py:** KMeans++ initialization algorithm, and caller of the CAPI module.
   * **linalg.py:** Various linear algebra calculations. 
     
        In particular, Modified Gram Schmidt, QR Iterations, Eigengap Heuristic.
   * **output_data.py:** Results processing and outputting. 
     
     In particular, prints informative messages,
   calculates summary from the results (e.g. Jaccard), outputs the final results to the output files.
   * **spectral_clustering.py:** Spectral clustering algorithm implementation.
   
#### Additional Modules:
   * **setup.py:**  Installation of Kmeans' CAPI extension.
   * **tasks.py:** Provides comfortable CLI for interaction with the project.


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
    - `$ python main.py 0 0 --random` : casts *n*, *k* to generate data with, 
      then calculates *heuristic-k* using Eigengap Method and uses *this* k to run the clustering algorithms.
    - `$ python main.py {k} {n}` : when {k}, {n} to be replaced with integers indicates the desired values of k and n.
    generates data and clusters using the given k and n.
    - **Using Invoke:** With `$ python -m invoke run {k} {n} {--Random || --no-Random}`, 
      following the same logic for `main.py` arguments.
    - **With given TXT:** Use `$ python -m invoke run {fname} {k} {--random || --no-random}`, to cluster a specific data-set given by the txt file`fname`. 
    - **With given Data** : one can provide an object of `params`, 
      matrix of `points` , array of `centers` and run *Full Demo* using the following Python line: 
      
        `main.run_clustering(params, points, centers)`
      

    - **Note:** When not given, the dimension *d* will be cast from {2,3}
   
## Max Capacities and Where To Find Them
 - As part of the project we were asked to analyze the runtime of our implementation depending on various input sizes.
   In particular, the instructions defined **Max Capacity** to be the {n,k} that requires exactly 4 minutes and 59 seconds.
 - In our way to obtain these much desired {n,k}, we've run (a lot of) time measurements and visualized them with plots and heat-maps.
   
   In the meantime we've observed the following:
   - Runtime differences between data of **2** and **3** **dimensions** are minor and neglectable.
   - When we fixed *n*, and tested the runtime as function of *k*, as well as when we put *n* through the same process;
     We've observed that ***k*'s impact** was minor in comparison to *n*'s impact.
  - From these empiric results, we concluded that *n* (as one should expect) is the **major factor** influencing our runtime.
  - *Side Note:* Looking for the justifications to these interesting empiric results we found out that **'QR Iterations'** algorithm (and method in particular) was 
    THE biggest time consumer in every run. A result which, of course, matches the fact that *n* is the major runtime factor. 
  - Using variation of binary search with a heuristic function we defined, we found what we consider a good approximation to the *Max Capacity*. 
    
   `max_capacity_n := 470, max_capacity_k := 350`
  - **Important Note:** All these runtime-tests were done around February-2021 on TAU's server, NOVA. Lately we've noticed that 
    the server started acting slowly, inconsistently and unpredictably. 
    Our conclusions are accurate to the time they were measured. 
    
    However, we believe that **in average** (on a very big bunch of tests) our calculation will still hold.