# Clustering Algorithms

## Description
Pure Implementations of the popular Clustering algorithms - Kmeans and the 
Normalized Spectral Clustering. 
Full demonstration of their utilization, 
and comparison between the two algorithm.

Coded with Python (Numpy, Matplotlib) and C (CAPI).

Was written as a final project for Software Project course.


  
## Demo:
#### Images:
  - Home:
  <img src="https://github.com/matanbt/final-project/blob/master/img/" width='500px'>



## Files and Modules
 - **main.py:** a


## Usage
### Setup:
   Using the modules requires building `mykmeanssp` from `kmeans.c`, 
   which can be done by the following command.
   
   `$ python invoke build`
### Usage:
 - ####Full Demonstration:
   Generates data points, clusters it using both algorithms (i.e. Kmeans and Spectral Clustering), 
   creates 3 output files:
   - `data.txt` - The generated points and the corresponding center of each. 
   - `clusters.txt` - **First line** shows the K that was used. 
     
     The **next K lines** shows the indices clustered to each label by Spectral Clustering. 
     The **later k lines** shows the indices as above by Kmeans.
   - `clusters.pdf` - provides a visualization of the clustering, as well as results summary and score for each clustering.
 - #### Run Options:
    - `$ python main.py 0 0 --random` - casts *n*, *k* to generate data with, 
      then calculates *heuristic-k* using Eigengap Method and uses this k to run the clustering algorithms.
    - `$ python main.py {k} {n}` - when {k}, {n} to be replaced with integers indicates the desired value of k and n.
      
    - **Note:** in both options, the dimension d with be cast from {2,3}
#### Handy Modules Utilization:
  - **Full Demonstration**
  - **Database:** Set Postgres' sql database with the following settings `postgresql://postgres:123456@localhost:5000/piggy `, 
  alternatively change the following line in `piggy/__init__.py`: 
    ``` 
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:123456@localhost:5000/piggy'
    ```
  - Then, in order to create the needed sql tables, run the following command from the project's dir: 
  ```
  $ python dev_scripts.py create_tables 
  ```
  - **Run:** 
  ```
  $ python run.py
  ```
  

