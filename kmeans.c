#include <Python.h>
#include <math.h>

/*================================ MACROS ==================================*/

/* Frees a given pointer iff it's not NULL */
#define FREE_MEM(mem) if (NULL != (mem)) { free((mem)); }
/* Frees all of the memory the program allocated */
#define FREE_ALL_MEM() do{ \
        free_memory(observations, observations_mem_region, clusters, \
            clusters_indices, K);} while(0);
/* Fail the program and free memory if condition `cond` happens */
#define FAIL_IF(cond) if ((cond)) { FREE_ALL_MEM(); return error_msg(rc); }
/* Value of an invalid cluster index, used for initializing the observations */
#define INVALID_CLUSTER (-1)
/* How accurate the equals sign will be */
#define EPSILON (0.0001)

/*================================ ENUMS ===================================*/

/* An enum that describes the different errors of this program */
typedef enum errors_e
{
    /* Success */
    E_SUCCESS = 0,
    /* Failed to allocate memory */
    E_NO_MEMORY,
    /* Accessing invalid index in a list */
    E_INVALID_INDEX,
    /* Trying to parse a bad value */
    E_BAD_VALUE,
    /* Invalid input from the caller of this module */
    E_INVALID_INPUT,
    /* Internal value, used to represent an uninitialized result variables */
    E_UNINITIALIZED = -1
} errors_t;

/*=============================== STRUCTS ==================================*/

/* A struct that holds data related to the clusters */
typedef struct cluster_s
{
    /*
     * mu - point to the d-vector represents the mu of the cluster
     * obs_array - each element of the array points to an observation BELONGS 
     *             to the cluster
     * len - keeps track of the length of the array (mentioned above)
     */
    double * mu;
    double ** obs_array;
    int len;
} cluster_t;

/* A struct that holds data related to each observation */
typedef struct obs_s
{
    /*
     * data - points to the actual observation's d-vector
     * cluster_index - index of the cluster the observation belongs to
     */
    double * data;
    int cluster_index;
} obs_t;

/*======================== FUNCTION DECLARATIONS ===========================*/

/*
 * Calculates the squared euclidean distance between 2 points.
 * @param p: First point
 * @param q: Second point
 * @param d: Dimension of points
 * @returns: the squared euclidean distance between the points.
 */
static double euclidean_distance(const double * p, const double * q, int d);

/*
 * Calculates MU and changes it in-place.
 * @param cluster: The cluster that we want to calculate its new MU
 * @param d: Dimension of points in the cluster
 * @param did_cluster_change: Will be 1 if MU changed, 0 otherwise.
 * @returns: E_SUCCESS on success, otherwise return the relevant error code.
 */
static errors_t calc_mu(cluster_t * cluster, int d, int * did_cluster_change);
                               
/*
 * Initializes the observations array, passed from Python.
 * @param obs_lst: Python's list of observations
 * @param N: Number of observations
 * @param d: Dimension of each point in the observations
 * @param observations_mem_region: will contain the entire memory region of the
 *                                  observations
 * @param observations: will contain the observations array
 * @return E_SUCCESS on success, otherwise return the relevant error code. Also,
 *  both on success and on failure, both pointers of observations can be
 *  allocated, so the user should free them in the calling function.
 */
static errors_t init_observations(PyObject * obs_lst, int N, int d, 
                                  double ** observations_mem_region, 
                                  obs_t ** observations);

/*
 * Initializes the cluster indices array, passed from Python.
 * @param indices_lst: Python's list of clusters indices
 * @param K: Number of clusters
 * @param clusters_indices: will contain the array of clusters indices
 * @return E_SUCCESS on success, otherwise return the relevant error code. Also,
 *  both on success and on failure, clusters_indices can be allocated, so the 
 *  user should free it in the calling function.
 */
static errors_t init_cluster_indices(PyObject * indices_lst, int K, 
                                     size_t ** clusters_indices);

/*
 * Allocates memory for the clusters and initialize them.
 * @param observations: The observations array
 * @param clusters_indices: The array of indices of the chosen observations to be
 *                           clusters
 * @param N: Number of observations
 * @param K: Number of clusters
 * @param d: Dimension of points in the clusters
 * @param clusters: Will contain the new clusters array
 * @returns: E_SUCCESS on success, otherwise return the relevant error code.
 * @note: If an error occurred, there is no need to free the clusters, the
 *        function does that.
 */
static errors_t build_clusters(obs_t * observations, 
                               const size_t * clusters_indices, int N, 
                               int K, int d, cluster_t ** clusters);

/*
 * Implementation of the KMeans algorithm.
 * @param observations: Array of the observations
 * @param clusters: Array of the clusters
 * @param d: Dimension of points of the observations and clusters
 * @param K: Number of clusters
 * @param N: Number of observations
 * @param MAX_ITER: Maximum times the algorithm will iterate over the clusters
 * @returns: E_SUCCESS on success, otherwise return the relevant error code.
 */
static errors_t kmeans_impl(obs_t * observations, cluster_t * clusters, int d, 
                            int K, int N, int MAX_ITER);

/*
 * Convert the results from a C array to Python's list
 * @param clusters_lst: will contain the results as Python's list
 * @param N: Number of observations
 * @param observations: the observations and their cluster indices
 * @return E_SUCCESS on success, otherwise return the relevant error code.
 */
static errors_t convert_result_clusters(PyObject ** clusters_lst, int N, 
                                        obs_t * observations);

/*
 * Convert the error code to the relevant error message in Python.
 * @param rc: The C error code
 * @returns: Always NULL.
 */
static PyObject * error_msg(errors_t rc);

/*
 * Frees the memory of the program,
 * @param observations: The observations array
 * @param observations_mem_region: The memory region the observations use
 * @param clusters: The clusters array
 * @param clusters_indices: The clusters indices array
 * @param K: Number of clusters we have
 */
static void free_memory(obs_t * observations,
                        double * observations_mem_region,
                        cluster_t * clusters, size_t * clusters_indices, int K);

/*
 * K-Means(observations, centroids_indices, K, N, d, MAX_ITER)
 * gets 6 positional arguments:
 * @param 1: observations: N-sized List with D-sized tuples (with float values)
 * @param 2: centroids_indices: K-sized List of indices (integer) indicates the 
 *                              chosen observations from the list above
 * @params 3-6: K,N,d,MAX_ITER: k-means algorithm arguments
 * @precondition: input is valid
 * @return N-sized list, where each element maps between the observation and its
 *  cluster index. On error, return NULL
 */
static PyObject * kmeans_api(PyObject * self, PyObject * args);

/*=============================== FUNCTIONS ================================*/

static double euclidean_distance(const double * p, const double * q, int d)
{
    double dis = 0;
    int i = 0;

    for (i = 0; i < d; i++)
    {
        dis += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return dis;
}

static errors_t calc_mu(cluster_t * cluster, int d, int * did_cluster_change)
{
    double * new_mu = NULL;
    int i = 0;
    int j = 0;

    new_mu = calloc(d, sizeof(*new_mu));
    if (NULL == new_mu)
    {
        return E_NO_MEMORY;
    }


    for (i = 0; i < cluster->len; i++)
    {
        for (j = 0; j < d; j++)
        {
            new_mu[j] += (cluster->obs_array[i][j]) / (cluster->len);
        }
    }

    /*checks if change has happened:*/
    *did_cluster_change = 0;
    for (j = 0; j < d; j++)
    {
        if (euclidean_distance(new_mu, cluster->mu, d) > EPSILON)
        {
            *did_cluster_change = 1;
            break;
        }
    }
    free(cluster->mu);
    cluster->mu = new_mu;
    return E_SUCCESS;
}

static errors_t init_observations(PyObject * obs_lst, int N, int d, 
                                  double ** observations_mem_region, 
                                  obs_t ** observations)
{
    int i = 0;
    int j = 0;

    *observations_mem_region = malloc(N * d * sizeof(**observations_mem_region));
    *observations = malloc(N * sizeof(**observations));
    if (NULL == *observations_mem_region || NULL == *observations)
    {
        return E_NO_MEMORY;
    }

    for (i = 0; i < N; i++)
    {
        /* Keeps pointer of the i-th observation */
        (*observations)[i].data = (*observations_mem_region) + i * d;

        /* Copies the current vector */
        PyObject * obs_vector = PyList_GetItem(obs_lst, i);
        if (NULL == obs_vector)
        {
            return E_INVALID_INDEX;
        }
        for (j = 0; j < d; j++)
        {
            PyObject * o_val = PyList_GetItem(obs_vector, j);
            if (NULL == o_val)
            {
                return E_INVALID_INDEX;
            }
            double val = PyFloat_AsDouble(o_val);
            if (-1 == val && PyErr_Occurred())
            {
                return E_BAD_VALUE;
            }
            (*observations)[i].data[j] = val;
        }
        
        /* Set the initial cluster to be invalid */
        (*observations)[i].cluster_index = INVALID_CLUSTER;
    }
    
    return E_SUCCESS;
}

static errors_t init_cluster_indices(PyObject * indices_lst, int K, 
                                     size_t ** clusters_indices)
{
    int i = 0;

    *clusters_indices = malloc(sizeof(**clusters_indices) * K);
    if (NULL == *clusters_indices)
    {
        return E_NO_MEMORY;
    }

    for (i = 0; i < K; i++)
    {
        PyObject * o_index = PyList_GetItem(indices_lst, i);
        if (NULL == o_index)
        {
            return E_INVALID_INDEX;
        }
        size_t index = PyLong_AsSize_t(o_index);
        if ((size_t)-1 == index && PyErr_Occurred())
        {
            return E_BAD_VALUE;
        }
        (*clusters_indices)[i] = index;
    }
    
    return E_SUCCESS;
}

static errors_t build_clusters(obs_t * observations, 
                               const size_t * clusters_indices, int N, 
                               int K, int d, cluster_t ** clusters)
{
    int i = 0;
    int j = 0;

    /* Allocate memory for the clusters array */
    cluster_t * temp_clust = calloc(K, sizeof(*temp_clust));
    if (NULL == clusters)
    {
        return E_NO_MEMORY;
    }

    for (i = 0; i < K; i++)
    {
        temp_clust[i].mu = calloc(d, sizeof(*temp_clust[i].mu));
        temp_clust[i].obs_array = calloc(N, sizeof(*temp_clust[i].obs_array));
        if (NULL == temp_clust[i].mu || NULL == temp_clust[i].obs_array)
        {
            /* Free memory in case of an error */
            for (j = 0; j < i; j++)
            {
                FREE_MEM(temp_clust[j].mu);
                FREE_MEM(temp_clust[i].obs_array);
            }
            FREE_MEM(temp_clust);
            return E_NO_MEMORY;
        }

        for (j = 0; j < d; j++)
        {
            /* Copy the clusters_indices[i] observation to be MU */
            temp_clust[i].mu[j] = observations[clusters_indices[i]].data[j];
        }
    }

    /* Success - return the new clusters array */
    *clusters = temp_clust;
    return E_SUCCESS;
}

static errors_t kmeans_impl(obs_t * observations, cluster_t * clusters, int d, 
                            int K, int N, int MAX_ITER)
{
    int did_cluster_change = 1;
    int cluster_change_status = 0;
    int iter_count = 0;
    int i = 0;
    int j = 0;
    errors_t rc = E_UNINITIALIZED;

    while ((1 == did_cluster_change) && (iter_count < MAX_ITER))
    {
        /* Reset clusters */
        for (i = 0; i < K; i++)
        {
            clusters[i].len = 0;
        }

        /* Running over all observations */
        for (i = 0; i < N; i++)
        {
            int closest_cluster = 0;
            int pos;
            double closest_distance = euclidean_distance(observations[i].data, 
                                                         clusters[0].mu, d);

            for (j = 1; j < K; j++)
            {
                double curr_distance = euclidean_distance(observations[i].data, 
                                                          clusters[j].mu, d);
                if (curr_distance < closest_distance)
                {
                    closest_cluster = j;
                    closest_distance = curr_distance;
                }
            }

            /* Append observation pointer to the closest cluster */
            pos = clusters[closest_cluster].len;
            clusters[closest_cluster].obs_array[pos] = observations[i].data;
            clusters[closest_cluster].len++;
            observations[i].cluster_index = closest_cluster;
        }

        /* Recalculating each mu and checks if a change happened */
        did_cluster_change = 0;
        for (i = 0; i < K; i++)
        {
            cluster_change_status = 0;
            rc = calc_mu(&clusters[i], d, &cluster_change_status);
            if (E_SUCCESS != rc)
            {
                return rc;
            }
            else if (1 == cluster_change_status)
            {
                did_cluster_change = 1;
            }
        }
        iter_count += 1;
    }

    return E_SUCCESS;
}

static errors_t convert_result_clusters(PyObject ** clusters_lst, int N, 
                                        obs_t * observations)
{
    int i = 0;
    PyObject * val = NULL;

    *clusters_lst = PyList_New(N);
    if (NULL == *clusters_lst)
    {
        return E_NO_MEMORY;
    }

    for (i = 0; i < N; i++)
    {
        val = PyLong_FromLong(observations[i].cluster_index);
        if (NULL == val)
        {
            Py_DecRef(*clusters_lst);
            return E_BAD_VALUE;
        }
        if (PyList_SetItem(*clusters_lst, i, val) < 0)
        {
            Py_DecRef(*clusters_lst);
            return E_INVALID_INDEX;
        }
    }

    return E_SUCCESS;
}

static PyObject * error_msg(errors_t rc)
{
    switch (rc)
    {
    case E_NO_MEMORY:
        return PyErr_Format(PyExc_MemoryError, "Memory allocation error");
        break;
    case E_INVALID_INDEX:
        return PyErr_Format(PyExc_IndexError, "Invalid list index");
        break;
    case E_BAD_VALUE:
        return PyErr_Format(PyExc_ValueError, "Couldn't parse given value");
        break;
    case E_INVALID_INPUT:
        return PyErr_Format(PyExc_ValueError, "Invalid input from the user");
        break;
    default:
        return PyErr_Format(PyExc_Exception, "Unknown error");
    }
}

static void free_memory(obs_t * observations, 
                        double * observations_mem_region,
                        cluster_t * clusters, size_t * clusters_indices, int K)
{
    FREE_MEM(observations_mem_region);
    FREE_MEM(observations);
    if (NULL != clusters)
    {
        int i = 0;
        for (i = 0; i < K; ++i)
        {
            FREE_MEM(clusters[i].mu);
            FREE_MEM(clusters[i].obs_array);
        }
        FREE_MEM(clusters);
    }
    FREE_MEM(clusters_indices);
}

static PyObject * kmeans_api(PyObject * self, PyObject * args)
{
    /* Process and validate arguments */
    PyObject * obs_lst = NULL;
    PyObject * indices_lst = NULL;
    PyObject * clusters_lst = NULL;
    int K, N, d, MAX_ITER;
    double * observations_mem_region = NULL;
    obs_t * observations = NULL;
    size_t * clusters_indices = NULL;
    cluster_t * clusters = NULL;
    errors_t rc = E_UNINITIALIZED;

    /* Processing Arguments */
    if (!PyArg_ParseTuple(args, "OOiiii; Expected args are: observations, centers indices, K, N, d, MAX_ITER",
                          &obs_lst, &indices_lst, &K, &N, &d, &MAX_ITER))
    {
        rc = E_INVALID_INPUT;
        return error_msg(rc);
    }
    if (!PyList_Check(obs_lst) || !PyList_Check(indices_lst))
    {
        rc = E_INVALID_INPUT;
        return error_msg(rc);
    }

    /*
     * Process Observations: python's obs_lst ---> observations_mem_region
     * observations_mem_region - keeps observations in a contiguous memory block
     * observations - an array of pointers; each pointer, points to the 
     *                correspondent observation in the memory
     */
    rc = init_observations(obs_lst, N, d, &observations_mem_region, 
                           &observations);
    FAIL_IF(E_SUCCESS != rc);
    
    /* Process Indices: python's indices_lst ---> clusters_indices */
    rc = init_cluster_indices(indices_lst, K, &clusters_indices);
    FAIL_IF(E_SUCCESS != rc);

    /* Build Clusters from given indices */
    rc = build_clusters(observations, clusters_indices, N, K, d, &clusters);
    FAIL_IF(E_SUCCESS != rc);

    /* Runs K-Means Implementation, it will mutate 'clusters' array 
     * (breaks program if it raised an error */
    rc = kmeans_impl(observations, clusters, d, K, N, MAX_ITER);
    FAIL_IF(E_SUCCESS != rc);

    /* pack clusters_lst to python-list */
    rc = convert_result_clusters(&clusters_lst, N, observations);
    FAIL_IF(E_SUCCESS != rc);

    /* Free all memory */
    free_memory(observations, observations_mem_region, clusters, 
                clusters_indices, K);
    return clusters_lst;
}

/*========================= Module Configuration ============================*/
PyDoc_STRVAR(kmeans_doc, "kmeans(observations, centroids_indices, K, N, d, MAX_ITER)\n"
                         "--\n\n"
                         " :param observations: N-sized List with D-sized tuples (with float values)\n"
                         " :param centroids_indices: K-sized List of indices (integer) indicates the chosen observations from the list above\n"
                         " :params 3-6: K, N, d, MAX_ITER: K-Means algorithm arguments\n"
                         " :precondition: Input is valid \n"
                         " :returns: N-sized List, each element represents the cluster of its index");
static PyMethodDef capiMethods[] = {
        {"kmeans", (PyCFunction) kmeans_api, METH_VARARGS, kmeans_doc},
        {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "mykmeanssp",
        NULL,
        -1,
        capiMethods
};

PyMODINIT_FUNC
PyInit_mykmeanssp(void)
{
    return PyModule_Create(&moduledef);
}