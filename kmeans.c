#include <Python.h>

/*================================ MACROS ==================================*/

/* Frees a given pointer iff it's not NULL*/
#define FREE_MEM(mem) if (NULL != (mem)) { free((mem)); }
/* Frees all-memory allocated, raises PYTHON_ERROR and returns NULL iff 
 * (=in case) condition HOLDS. */
#define FREE_ALL_MEM_IN_CASE(cond) do{ \
        if((cond))  { \
        free_memory(observations, observations_mem_region, clusters, \
            clusters_indices, K, 1);\
        return NULL;}} while(0)
/* Frees all memory */
#define FREE_ALL_MEM() FREE_ALL_MEM_IN_CASE(1)
/* Value of an invalid cluster index, used for initializing the observations */
#define INVALID_CLUSTER (-1)

/*================================ ENUMS ===================================*/
/* An enum that describes the different errors of this program */
typedef enum errors_e
{
    E_SUCCESS = 0,
    E_NO_MEMORY,
    E_INVALID_INDEX,
    E_BAD_VALUE,
    E_UNINITIALIZED = -1
} errors_t;

/*============================ General Helpers ==============================*/
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


typedef struct obs_s
{
    /*
     * data - points to the actual observation's d-vector
     * cluster_index - index of the cluster the observation belongs to
     */
    double * data;
    int cluster_index;
} obs_t;


/*
 * Calculates the squared euclidean distance between 2 points and returns it.
 */
static double euclidean_distance(const double * p, const double * q, int d)
{
    double dis = 0;
    int i = 0;
    dis = 0;

    for (i = 0; i < d; i++)
    {
        dis += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return dis;
}

/*
 * calculates the MU and changes it inplace
 * returns 1 if mu changed in the process, -1 if error occurred
 */
static int calc_mu(cluster_t * cluster, int d)
{
    double * new_mu = NULL;
    int did_cluster_change = 0;
    int i = 0;
    int j = 0;

    new_mu = calloc(d, sizeof(*new_mu));
    if (NULL == new_mu)
    {
        PyErr_Format(PyExc_MemoryError, "Memory Allocation Error");
        return -1;
    }


    for (i = 0; i < cluster->len; i++)
    {
        for (j = 0; j < d; j++)
        {
            new_mu[j] += (cluster->obs_array[i][j]) / (cluster->len);
        }
    }

    /*checks if change has happened:*/
    did_cluster_change = 0;
    for (j = 0; j < d; j++)
    {
        if (new_mu[j] != cluster->mu[j])
        {
            did_cluster_change = 1;
            break;
        }
    }
    free(cluster->mu);
    cluster->mu = new_mu;
    return did_cluster_change;
}

/*
 * frees ALL given pointers
 * raises PYTHON_ERROR iff err_flag
 * returns NULL
 */
static void free_memory(double ** observations, 
                        double * observations_mem_region,
                        cluster_t * clusters, size_t * clusters_indices, 
                        int K, int err_flag)
{
    FREE_MEM(observations_mem_region)
    FREE_MEM(observations)
    if (NULL != clusters)
    {
        int i = 0;
        for (i = 0; i < K; ++i)
        {
            FREE_MEM(clusters[i].mu)
            FREE_MEM(clusters[i].obs_array)
        }
        free(clusters);
    }
    FREE_MEM(clusters_indices)

    /* raises error if needed */
    if (err_flag)
    {
        PyErr_Format(PyExc_RuntimeError, "Exception while running K-Means++");
    }
}

/*========================== Data Structuring Helpers ========================*/

/*
 * @param observations - array of observations
 * @param clusters_indices - array of indices of the chosen observations to be 
 *                           clusters
 * builds an array of cluster_t from the given observations indices
 */
static cluster_t * build_clusters(obs_t * observations, 
                                  const size_t * clusters_indices, int N, 
                                  int K, int d)
{
    int i = 0;
    int j = 0;
    cluster_t * clusters = malloc(sizeof(*clusters) * K);
    if (NULL == clusters)
    {
        PyErr_Format(PyExc_MemoryError, "Memory Allocation Error");
        return NULL;
    }

    for (i = 0; i < K; i++)
    {
        clusters[i].mu = malloc(d * sizeof(*clusters[i].mu));
        clusters[i].obs_array = malloc(N * sizeof(*clusters[i].obs_array));
        if (NULL == clusters[i].mu || NULL == clusters[i].obs_array)
        {
            PyErr_Format(PyExc_MemoryError, "Memory Allocation Error");
            return NULL;
        }

        for (j = 0; j < d; j++)
        {
            /* copies the clusters_indices[i] observation to be the MU */
            clusters[i].mu[j] = observations[clusters_indices[i]].data[j];
        }
    }

    return clusters;
}

/*================== KMeans Original Implementation ===================*/
/*
 * K-Means Implementation
 * @param observations - each pointer, points to the correspondent observation 
 *                       in the memory (observation = d-size array)
 * @param clusters - array of initialized-clusters
 * @return - changes 'clusters' array IN-PLACE, returns 0 iff no errors
 */
static int kmeans_impl(obs_t * observations, cluster_t * clusters, int d, 
                       int K, int N, int MAX_ITER)
{
    int did_cluster_change = 1;
    int iter_count = 0;
    int i = 0;
    int j = 0;

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

            /* append observation pointer to the closest cluster */
            pos = clusters[closest_cluster].len;
            clusters[closest_cluster].obs_array[pos] = observations[i].data;
            clusters[closest_cluster].len++;
            observations[i].cluster_index = closest_cluster;
        }

        /* Recalculating each mu and checks if a change happened */
        did_cluster_change = 0;
        for (i = 0; i < K; i++)
        {
            int cluster_change = calc_mu(&clusters[i], d);
            if (cluster_change < 0)
                return -1;
            else if (1 == cluster_change)
                did_cluster_change = 1;
        }
        iter_count += 1;
    }

    return 0; /* ran without errors */
}

/*========================= Python Integration ============================*/
/*
 * Initializes the observations array, passed from Python.
 * @param obs_lst - Python's list of observations
 * @param observations_mem_region - will contain the entire memory region of the
 *                                  observations
 * @param observations - will contain the observations array
 * @return E_SUCCESS on success, otherwise return the relevant error code. Also,
 *  both on success and on failure, both pointers of observations can be
 *  allocated, so the user should free them in the calling function.
 */
static errors_t init_observations(PyObject * obs_lst, int N, int d, 
                                  double ** observations_mem_region, 
                                  obs_t ** observations)
{
    int i = 0;
    int j = 0;

    *observations_mem_region = malloc(N * d, sizeof(**observations_mem_region));
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
            PyObject * o_val = PyTuple_GetItem(obs_vector, j);
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

/*
 * Initializes the cluster indices array, passed from Python.
 * @param indices_lst - Python's list of clusters indices
 * @param clusters_indices - will contain the array of cluters indices
 * @return E_SUCCESS on success, otherwise return the relevant error code. Also,
 *  both on success and on failure, clusters_indices can be allocated, so the 
 *  user should free it in the calling function.
 */

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

/*
 * Convert the results from a C array to Python's list
 * @param clusters_lst - will contain the results as Python's list
 * @param observations - the observations and their cluster indices
 * @return E_SUCCESS on success, otherwise return the relevant error code.
 */
static errors_t convert_result_clusters(PyObject ** clusters_lst, int N, 
                                        obs_t * observations)
{
    int i = 0;

    *clusters_lst = PyList_New(N);
    if (NULL == *clusters_lst)
    {
        return E_NO_MEMORY;
    }

    for (i = 0; i < N; i++)
    {
        if (PyList_SetItem(*clusters_lst, i, 
                           PyLong_FromLong(observations[i].cluster_index)) < 0)
        {
            Py_DecRef(*clusters_lst);
            return E_BAD_VALUE;
        }
    }

    return E_SUCCESS;
}

/*
 * Convert the error code to the relevant error message in Python
 */
static PyObject * error_msg(errors_t rc)
{
    switch (rc)
    {
    case E_NO_MEMORY:
        return PyErr_Format(PyExc_MemoryError, "Memory Allocation Error");
        break;
    case E_INVALID_INDEX:
        return PyErr_Format(PyExc_ValueError, "Invalid Index");
        break;
    case E_BAD_VALUE:
        return PyErr_Format(PyExc_ValueError, "Bad value");
        break;
    default:
        return PyErr_Format(PyExc_ValueError, "Unknown Error");
    }
}

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
    int i, j, pos;
    errors_t rc = E_UNINITIALIZED;

    /* Processing Arguments */
    if (!PyArg_ParseTuple(args, "OOiiii; Expected args are: observations, centers indices, K, N, d, MAX_ITER",
                          &obs_lst, &indices_lst, &K, &N, &d, &MAX_ITER))
        return PyErr_Format(PyExc_ValueError, "Input is not valid");
    if (!PyList_Check(obs_lst) || !PyList_Check(indices_lst))
        return PyErr_Format(PyExc_ValueError, "Input is not valid");

    /*
     * Process Observations: python's obs_lst ---> observations_mem_region
     * observations_mem_region - keeps observations in a contiguous memory block
     * observations - an array of pointers; each pointer, points to the 
     *                correspondent observation in the memory
     */
    rc = init_observations(obs_lst, N, d, &observations_mem_region, 
                           &observations)
    if (E_SUCCESS != rc)
    {
        FREE_ALL_MEM();
        return error_msg(rc);
    }

    /* Process Indices: python's indices_lst ---> clusters_indices */
    rc = init_cluster_indices(indices_lst, K, &clusters_indices);
    if (E_SUCCESS != rc)
    {
        FREE_ALL_MEM();
        return error_msg(rc);
    }

    /* Build Clusters from given indices */
    /* TODO: Should this be converted to the format of init_cluster_indices above or not? */
    /* TODO: Should we just use PyErr_Format and drop the enum? */
    clusters = build_clusters(observations, clusters_indices, N, K, d);
    FREE_ALL_MEM_IN_CASE(NULL == clusters);

    /* Runs K-Means Implementation, it will mutate 'clusters' array 
     * (breaks program if it raised an error */
    FREE_ALL_MEM_IN_CASE(kmeans_impl(observations, clusters, d, K, 
                                     N, MAX_ITER) < 0);

    /* pack clusters_lst to python-list */
    rc = convert_result_clusters(&clusters_lst, N, observations);
    if (rc != E_SUCCESS)
    {
        FREE_ALL_MEM();
        return error_msg(rc);
    }

    /* Free all memory */
    free_memory(observations, observations_mem_region, clusters, 
                clusters_indices, K, 0);
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