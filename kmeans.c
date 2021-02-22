#include <Python.h>

/*================================ MACROS ==================================*/

/* frees given pointer iff it's NOT NULL*/
#define FREE_MEM(mem) if (NULL != (mem)) { free((mem)); }
/* frees all-memory allocated, raises PYTHON_ERROR and returns NULL iff (=in case) condition HOLDS. */
#define FREE_ALL_MEM_IN_CASE(cond) do{ \
        if((cond))  { \
        free_memory(observations, observations_arr, clusters, clusters_indices, K, 1);\
        return NULL;}} while(0)


/*============================ General Helpers ==============================*/
typedef struct
{
    /*
     * mu - point to the d-vector represents the mu of the cluster
     * obs_array - each element of the array is a points to an observation BELONGS to the cluster
     * len - keeps track of the length of the array (mentioned above)
     */
    double * mu;
    double ** obs_array;
    int len;
} cluster_t;

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
static void free_memory(double ** observations, double * observations_arr,
                              cluster_t * clusters, size_t * clusters_indices, int K, int err_flag)
{
    FREE_MEM(observations_arr)
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
 * @param observations - each pointer, points to the correspondent observation in the memory
 * @param clusters_indices - array of the chosen observations to be clusters by indexes
 * builds an array of cluster_t from the given observations indices
 */
static cluster_t * build_clusters(double ** observations, const size_t * clusters_indices, int N, int K, int d)
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
            /* copies the #clusters_indices[i] observation to be the current MU */
            clusters[i].mu[j] = observations[clusters_indices[i]][j];
        }
    }

    return clusters;
}

/*========================= KMeans Original Implementation ============================*/
/*
 * K-Means Implementation from HW1 - modified to receive observations and clusters
 * @param observations - each pointer, points to the correspondent observation in the memory (observation = d-size array)
 * @param clusters - array of initialized-clusters
 * @return - changes 'clusters' array IN-PLACE, returns 0 iff no errors
 */
static int kmeans_impl(double ** observations, cluster_t * clusters, int d, int K, int N, int MAX_ITER)
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
            double closest_distance = euclidean_distance(observations[i], clusters[0].mu, d);

            for (j = 1; j < K; j++)
            {
                double curr_distance = euclidean_distance(observations[i], clusters[j].mu, d);
                if (curr_distance < closest_distance)
                {
                    closest_cluster = j;
                    closest_distance = curr_distance;
                }
            }

            /* append observations pointer to the closest cluster */
            pos = clusters[closest_cluster].len;
            clusters[closest_cluster].obs_array[pos] = observations[i];
            clusters[closest_cluster].len++;
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
 * K-Means(observations, centroids_indices,
 * gets 6 positional arguments:
 * @param 1: observations: N-sized List with D-sized tuples (with float values)
 * @param 2: centroids_indices: K-sized List of indices (integer) indicates the chosen observations from the list above
 * @params 3-6: K,N,d,MAX_ITER: k-means algorithm arguments
 * @precondition: input is valid
 * @return K-sized List with D-sized tuples, each tuples represent a final centroid
 */

static PyObject * kmeans_api(PyObject * self, PyObject * args)
{
    /* Process and validate arguments */
    PyObject * obs_lst, * indices_lst;
    int K, N, d, MAX_ITER;
    double * observations_arr = NULL;
    double ** observations = NULL;
    size_t * clusters_indices = NULL;
    cluster_t * clusters = NULL;

    int i, j;
    if (!PyArg_ParseTuple(args, "OOiiii; Y'all Better check your args before you run me!",
                          &obs_lst, &indices_lst, &K, &N, &d, &MAX_ITER))
        return PyErr_Format(PyExc_ValueError, "Input is not valid");
    if (!PyList_Check(obs_lst) || !PyList_Check(indices_lst))
        return PyErr_Format(PyExc_ValueError, "Input is not valid");

    /*
     * Process Observations: python's obs_lst ---> observations_arr
     * observations_arr - keeps observations in a contiguous memory block
     * observations - an array of pointers; each pointer, points to the correspondent observation in the memory
     */
    observations_arr = (double *) malloc(N * d * sizeof(*observations_arr));
    observations = (double **) malloc(N * sizeof(*observations));
    if (NULL == observations_arr || NULL == observations)
    {
        return PyErr_Format(PyExc_MemoryError, "Memory Allocation Error");
    }

    for (i = 0; i < N; i++)
    {
        /* keeps pointer of the i-th observation */
        observations[i] = observations_arr + i * d;
        /* gets the current vector (borrowed ref) */
        PyObject * obs_vector = PyList_GetItem(obs_lst, i);
        FREE_ALL_MEM_IN_CASE(!PyTuple_Check(obs_vector));
        /* iterates on the vector, and copies its to the array */
        for (j = 0; j < d; j++)
        {
            PyObject * o_val = PyTuple_GetItem(obs_vector, j);
            double val = PyFloat_AsDouble(o_val);
            FREE_ALL_MEM_IN_CASE(-1 == val && PyErr_Occurred());
            observations[i][j] = val;
        }
    }

    /* Process Indices: python's indices_lst ---> clusters_indices */
    clusters_indices = malloc(sizeof(*clusters_indices) * K);
    FREE_ALL_MEM_IN_CASE(NULL == clusters_indices);

    for (i = 0; i < K; i++)
    {
        PyObject * o_index = PyList_GetItem(indices_lst, i);
        size_t index = PyLong_AsSize_t(o_index);
        FREE_ALL_MEM_IN_CASE(index == (size_t) -1 && PyErr_Occurred());
        clusters_indices[i] = index;
    }

    /* Build Clusters from given indices */
    clusters = build_clusters(observations, clusters_indices, N, K, d);
    FREE_ALL_MEM_IN_CASE(NULL == clusters);


    /* Runs K-Means Implementation, it will mutate 'clusters' array (breaks program if it raised an error */
    FREE_ALL_MEM_IN_CASE(kmeans_impl(observations, clusters, d, K, N, MAX_ITER) < 0);


    /* Pack clusters (=the MU value of each) in Python-List */
    PyObject * clusters_lst = PyList_New(K);
    for (i = 0; i < K; i++)
    {
        /* Builds a vector (=tuple) that'll represent each MU */
        PyObject * mu_vector = PyTuple_New(d);
        for (j = 0; j < d; j++)
        {
            if (PyTuple_SetItem(mu_vector, j, PyFloat_FromDouble(clusters[i].mu[j])) < 0)
            {
                Py_DecRef(mu_vector);
                Py_DecRef(clusters_lst);
                FREE_ALL_MEM_IN_CASE(1);
            }
        }
        /* adds mu_vector to the list, also steals its reference - so we don't have to decref' it */
        if (PyList_SetItem(clusters_lst, i, mu_vector) < 0)
        {
            Py_DecRef(mu_vector);
            Py_DecRef(clusters_lst);
            FREE_ALL_MEM_IN_CASE(1);
        }
    }

    /* Free all memory */
    free_memory(observations, observations_arr, clusters, clusters_indices, K, 0);

    return clusters_lst;
}


/*========================= Module Configuration ============================*/
PyDoc_STRVAR(kmeans_doc, "kmeans(observations, centroids_indices, K, N, d, MAX_ITER)\n"
                         "--\n\n"
                         " :param observations: N-sized List with D-sized tuples (with float values)\n"
                         " :param centroids_indices: K-sized List of indices (integer) indicates the chosen observations from the list above\n"
                         " :params 3-6: K, N, d, MAX_ITER: K-Means algorithm arguments\n"
                         " :precondition: Input is valid \n"
                         " :returns: K-sized List with D-sized tuples, each tuple represent a final centroid");
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