"""
----- Main Module -----
"""
import user_input


def main():
    args = user_input.get_args()
    if user_input.check_user_input(args):
        params, points, clusters = user_input.generate_points(args)


if __name__ == '__main__':
    main()
    
# TODO: Find a new home for this code:
import numpy as np

def print_clusters(k, labeled_obs_nsc, labeled_obs_km):
    """
    :param labeled_obs_nsc: gets np-array, in which each index represent an obs and each element is its cluster
    :param labeled_obs_km: same as above
    :return: formatted string to be printed to  clusters.txt
    """
    #TODO - what if the k's are different??

    s = f"{k}"

    def add_each_clusters(labeled_obs):
        _s = ""
        inds_nsc = np.argsort(labeled_obs)
        curr_cluster = -1
        for i in inds_nsc:
            if labeled_obs[i] == curr_cluster:
                #still printing the same cluster --> in the same line
                _s += ","
            else:
                # break line, and set new cluster
                _s += "\n"
                curr_cluster = labeled_obs[i]
            _s += str(i)
        return _s

    s += add_each_clusters(labeled_obs_nsc)
    s += add_each_clusters(labeled_obs_km)
    return s
