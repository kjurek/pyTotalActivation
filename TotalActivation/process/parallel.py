from temporal import temporal_TA


def parallel_temporalTA(input, output, voxels, l, f_Analyze, maxeig, n_tp, t_iter, cost_save):
    """
    This function allows to run a process and dump the results to memory-shared object

    :param input:
    :param output:
    :param voxels:
    :return:
    """

    output[:, voxels] = temporal_TA(input, f_Analyze, maxeig, n_tp, t_iter,
                                    noise_estimate_fin=None, cost_save=cost_save, l=l,
                                    voxels=voxels)
