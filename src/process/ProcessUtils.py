from multiprocessing import Queue


def join_all(processes):
    """
    Wait for completion of all processes in an array
    :param processes: The array of processes
    """

    for process_obj in processes:
        process_obj.join()


def get_all(processes, queue: Queue):
    """
    Get all results from the processes
    :param queue: The Queue of the multiprocess
    :param processes: The array of results
    """

    results = []
    for _ in processes:
        results.append(queue.get())

    return results
