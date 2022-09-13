import time
from multiprocessing import Queue
from multiprocessing.synchronize import Event

from src.gui.GuiUtils import calculate_time
from src.gui.GuiWorker import GuiWorker
from src.process.ProcessManager import ProcessManager
from src.process.ProcessWorker import ProcessWorker

"""
An example Worker Method to use in the Library Dynamic Process Worker
"""


def do_work(thread_id, loop_count, event: Event, queue: Queue, process_manager: ProcessManager, args):
    """
    An example Worker that demonstrates how to use it correctly
    :param thread_id: The thread ID of the current worker
    :param loop_count: The iteration count of this process
    :param event: The event handler to detect stop signals
    :param queue: The result queue to store outcomes of this worker
    :param process_manager: The process manager to access the update method
    :param args: Other args that we require to use (given in the main)
    """

    def close_process():
        """
        If needed, push data to the result queue
        """

        process = process_manager.get_process(thread_id)
        calculated_time = calculate_time(process.time_started)

        queue.put(
            {
                "process_id": thread_id,
                "process_result": calculated_time
            }
        )

    for i in range(loop_count):
        for x in range(100):
            process_manager.update(thread_id, progress=x + 1, iteration=i)
            """
            Update the current processes, with new information
            """

            time.sleep(0.1)
            if event.is_set():
                close_process()
                """
                Received event for closing process
                So call close_process() method
                """

                break

    close_process()


def retrieve_results(result):
    """
    Result has the individual results of the processes
    saved in an array
    :param result: Result Array
    """

    print(result)


def start_process_worker():
    """
    Start the Process Worker without a GUI and call retrieve_results() when
    the processes have all finished
    """

    ProcessWorker(do_work, gui=False, thread_count=1, callback=retrieve_results)


def start_process_worker_gui():
    """
    Start the Process Worker with a GUI and call retrieve_results() when
    the processes have all finished
    """

    gui_worker = GuiWorker()
    process_worker = ProcessWorker(do_work, gui_worker=gui_worker, thread_count=3, callback=retrieve_results)

    time.sleep(1)

    process_worker.add_worker(do_work, 1, None)


if __name__ == '__main__':
    start_process_worker_gui()
