import time
from multiprocessing import Queue
from multiprocessing.synchronize import Event

from src.gui.GuiUtils import calculate_time
from src.process.ProcessManager import ProcessManager


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

        quit(0)

    for i in range(loop_count):
        for x in range(100):
            process_manager.update(thread_id, progress=x + 1, iteration=i + 1)
            """
            Update the current processes, with new information
            """

            time.sleep(0.05)
            if event.is_set():
                """
                Received event for closing process
                So call close_process() method
                """

                close_process()
                break

    close_process()


def result(result):
    print("result")

