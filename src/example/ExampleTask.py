import time

from src.example import ExampleWorker
from src.gui.GuiWorker import GuiWorker
from src.process.ProcessWorker import ProcessWorker

"""
An example Worker Class to use in the Library Dynamic Process Worker
"""


def retrieve_results(result):
    """
    For every process that finishes, this method will get called
    :param result: Result dict, specified in the do_work() worker method
    """

    print(result)


def start_process_worker():
    """
    Start the Process Worker without a GUI and call retrieve_results() when
    the processes have all finished
    """

    ProcessWorker(ExampleWorker.do_work, gui=False, thread_count=10, limit_logical=False,
                  use_threads=False, callback=ExampleWorker.result)


def start_process_worker_gui():
    """
    Start the Process Worker with a GUI and call retrieve_results() when
    the processes have all finished
    """

    gui_worker = GuiWorker()
    process_worker = ProcessWorker(ExampleWorker.do_work, gui_worker=gui_worker, use_threads=False, limit_logical=False,
                                   thread_count=10, iteration_count=1, callback=ExampleWorker.result)

    time.sleep(3)

    process_worker.add_worker(ExampleWorker.do_work, 1, None)


if __name__ == '__main__':
    # start_process_worker()
    start_process_worker_gui()
