import sys
from multiprocessing import Process, Event, Queue

from gui.GuiWorker import GuiWorker
from process.ProcessManager import ProcessManager
from process.ProcessModel import ProcessModel
from process.ProcessUtils import join_all, get_all


class ProcessWorker:
    """
    This is the main Class for creating a Process Worker
    """

    def __init__(self, worker, gui_worker=None, thread_count=1, loop_count=1,
                 max_progress=10, gui=True, callback=None, *args):
        """
        Initialize the Process Worker with the given arguments
        :param worker: The worker method, which gets spawned
        :param gui_worker: An Instance of the Object GUI Worker
        :param thread_count: The number of parallel processes
        :param loop_count: The number of iteration per process
        :param max_progress: The maximum progress (e.x 100)
        :param gui: Enable or disable the GUI Interface
        :param args: Other args required for the worker to run
        """

        self.args = args
        self.worker = worker
        self.gui = gui
        self.event = Event()
        self.processes = []
        self.queue = Queue()
        self.callback = callback
        self.thread_count = thread_count
        self.loop_count = loop_count
        self.max_progress = max_progress
        self.gui_worker: GuiWorker = gui_worker
        self.process_manager = ProcessManager(gui, None)

        self.__init_worker()

    def __init_worker(self):
        """
        Start the Process Worker and start all processes
        """

        self.__start_gui_worker()

        for process_id in range(self.thread_count):
            process = Process(target=self.worker, args=(process_id, self.loop_count,
                                                        self.event, self.queue, self.process_manager, self.args))
            """
            :param target: The Worker to run
            :param process_id: The Process (PID)
            :param loop_count: Amount of Loops
            :param event: Multiprocessing.Event for 
                            handling stop events
            :param queue: Collecting results of the
                            running processes
            :param process_manager: Update the current
                            process information
            :param args: Other arguments for worker
            """

            self.processes.append(process)
            self.process_manager.add(ProcessModel(pid=process_id, max_iterations=self.loop_count))

            process.start()

        join_all(self.processes)
        results = get_all(self.processes, self.queue)

        self.event.set()

        if self.callback:
            self.callback(results)

    def __start_gui_worker(self):
        """
        Start the Gui Worker, if enabled
        """

        if self.gui:
            self.process_manager.progress_queue = self.gui_worker.incoming_data
            self.gui_worker.start_worker(self.process_manager,
                                         self.thread_count, self.loop_count,
                                         self.max_progress, self.event)
