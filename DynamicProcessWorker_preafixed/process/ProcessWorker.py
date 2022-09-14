import time
from threading import Thread
from multiprocessing import Process, Event, Queue

from DynamicProcessWorker_preafixed.gui import GuiWorker
from DynamicProcessWorker_preafixed.gui.GuiUtils import print_message
from DynamicProcessWorker_preafixed.process.ProcessManager import ProcessManager
from DynamicProcessWorker_preafixed.process.ProcessBalancer import ProcessBalancer


class ProcessWorker:
    """
    This is the main Class for creating a Process Worker
    """

    def __init__(self, worker, *args, gui_worker=None, thread_count=1,
                 iteration_count=1, max_progress=100, gui=True, callback=None,
                 use_threads=True, max_cpu=80, limit_logical=True):
        """
        Initialize the Process Worker with the given arguments
        :param worker: The worker method, which gets spawned
        :param gui_worker: An Instance of the Object GUI Worker
        :param thread_count: The number of parallel processes
        :param iteration_count: The number of iteration per process
        :param max_progress: The maximum progress (e.x 100)
        :param callback: Callback for receiving the results
        :param gui: Enable or disable the GUI Interface
        :param args: Other args required for the worker to run
        :param use_threads: Use threads instead of processes
        :param limit_logical: True, if process cap should be count
                                of logical process units
        :param max_cpu: The maximal amount of CPU Usage
        """

        self.args = args
        self.gui = gui
        self.processes = []
        self.event = Event()
        self.worker = worker
        self.max_cpu = max_cpu
        self.callback = callback
        self.result_queue = Queue()
        self.process_balancer = None
        self.use_threads = use_threads
        self.thread_count = thread_count
        self.max_progress = max_progress
        self.limit_logical = limit_logical
        self.result_listener_thread = None
        self.iteration_count = iteration_count
        self.gui_worker: GuiWorker = gui_worker
        self.process_manager = ProcessManager(gui, None)

        self.__init_worker()

    def __init_worker(self):
        """
        Start the Process Worker and start all processes
        """

        self.__start_gui_worker()
        self.process_manager.result_queue = self.result_queue
        self.process_manager.result_callback = self.callback
        self.process_balancer = ProcessBalancer(self, self.event, self.max_cpu, self.limit_logical)
        self.process_balancer.active_processes = self.active_processes

        for process_id in range(self.thread_count):
            self.add_worker(self.worker, self.iteration_count, self.args)

    def add_worker(self, worker, iteration_count, *args):
        """
        Add a Worker to the Process Pool
        :param worker: The worker method
        :param iteration_count: The max iteration
        :param args: The required args
        """

        if self.use_threads:
            process = self.create_thread(worker, iteration_count, args)
        else:
            process = self.create_process(worker, iteration_count, args)

        self.processes.append(process)
        self.process_balancer.add_process(iteration_count)

    def create_process(self, worker, iteration_count, *args):
        """
        Create a thread object with the given args
        :param worker: The given worker method
        :param iteration_count: The max iteration count
        :param args: The optional arguments
        :return: threading.Thread
        """

        return Process(target=worker,
                       args=(self.process_manager.get_size(),
                             iteration_count, self.event,
                             self.result_queue, self.process_manager, args))

    def create_thread(self, worker, iteration_count, args):
        """
        Create a thread object with the given args
        :param worker: The given worker method
        :param iteration_count: The max iteration count
        :param args: The optional arguments
        :return: threading.Thread
        """

        return Thread(target=worker,
                      args=(self.process_manager.get_size(),
                            iteration_count, self.event,
                            self.result_queue, self.process_manager, args))

    def start_process(self, index):
        """
        Start a process, that is already created
        :param index: The index of the process
        """

        process: Process = self.processes[index]
        process_model = self.process_manager.get_process(index)
        process.start()

        process_pid = process_model.index if self.use_threads else process.pid

        self.process_manager.assign_pid(process_model.index, process_pid)

        self.start_listening()

    def active_processes(self):
        """
        Returns the number of active processes
        """

        count = 0

        for process in self.processes:
            count = count + 1 if process.is_alive() else count

        return count

    def keep_listening(self):
        """
        Returns true if there are still tasks being processed
        :return: True or False (Boolean)
        :return:
        """

        return self.active_processes() > 0 or self.process_balancer.check_listener() or self.result_queue.qsize() > 0

    def start_listening(self):
        """
        Start listening for items in the result queue
        :return:
        """

        if not self.result_listener_thread or not self.result_listener_thread.is_alive():
            print_message("Listening for incoming results.. ")

            self.result_listener_thread = Thread(target=self.receive_queue)
            self.result_listener_thread.start()

    def receive_queue(self):
        """
        Listen for incoming results and send them back
        to the origin (every second update)
        """

        while self.keep_listening():
            while self.result_queue.qsize() > 0:
                result = self.result_queue.get()
                self.callback(result)

            time.sleep(0.5)

        self.event.set()

        if self.gui:
            self.gui_worker.terminate_gui_worker()

    def __start_gui_worker(self):
        """
        Start the Gui Worker, if enabled
        """

        if self.gui:
            self.process_manager.progress_queue = self.gui_worker.incoming_data
            self.gui_worker.start_worker(self.process_manager,
                                         self.thread_count, self.iteration_count,
                                         self.max_progress, self.event)
