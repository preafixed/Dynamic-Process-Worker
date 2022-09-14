import time
from multiprocessing import Queue
from threading import Thread

import psutil

from src.gui.GuiUtils import print_message
from src.process.ProcessManager import ProcessManager
from src.process.ProcessModel import ProcessModel


class ProcessBalancer:
    """
    This class is for managing and balancing out processes to cap the
    cpu usage limit and to organize processes coming in and out
    """

    def __init__(self, process_worker, event, max_cpu, limit_logical):
        """
        Initialize the process balancer
        :param process_worker: The main process worker
        :param event: The kill event of the worker
        :param max_cpu: The maximum CPU usage, to cap at
        :param limit_logical: True, if the cap should be logical
                                process units (hyper threaded)
        : core_count: Maximum number of processes possible
        """

        self.event = event
        self.max_cpu = max_cpu
        self.process_queue = Queue()
        self.active_processes = None
        self.limit_logical = limit_logical
        self.queue_listen_thread = None
        self.process_worker = process_worker
        self.process_manager: ProcessManager = process_worker.process_manager

    def add_process(self, loop_count):
        """
        Add a process to the Queue and start the listening process
        :param loop_count: Loop count of the process
        """

        time.sleep(0.1)

        if not self.event.is_set():
            process_model = ProcessModel(max_iterations=loop_count)
            process_model = self.process_manager.add(process_model)

            self.process_queue.put(process_model.index)
            self.start_listen()
        else:
            quit(0)

    def start_listen(self):
        """
        Start a new Queue Listener only if none is running
        """

        if not self.queue_listen_thread or not self.queue_listen_thread.isAlive():
            if not self.event.is_set():
                self.queue_listen_thread = Thread(target=self.receive_queue).start()
            else:
                quit(0)

    def allowed_task_count(self):
        """
        Get the allowed number of parallel processes
        :return: Core Count (Number)
        """

        if self.limit_logical:
            return psutil.cpu_count(logical=True)
        else:
            return psutil.cpu_count(logical=False)

    def can_start_new_tasks(self):
        cpu_usage = psutil.cpu_percent()

        return cpu_usage < self.max_cpu

    def receive_queue(self):
        """
        Checks if the queue has new tasks to be fulfilled, if yes
        try to start the process and stop listening
        """

        while self.process_queue.qsize() > 0 and not self.event.is_set():
            if self.active_processes() < self.allowed_task_count() and self.can_start_new_tasks():
                index = self.process_queue.get()
                print_message("Received process Task.. [ID:" + str(index) + "]")

                try:
                    self.start_process(index)
                except Exception as e:
                    print_message("Error starting process [ID:" + str(index) + "],"
                                  " trying to restart... [Error:" + str(e) + "]")

                    self.process_queue.put(index)

                    time.sleep(0.5)

                break
            else:
                time.sleep(0.5)

        if self.event.is_set():
            quit(0)

    def check_listener(self):
        return self.process_queue.qsize() > 0

    def start_process(self, index):
        """
        Starts a process with a unique index
        :param index: The index of the process
        """

        self.process_worker.start_process(index)
