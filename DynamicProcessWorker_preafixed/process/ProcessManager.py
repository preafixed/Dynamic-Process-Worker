import json
import ProcessModel
from multiprocessing import Queue


class ProcessManager:
    """
    A Class for managing and interacting with Process Models
    """

    def __init__(self, gui, progress_queue):
        """
        Initialize the Process Manager
        :param progress_queue: The Queue for updating progress
                                ! Only if gui=True !
        result_queue: The Queue for receiving results
        """

        self.gui = gui
        self.process_queue = []
        self.result_queue = None
        self.active_process = None
        self.result_callback = None
        self.progress_queue: Queue = progress_queue

    def get_size(self):
        return len(self.process_queue)

    def assign_pid(self, process_id, process_pid):
        """
        Assign a pid to a started process
        :param process_id: The index of the process model
        :param process_pid: The pid of the process
        """

        queue_content = self.get_process(process_id, keep=False)
        queue_content = queue_content.update_pid(process_pid)

        self.process_queue[process_id].put(json.dumps(queue_content.to_json()))

    def add(self, process_model: ProcessModel):
        """
        Add a process using a Process Model
        :param process_model: The model of the process
        """

        self.process_queue.append(Queue())
        index = self.get_size() - 1
        process_model.index = index
        self.process_queue[index].put(json.dumps(process_model, default=vars))

        print_message("Added new Process Queue(id=" + str(process_model.index) + ")")

        return process_model

    def get_process(self, process_id, keep=True):
        """
        Get a process by an index
        :param keep: Remove the process from Queue or not
        :param process_id: The index of the process
        :return: Process Model (Object)
        """

        queue_content = json.loads(self.process_queue[process_id].get())

        if keep:
            self.process_queue[process_id].put(json.dumps(queue_content, default=vars))

        return ProcessModel.ProcessModel().from_json(queue_content)

    def update(self, process_id, iteration=None, progress=None, args=None):
        """
        Update only needed parameters
        :param process_id: The ID of the Process
        :param iteration: The new iteration
        :param progress: The new progress
        :param args: The changed args
        """

        queue_content = self.get_process(process_id, keep=False)
        queue_content = queue_content.update(iteration, progress, args)

        self.process_queue[process_id].put(json.dumps(queue_content.to_json()))

        if self.gui:
            self.progress_queue.put(queue_content)
