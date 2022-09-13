import json
from multiprocessing import Queue

from src.gui.GuiUtils import print_message
from src.process.ProcessModel import ProcessModel


class ProcessManager:
    """
    A Class for managing and interacting with Process Models
    """

    def __init__(self, gui, progress_queue):
        """
        Initialize the Process Manager
        :param progress_queue: The Que for updating progress
                                ! Only if gui=True !
        """

        self.progress_queue: Queue = progress_queue
        self.gui = gui
        self.process_queue = []

    def get_size(self):
        return len(self.process_queue)

    def add(self, process_model: ProcessModel):
        """
        Add a process using a Process Model
        :param process_model: The model of the process
        """

        self.process_queue.append(Queue())
        self.process_queue[process_model.pid].put(json.dumps(process_model, default=vars))

        print_message("Added new Process Queue(pid=" + str(process_model.pid) + ")")

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

        return ProcessModel().from_json(queue_content)

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
