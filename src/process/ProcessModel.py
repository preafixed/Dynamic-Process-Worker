import json

from src.gui.GuiUtils import convert_string_to_date, get_current_date, get_current_datetime


def default_elements():
    """
    Get the elements for the GUI Styling
    :return: Tuple of elements
    """

    return [("%PID", "pid"), ("%ITER", "iteration"),
            ("%MAX_ITER", "max_iteration"), ("%TICKS", "ticks"),
            ("%PROGRESS", "progress"), ("%TIME", "time_started")]


class ProcessModel:
    """
    A Process Model used to represent a unique Process
    """

    def __init__(self, pid=0, is_active=False, iteration=1, max_iterations=1,
                 progress=0, time_started=get_current_date(), last_update=0, ticks=0, *args):
        """
        Creates a Process Model Instance
        :param pid: The ID of the Process
        :param is_active: The state of the Process
        :param iteration: The current loop iteration
        :param max_iterations: The maximal iterations
        :param progress: Progress of current interation
        :param time_started: Time of process start
        :param last_update: Last update time to calculate ticks
        :param ticks: The ticks the process takes for one iteration
        :param args: If needed, add other arguments
        """

        self.pid = pid
        self.is_active = is_active
        self.iteration = iteration
        self.max_iteration = max_iterations
        self.progress = progress
        self.time_started = time_started
        self.last_update = last_update
        self.ticks = ticks
        self.args = args

    def update(self, iteration=None, progress=None, *args):
        """
        Update only needed parameters
        :param iteration: The new iteration
        :param progress: The new progress
        :param args: The changed args
        :return: Process Model (Object)
        """

        self.iteration = iteration if iteration is not None else self.iteration
        self.progress = progress if progress is not None else self.progress
        self.args = args if args is not None else self.args

        if self.last_update != 0:
            self.ticks = str((get_current_datetime() - convert_string_to_date(self.last_update)).total_seconds())

        self.last_update = get_current_date()

        return self

    def to_json(self):
        """
        Serialize the Process Model into JSON
        :return: JSON String
        """

        return json.dumps(self, default=vars)

    def from_json(self, json_dict):
        """
        Deserializes the JSON String (works even if we get a dict)
        :param json_dict: Serialized Process Model
        :return: Process Model (Object)
        """

        if type(json_dict) == ProcessModel:
            return json_dict

        if type(json_dict) == str:
            json_dict = json.loads(json_dict)

        self.__init__(json_dict['pid'], json_dict['is_active'], json_dict['iteration'],
                      json_dict['max_iteration'], json_dict['progress'], json_dict['time_started'],
                      json_dict['last_update'], json_dict['ticks'], json_dict['args'])

        return self
