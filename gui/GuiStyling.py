from gui.GuiUtils import convert_string_to_date, calculate_time
from process.ProcessModel import default_elements


def default():
    """
    Default GUI Styling
    :return: Styling elements (Object)
    """

    prefix = "Process [ID:%PID] (%PROGRESS%)"
    suffix = "Iteration [%ITER/%MAX_ITER], Time [%TIME], Ticks [%TICKS]"

    return GuiStyling(prefix, suffix, [])


class GuiStyling:
    """
    A Class for personalizing the GUI
    """

    def __init__(self, prefix: str, suffix: str, elements: [], max_progress=100):
        """
        Initialize the Gui Styling
        :param prefix: The prefix as a String
        :param suffix: The suffix as a String
        :param suffix: The progress as a String
        :param elements: The tuple of elements
        :param max_progress: The maximum progress
        """

        self.prefix = prefix
        self.suffix = suffix
        self.max_progress = max_progress
        self.elements = default_elements() + elements

    def replace_string(self, text: str, process_model: dict):
        """
        Check for every tuple (%identifier, attr_name) and replace all
        :param text: The text to replace
        :param process_model The Process Model that holds the Data
        :return: Prefix (String)
        """

        for element in self.elements:
            identifier: str = element[0]
            attr_name: str = element[1]

            text = text.replace(identifier, str(self.special_attr(attr_name, process_model)))

        return text

    def special_attr(self, attr_name: str, process_model: dict):
        """
        Calculate the progress and time

        :param attr_name:
        :param process_model:
        :return:
        """

        if attr_name == "progress":
            return round(float(process_model[attr_name]) / float(self.max_progress) * 100)
        elif attr_name == "time_started":
            return calculate_time(process_model[attr_name])
        else:
            return process_model[attr_name]

    def get_prefix(self, process_model: dict):
        """
        Get the replaced prefix
        :param process_model: The Process Model that holds the Data
        :return: Prefix (String)
        """

        return self.replace_string(self.prefix, process_model)

    def get_suffix(self, process_model: dict):
        """
        Get the replaced prefix
        :param process_model: The Process Model that holds the Data
        :return: Suffix (String)
        """

        return self.replace_string(self.suffix, process_model)
