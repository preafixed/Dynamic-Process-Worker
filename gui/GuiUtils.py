from datetime import datetime


def get_current_date():
    """
    :return: Returns the current date in String format
    """

    return datetime.today().strftime('%Y-%m-%d %H:%M:%S')


def get_current_datetime():
    """
    :return: Returns the current date in String format
    """

    return datetime.today().strptime(get_current_date(), '%Y-%m-%d %H:%M:%S')


def convert_string_to_date(string):
    """
    Converts a string in a datetime object
    :param string: the date as a String
    :return: datetime (Object)
    """
    return datetime.strptime(string, '%Y-%m-%d %H:%M:%S')


def calculate_time(date_time):
    """
    Calculate the time passed
    :param date_time: The datetime object as a String
    """

    return get_current_datetime() - convert_string_to_date(date_time)


def print_message(message):
    """
    Simple print method
    :param message: The message to send
    """

    print("[PM]: ", message)
