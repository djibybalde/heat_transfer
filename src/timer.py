"""
heat_transfer/src/timer.py

"""


def timer(start_time, end_time):

    """
    Build a timer function for taking in account the time interval between two instances.

    Args:
        start_time (float): Time in seconds of the system at the beginning of the instance.
        end_time (float): Time in seconds of the system at the ending of the instance.

    Returns:
        Wall time between the beginning and the end of the instance.
    """

    real_time = end_time - start_time

    # if seconds
    if real_time < 60:
        _time = f'{round(real_time, 2)}sec'

    # if minute(s)
    elif real_time >= 3600:
        _time = (f'{int(real_time / 3600)}h'
                 f' {int((real_time % 3600) / 60)}min {round(real_time % 60, 2)}sec')

    # if hour(s) or day(s)
    else:
        _time = f'{int(real_time / 60)}min {round(real_time % 60, 2)}sec'

    return _time


"""
# Example
import time
from timer import timer

start = time.time()
time.sleep(2)

print(timer(start, time.time()))
"""
