import time
import datetime
from contextlib import contextmanager


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


@contextmanager
def calc_time():
    """
    get elapsed time in `with` statement

    Usage:
    with calc_time() as done:
        # do something
        elapsed = done()
    """
    try:
        t0 = time.time()

        def gettime():
            return format_time(time.time() - t0)

        yield gettime
    finally:
        pass
