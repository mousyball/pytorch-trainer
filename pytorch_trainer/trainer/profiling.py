import time

seconds_in_day = 60 * 60 * 24
seconds_in_hour = 60 * 60
seconds_in_minute = 60


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def profiling(func):
    def wrapper(*args, **kwargs):
        tic = time.time()
        ret = func(*args, **kwargs)
        toc = time.time()
        seconds = toc - tic

        obj = args[0]  # Get self
        # Time in seconds
        obj.logger.info(
            f"{bcolors.FAIL}[Total Elapsed Time]{bcolors.ENDC}: {bcolors.OKGREEN}{seconds:.5f} s.{bcolors.ENDC}")

        # Time in {days:hours:minutes}
        days = seconds // seconds_in_day
        hours = (seconds - (days * seconds_in_day)) // seconds_in_hour
        minutes = (seconds - (days * seconds_in_day) -
                   (hours * seconds_in_hour)) // seconds_in_minute
        obj.logger.info(
            f"{bcolors.FAIL}[Total Elapsed Time]{bcolors.ENDC}: "
            + f"{bcolors.OKGREEN}{int(days)} days / {int(hours)} hrs / {int(minutes)} mins.{bcolors.ENDC}")

        return ret
    return wrapper
