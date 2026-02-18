import logging
import time


class Timer:
    def __init__(self, operation=''):
        self.operation = operation
        self.start_time = None
        self.end_time = None

    def start(self):
        self.start_time = time.time()
        return self

    def end(self):
        self.end_time = time.time()
        self.__log_end()

    def __log_end(self):
        delta = self.end_time - self.start_time
        if delta < 60:
            elapsed_time = f'{round(delta, 2)} sec.'
        else:
            elapsed_time = f'{round(delta / 60.0, 2)} min.'
        logging.info(f'{self.operation}: elapsed time {elapsed_time}')
