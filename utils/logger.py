import os

class Logger(object):

    def __init__(self, log_path, header=None, mode='a'):
        self.log_path = log_path
        self.log_file = open(log_path, mode)
        if not os.path.exists(self.log_path) and header is not None:
            self.log_file.write(header)
    
    def __del(self):
        self.log_file.close()

    def write(self, message):
        self.log_file.write(message)
        self.log_file.flush()

    def print_and_write(self, message):
        print(message)
        self.write(message)