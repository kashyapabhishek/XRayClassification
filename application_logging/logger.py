from datetime import datetime


class Logger(object):

    def __init__(self):
        self.date = None
        self.now = None
        self.time = None

    def log(self, file_object, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.time = self.now.strftime("%H:%M:%S")
        file_object.write(f'{str(self.date)} {self.time}   {log_message}')
