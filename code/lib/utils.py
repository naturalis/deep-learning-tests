import time

class Timer:

    start_time = None
    end_time = None
    formats = {
        "pretty" : "%02dd %02dh %02dm %02ds"
    }
    milestones = []

    def __init__(self):
        self.set_start_time()

    def set_start_time(self):
        self._set_time(self.start_time)

    def set_end_time(self):
        self._set_time(self.end_time)

    def _set_time(self,var):
        var = self.get_timestamp()

    def get_timestamp(self):
        return time.time()

    def add_milestone(self,label):
        self.milestones.append({ "label" : label, "timestamp" : self.get_timestamp() })

    def get_milestones(self):
        return self.milestones

    def reset_milestones(self):
        self.milestones = []

    def get_time_passed(self,format="pretty"):
        if None is self.end_time:
            self.set_end_time()
        time = float(self.end_time - self.start_time)
        day = time // (24 * 3600)
        time = time % (24 * 3600)
        hour = time // 3600
        time %= 3600
        minutes = time // 60
        time %= 60
        seconds = time
        return self.formats[format] % (day, hour, minutes, seconds)
