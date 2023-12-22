from typing import Optional, Callable, List
import os
import cv2

class PathLoader:
    def __init__(self, 
        path: str, 
        start : Optional[int] = None,
        stop : Optional[int] = None,
        stride : Optional[int] = None,
        filter: Optional[Callable[[str], bool]] = None):

        self.filenames = []

        # open directory and read all filenames
        with os.scandir(path) as it:
            for entry in it:
                if entry.is_file() and (filter is None or filter(entry.name)):
                    self.filenames.append(entry.name)

        # sort filenames
        self.filenames.sort()

        self.start = start if start is not None else 0
        self.stop = stop if stop is not None else len(self.filenames)
        self.stride = stride if stride is not None else 1

        self.path = path    

    def __iter__(self):
        self.state = self.start
        return self
    
    def __next__(self):
        self.state += self.stride
        if self.state >= self.stop:
            raise StopIteration
    
        return self.path + self.filenames[self.state]