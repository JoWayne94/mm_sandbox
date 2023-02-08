"""
Description: Useful functions

Author(s): Jo Wayne Tan
"""
import os
import psutil


def printMemoryUsageInMB():
    print("\nMemory usage is: " + str(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2) + ' MB\n')
