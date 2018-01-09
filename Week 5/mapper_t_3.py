#!/usr/bin/env python

import sys
import io 
import time

# input comes from STDIN (standard input)

start_time = time.time()

counter = 0 
for line in sys.stdin: 
    # remove leading and trailing whitespace
    try:
        line = line.strip()
    # split the line into words
        words = line.split(';')
    
        if ( words[5] != "" and float(words[2]) ):
                print ('%s;%s' % (words[5],[float(words[2]),start_time]))
    except ValueError:
            pass
    