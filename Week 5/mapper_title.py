    #!/usr/bin/env python
    
import sys
import io 
import time


start_time = time.time()
    # input comes from STDIN (standard input)
    

    
for line in sys.stdin: 
        # remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split(';')
    
    try:
        if (words[4] != "" and words[2] != "" ):
            if (float(words[2])):
                print ('%s;%s' % (words[4],[float(words[2]),start_time]))
    except ValueError:
        pass
        #counter += 1