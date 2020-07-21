#Read more about this project at https://medium.com/gsi-technology/how-to-benchmark-ann-algorithms-a9f1cef6be08
#Originally built for Deep1b dataset

import numpy
import os
import sys
import time

def trim_end(location, vector):return vector[:len(vector)-location]


def vector_order_valid(fv, known_vector_length):
    #Function to check if fvecs files such as deep1b (http://sites.skoltech.ru/compvision/noimi/) are valid
    #Takes in the fvecs file as numpy array
    #Takes in expected length of each vector, (first element of the file)
    count = 0
    print("Begining checks ...")

    for i in range(0,len(fv),known_vector_length):
        
        if fv[i] != known_vector_length:
            count = count + 1
            print("no "+str(known_vector_length)+" at position:  "+ str(i) +" instead it is " + str(fv[i]))
            
    print("vector order check done, there were " + str(count) + " mismatches")
    print()
    print()
    
    if len(fv)/97 != 0:
        print("Vector has been clipped checking for clip location")
        
        for i in range(1,100):
            if fv[-i] == known_vector_length:
                print("final vector is only: " + str(i) + " dimentions long")
                fv_new = trim_end(i,fv)
                break
                
    print("Done!")
    return fv_new


def convert_base_file(fvecs_file_name):
    #Converts the fvecs file to to a numpy array
    #Takes in file location
    
    start = time.time()
    print("Starting......")
    
    fv = numpy.fromfile("fvecs_file_name",dtype="int32")
    dim = fv.view(numpy.int32)[0] #Get vector length
    
    fv_new = vector_order_valid(fv,dim) #Validate fvecs file
    
    
    new = fv_new.reshape(-1, dim + 1)[:,1:] # reshapes file
    f_new = new.view(numpy.float32)
    end = time.time()
    print("Function done at " +str(end-start) + " seconds")
    return f_new
