#!/bin/sh
#g++ -pthread -O4 -g svdDynamic.c RayTracer.c utils.c -lm -lpthread -o RayTracer

# For multi-threading
g++ -O4 -g svdDynamic.c RayTracer.c utils.c -lm -lpthread -o RayTracer
