#!/bin/sh
cmake -B  build  && cmake --build build  --config Release -j 8 #&& ./build/tldr_cpp
