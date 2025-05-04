#!/bin/sh

CUR_DIR=$(PWD)
# build llama.cpp
#cd ../../ref_projects/llama.cpp && scripts/build-all-release.sh
cd $CUR_DIR
cd npu && ./package_release.sh
cd $CUR_DIR
