#!/bin/zsh
#conda activate llms && python ../tldr_npu/cosine_similarity.py
PKG_NAME=CosineSimilarityBatched.mlpackage
python ../tldr_npu/cosine_similarity.py
rm -rf npu-acclerator/$PKG_NAME
cp -rv $PKG_NAME npu-acclerator/
rm -rf ./$PKG_NAME
# cp -rv ../tldr_npu/CosineSimilarityBatched.mlpackage npu-acclerator
# xcrun coremlc compile npu-acclerator/CosineSimilarityBatched.mlpackage npu-acclerator