#!/bin/sh
 cp -rv ../tldr_npu/CosineSimilarityBatched.mlpackage npu-acclerator
 xcrun coremlc compile npu-acclerator/CosineSimilarityBatched.mlpackage npu-acclerator