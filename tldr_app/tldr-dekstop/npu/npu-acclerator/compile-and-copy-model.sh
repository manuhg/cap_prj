#!/bin/sh
 cp -rv ../tldr_npu/CosineSimilarity.mlpackage npu-acclerator
 xcrun coremlc compile npu-acclerator/CosineSimilarity.mlpackage npu-acclerator