#!/bin/sh
mkdir -p /Users/manu/proj_tldr/tldr-dekstop/release-products/libs/system
ln -s /opt/homebrew/opt/libpq/lib/ /Users/manu/proj_tldr/tldr-dekstop/release-products/libs/system/libpq
ln -s /opt/homebrew/opt/libpqxx/lib/ /Users/manu/proj_tldr/tldr-dekstop/release-products/libs/system/libpqxx
ln -s /opt/homebrew/opt/poppler/lib/ /Users/manu/proj_tldr/tldr-dekstop/release-products/libs/system/poppler
ln -s /opt/homebrew/opt/openssl/lib /Users/manu/proj_tldr/tldr-dekstop/release-products/libs/system/openssl
ln -s /opt/homebrew/opt/sqlite/lib /Users/manu/proj_tldr/tldr-dekstop/release-products/libs/system/sqlite
ln -s /opt/homebrew/opt/icu4c/lib /Users/manu/proj_tldr/tldr-dekstop/release-products/libs/system/icu4c
ln -s /opt/homebrew/opt/zlib/lib /Users/manu/proj_tldr/tldr-dekstop/release-products/libs/system/zlib
ln -s /opt/homebrew/opt/curl/lib /Users/manu/proj_tldr/tldr-dekstop/release-products/libs/system/curl


cp -v libtldr.a /Users/manu/proj_tldr/tldr-dekstop/release-products/libs/
cp -v ../src/lib_tldr/tldr_api.h /Users/manu/proj_tldr/tldr-dekstop/release-products/include/
cp -v ../src/lib_tldr/definitions.h /Users/manu/proj_tldr/tldr-dekstop/release-products/include/