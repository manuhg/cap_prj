#!/bin/sh
pbpaste| grep -Eo '@[^$]+' |sort|uniq -c|sort | grep -v '1 @'
