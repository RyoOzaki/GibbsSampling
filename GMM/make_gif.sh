#!/bin/sh

DELAY=10
TOP_DELAY=100
REAR_DELAY=100

convert -layers optimize -loop 0 -delay ${TOP_DELAY} tmp/truth.png -delay ${DELAY} tmp/frame*.png -delay ${REAR_DELAY} tmp/final.png result.gif
