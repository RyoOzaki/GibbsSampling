#!/bin/sh

DELAY=10
TOP_DELAY=100
REAR_DELAY=100

convert -layers optimize -loop 0 -delay ${TOP_DELAY} fig/truth.png -delay ${DELAY} fig/frame*.png -delay ${REAR_DELAY} fig/final.png result.gif
