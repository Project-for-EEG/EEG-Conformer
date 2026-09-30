"""Container entry point.

SzCORE invokes the image with no arguments, passing the recording and the
destination through the environment:

    docker run -v ./data:/data -v /tmp:/output \
        -e INPUT=rec.edf -e OUTPUT=rec.tsv <image>

The template CMD passes /data/$INPUT and /output/$OUTPUT positionally, so both
routes reach the same place. szcore_run resolves whichever it is given.
"""
import sys

from szcore_run import main

if __name__ == "__main__":
    sys.exit(main())
