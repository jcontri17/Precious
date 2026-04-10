#!/bin/bash

# Toggle runtime output: set to “yes” to show, “no” to suppress
SHOW_OUTPUT=no

# usage check
if [ -z "$1" ]; then
  echo "Usage: $0 <python-script.py>"
  exit 1
fi

FILENAME="$1"

# check existence
if [ ! -f "$FILENAME" ]; then
  echo "❌ File '$FILENAME' not found!"
  exit 1
fi

# strip .py, build output name
BASENAME=$(basename "$FILENAME" .py)
OUTFILE="${BASENAME}.PDF"
HEADER="$(pwd)/$FILENAME"

# bundle source + runtime output (conditionally) into enscript → ps2pdf
(
  echo "### Source: $FILENAME"
  cat "$FILENAME"
  echo

  if [ "$SHOW_OUTPUT" = yes ]; then
    echo "### Output:"
    python3 "$FILENAME"
  else
    echo "### Output: (suppressed)"
  fi
) | enscript -Epython \
    --header="$HEADER||" \
    --font=Courier8 \
    -o - \
| ps2pdf - "$OUTFILE"

echo "✅ PDF created: $OUTFILE"

