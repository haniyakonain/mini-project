#!/bin/bash
# Downloads real, ordinary photos (color + grayscale) to use as "Not a Scan"
# negative training examples, via the free picsum.photos service. Run this
# from anywhere; it writes into photos/ and gray/ next to this script.
set -e
cd "$(dirname "$0")"
mkdir -p photos gray

echo "Fetching 150 color photos..."
for i in $(seq 0 149); do
  curl -sL -o "photos/photo_${i}.jpg" --max-time 15 "https://picsum.photos/id/${i}/300/300" &
  if (( i % 15 == 14 )); then wait; fi
done
wait

echo "Fetching 80 grayscale photos..."
for i in $(seq 200 279); do
  curl -sL -o "gray/gray_${i}.jpg" --max-time 15 "https://picsum.photos/id/${i}/300/300?grayscale" &
  if (( i % 15 == 14 )); then wait; fi
done
wait

# Drop any failed/empty downloads
python3 -c "
from PIL import Image
import os
for d in ('photos', 'gray'):
    for f in os.listdir(d):
        p = os.path.join(d, f)
        try:
            Image.open(p).verify()
        except Exception:
            os.remove(p)
            print('removed bad download:', p)
"
echo "Done. photos/: $(ls photos | wc -l), gray/: $(ls gray | wc -l)"
