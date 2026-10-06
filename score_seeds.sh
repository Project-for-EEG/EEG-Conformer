set -e
for S in 43 44; do
  echo "=== event-scoring seed $S ==="
  python -u helsinki_curve_events.py --seed "$S"
done
echo "=== event scoring done ==="
