#!/usr/bin/env bash
# Start a local mongod for DocuTrust dev. Idempotent: if mongod is already
# listening on $MONGO_PORT (default 27017), we don't start a second instance.
set -euo pipefail

MONGO_PORT="${MONGO_PORT:-27017}"
DB_DIR="${DB_DIR:-./data/mongo}"
LOG_FILE="${LOG_FILE:-./data/mongo.log}"

mkdir -p "$DB_DIR"

# Check if anything is already listening on the port.
if (echo > "/dev/tcp/127.0.0.1/${MONGO_PORT}") >/dev/null 2>&1; then
  echo "mongod already listening on 127.0.0.1:${MONGO_PORT}"
  exit 0
fi

if ! command -v mongod >/dev/null 2>&1; then
  echo "mongod not found in PATH. Install MongoDB or update PATH." >&2
  exit 1
fi

echo "Starting mongod on 127.0.0.1:${MONGO_PORT} (dbpath=${DB_DIR})"
nohup mongod \
  --bind_ip 127.0.0.1 \
  --port "${MONGO_PORT}" \
  --dbpath "${DB_DIR}" \
  --logpath "${LOG_FILE}" \
  --logappend \
  --quiet \
  >/dev/null 2>&1 &

# Wait up to 10s for it to accept connections.
for i in $(seq 1 50); do
  if (echo > "/dev/tcp/127.0.0.1/${MONGO_PORT}") >/dev/null 2>&1; then
    echo "mongod ready"
    exit 0
  fi
  sleep 0.2
done

echo "mongod did not start within 10s. See ${LOG_FILE}" >&2
exit 1
