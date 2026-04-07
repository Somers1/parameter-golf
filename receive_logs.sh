#!/bin/bash
# Run this on your Mac after running sync_logs.sh on the pod.
# Usage: bash receive_logs.sh <code>
# Example: bash receive_logs.sh 8338-galileo-collect-fidel

set -e

if [ -z "$1" ]; then
    echo "Usage: bash receive_logs.sh <code>"
    echo "Run sync_logs.sh on the pod first to get the code."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"
cd "$SCRIPT_DIR/logs"

runpodctl receive "$1"
tar xzf logs.tar.gz
rm logs.tar.gz

echo ""
echo "Logs synced"
