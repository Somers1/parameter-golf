#!/bin/bash
# Run this ON THE RUNPOD POD to send logs to your Mac.
# Then on your Mac run: runpodctl receive <code>
# Then: python plot_loss.py --no-sync

cd /workspace/parameter-golf/logs
tar czf /tmp/logs.tar.gz *.txt
runpodctl send /tmp/logs.tar.gz
