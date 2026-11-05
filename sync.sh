#!/usr/bin/env bash
# sync.sh -- push the working tree to the GPU pod and normalise line endings.
#
# WHY: this repo is edited on Windows, where several tools write CRLF.  A shell
# script with CRLF fails remotely with the notoriously unhelpful
#   syntax error near unexpected token `$'{\r''
# and Python only fails at import time.  Rather than trusting every editor, the
# remote side strips CR unconditionally after unpacking.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-school-gpu}"
DEST="${DEST:-/home/user/BRCTTA}"
cd "$HERE"
tar -czf /tmp/brctta_src.tgz src scripts configs docs README.md sync.sh 2>/dev/null
scp -q /tmp/brctta_src.tgz "$HOST:$DEST/"
ssh "$HOST" "cd $DEST && tar -xzf brctta_src.tgz && find scripts src configs docs -type f -exec sed -i 's/\r\$//' {} + && bash -n scripts/gpu_run.sh && echo '[sync] ok'" 2>&1 | grep -v "post-quantum\|store now\|openssh\|^\*\*"
