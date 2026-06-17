#!/usr/bin/env bash
#
# Stream /home/marek/batch to Google Drive as per-subdir .tar archives, then
# delete each subdir locally — freeing disk space incrementally without ever
# needing a local copy of the tar (tar is piped straight into rclone).
#
# Why tar instead of `rclone move` of the tree: /home/marek/batch has ~184k
# small files; uploading them individually is dominated by Drive per-file API
# overhead. A handful of big .tar objects upload far faster and keep Drive tidy.
#
# Safety: a subdir is deleted ONLY after the bytes rclone uploaded match the
# bytes tar produced (counted in-line via tee+wc). If anything fails, the local
# data is left untouched. Re-running resumes: already-uploaded-and-deleted
# subdirs are simply skipped (they no longer exist locally).
#
# Prerequisites:
#   1) rclone remote configured (run `rclone config`, e.g. a "gdrive" drive remote)
#   2) >= 240 GB free quota on the Drive account
#
# Usage:
#   ./backup_batch_to_gdrive.sh                 # upload + delete (default remote gdrive:batch_backup)
#   REMOTE=gdrive:batch_backup ./backup_batch_to_gdrive.sh
#   DELETE=0 ./backup_batch_to_gdrive.sh        # upload only, keep local copies (safe first pass)
#
set -euo pipefail

SRC="${SRC:-/home/marek/batch}"
REMOTE="${REMOTE:-gdrive:batch_backup}"
DELETE="${DELETE:-1}"                 # 1 = delete each subdir after verified upload
LOG="${LOG:-$SRC/../batch_backup.log}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Build the list of chunks: every immediate subdir of outputs/, plus the other
# top-level dirs in $SRC. Each becomes one .tar on the remote, mirroring paths.
chunks=()
for d in "$SRC"/outputs/*/; do chunks+=("outputs/$(basename "$d")"); done
for d in "$SRC"/*/; do
  name="$(basename "$d")"
  [ "$name" = "outputs" ] && continue
  chunks+=("$name")
done

log "Backing up $SRC -> $REMOTE  (${#chunks[@]} chunks, DELETE=$DELETE)"

for chunk in "${chunks[@]}"; do
  local_path="$SRC/$chunk"
  remote_path="$REMOTE/$chunk.tar"

  if [ ! -e "$local_path" ]; then
    log "SKIP  $chunk  (already gone locally — done in a previous run)"
    continue
  fi

  log "PACK+UPLOAD  $chunk  ->  $remote_path"
  sent_bytes_file="$(mktemp)"

  # tar -> tee(count bytes) -> rclone rcat. pipefail makes any stage's failure fail the run.
  if tar -C "$SRC" -cf - "$chunk" \
        | tee >(wc -c >"$sent_bytes_file") \
        | rclone rcat --drive-chunk-size 256M "$remote_path"; then
    sent="$(cat "$sent_bytes_file")"
    rm -f "$sent_bytes_file"
    remote="$(rclone size --json "$remote_path" 2>/dev/null | grep -o '"bytes":[0-9]*' | grep -o '[0-9]*' || echo 0)"

    if [ "$sent" -gt 0 ] && [ "$sent" = "$remote" ]; then
      log "VERIFIED  $chunk  ($sent bytes match on Drive)"
      if [ "$DELETE" = "1" ]; then
        rm -rf "$local_path"
        log "DELETED   $local_path  (freed locally)"
      fi
    else
      log "MISMATCH  $chunk  sent=$sent remote=$remote — NOT deleting. Investigate/re-run."
      exit 1
    fi
  else
    rm -f "$sent_bytes_file"
    log "FAILED    $chunk upload — local data left intact. Re-run to retry."
    exit 1
  fi
done

log "All chunks done. Local space freed: re-check with  df -h $SRC"
