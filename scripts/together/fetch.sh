#!/usr/bin/env bash
# Pull finished runs (.done) from the shared volume into outputs_h100/. Usage: fetch.sh [--hidden]
K=${K:-kubectl}; DST=outputs_h100; mkdir -p $DST/logs
EXCL="--exclude=hidden.npz"; [ "${1:-}" = "--hidden" ] && EXCL=""
for run in $($K exec wq-w0 -- bash -c 'cd /mnt/shared/outputs && for d in *__*; do { [ -f $d/.done ] || [ -f $d/.gpu_done ]; } && echo $d; done'); do
  [ -z "$EXCL" ] || [ ! -f $DST/$run/.done ] || continue
  $K exec wq-w0 -- tar -C /mnt/shared/outputs -cf - $EXCL $run | tar -C $DST -xf - && echo "fetched $run"
done
$K exec wq-w0 -- tar -C /mnt/shared -cf - logs | tar -C $DST -xf -
