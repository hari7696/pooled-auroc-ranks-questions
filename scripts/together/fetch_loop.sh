#!/usr/bin/env bash
# Keep pulling finished runs (light files + hidden.npz, 3 in parallel) until all expected runs are local.
export MSYS_NO_PATHCONV=1; K=${K:-kubectl}; DST=outputs_h100; N=${N:-18}
pull() {  # $1 = run; hidden.npz is retried until its size matches the remote copy
  $K exec wq-w0 -- tar -C /mnt/shared/outputs -cf - --exclude=hidden.npz $1 | tar -C $DST -xf - 2>/dev/null
  local want=$($K exec wq-w0 -- stat -c %s /mnt/shared/outputs/$1/hidden.npz)
  for i in 1 2 3 4 5; do
    [ "$(stat -c %s $DST/$1/hidden.npz 2>/dev/null)" = "$want" ] && break
    $K exec wq-w0 -- cat /mnt/shared/outputs/$1/hidden.npz > $DST/$1/hidden.npz.part 2>/dev/null
    [ "$(stat -c %s $DST/$1/hidden.npz.part)" = "$want" ] && mv $DST/$1/hidden.npz.part $DST/$1/hidden.npz
  done
  if [ "$(stat -c %s $DST/$1/hidden.npz 2>/dev/null)" = "$want" ]; then touch $DST/$1/.fetched; echo "fetched $1 $(date +%T)"
  else echo "hidden incomplete $1"; fi
}
while true; do
  for run in $($K exec wq-w0 -- bash -c 'cd /mnt/shared/outputs && for d in *__*; do { [ -f $d/.done ] || [ -f $d/.gpu_done ]; } && echo $d; done'); do
    [ -f $DST/$run/.fetched ] && continue
    while [ $(jobs -r | wc -l) -ge 2 ]; do sleep 2; done
    pull $run &
  done
  wait
  $K exec wq-w0 -- tar -C /mnt/shared -cf - logs | tar -C $DST -xf - 2>/dev/null
  [ $(ls -d $DST/*__*/.fetched 2>/dev/null | wc -l) -ge $N ] && { echo "ALL FETCHED"; break; }
  sleep 30
done
