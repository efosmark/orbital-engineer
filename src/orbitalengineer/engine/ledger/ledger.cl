#include "kernel/stride.clh"
#include "flags.clh"
#include "kernel/debug.clh"
#include "kernel/ledger.clh"

__kernel void commit_ledger(
    const uint prev_ledger_entry,
    const uint tick_id,
    const uint step_id,
    __global LedgerEntry* ledger
) {
    uint offset = get_global_id(0);
    uint ledger_full_offset = (prev_ledger_entry + 1 + offset) % LEDGER_SIZE;

    LedgerEntry* le = &ledger[ledger_full_offset];
    le->tick_id = tick_id;
    le->step_id = step_id;
}
