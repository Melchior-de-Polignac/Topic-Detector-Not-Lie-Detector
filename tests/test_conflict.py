from jspace.conflict import conflict_signal


def test_conflict_signal_high_when_active_but_not_asserted():
    tid = 42
    records = [
        {"asserts_fact": False, "activation": {tid: 5.0}},  # concealment: active, unsaid
        {"asserts_fact": False, "activation": {tid: 4.0}},
        {"asserts_fact": True,  "activation": {tid: 6.0}},   # asserted -> excluded from C
    ]
    out = conflict_signal(records, [tid])
    assert out[tid]["C"] == 4.5           # mean over the two not-asserting records
    assert out[tid]["n_conflict"] == 2
