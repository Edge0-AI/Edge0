"""Explicit family continuation and expert-staging hooks."""
FIELDS = ('logits', 'logits_prev', 'pred_inds', 'pred_scores', 'oh', 'oh_prev')


def capture(engine, family):
    st = engine._pg_state
    result = {'pg': {k: getattr(st, k) for k in FIELDS} if st else None,
              'experts': {}, 'blocks': {}}
    for li, exp in engine._all_stream_layers.items():
        # Finish fills before observing the next decode's staged set.
        exp.wait_staged()
        staged = exp._staged_state
        result['experts'][li] = dict(last=list(exp.last_used),
            staged=sorted(staged[2]) if staged else None,
            prefill=exp._last_prefill_topk)
    if engine._pg_stager:
        result['cur_step'] = engine._pg_stager.cur_step
        if family == 'ling':
            result['pg_cache'] = engine._pg_stager.pg_cache
        for owner in st.owners:
            block = engine.cfg.moe_spec.block_of(engine.model, owner)
            fields = ('prev_topk_oh', 'last_topk') if family == 'ling' else ('prerouter_m_in', 'prerouter_oh')
            result['blocks'][owner] = {k: getattr(block, k, None) for k in fields}
            if family == 'ling':
                result['blocks'][owner]['m_in_cache'] = getattr(engine.cfg.moe_spec.layer_of(engine.model, owner), 'm_in_cache', None)
    return result


def restore(engine, state, family):
    if engine._pg_state:
        for key in FIELDS:
            setattr(engine._pg_state, key, state['pg'][key])
    if engine._pg_stager:
        engine._pg_stager.cur_step = state['cur_step']
        if family == 'ling':
            engine._pg_stager.pg_cache = state['pg_cache']
        for owner, values in state['blocks'].items():
            block = engine.cfg.moe_spec.block_of(engine.model, owner)
            for key, value in values.items():
                target = engine.cfg.moe_spec.layer_of(engine.model, owner) if key == 'm_in_cache' else block
                setattr(target, key, value)
    for li, saved in state['experts'].items():
        exp = engine._all_stream_layers[li]
        exp.wait_staged()
        exp.reset()
        exp.last_used = saved['last']
        exp._last_prefill_topk = saved['prefill']
        if saved['staged'] is not None:
            exp.stage_experts(saved['staged'])
            exp.wait_staged()
