"""Lossless mixed attention/recurrent state and lifecycle coverage."""
from types import SimpleNamespace
import pytest

from edge0.backends import core
from edge0.backends.mlx.checkpoint import KVCache, ArraysCache, read
from edge0.conversation import CacheConfig, CheckpointStore
from edge0.engine.base import Edge0Engine
from edge0.config import GenerationConfig


class ToyEngine(Edge0Engine):
    """State depends on every token and a prerouter-like recurrent scalar."""
    def _build(self):
        self._reset_state()

    def _reset_state(self):
        self.cache = [KVCache(), ArraysCache(1)]
        self.routing = core.array([0.0])

    def _forward(self, ids, intra_stage=True):
        kv, recurrent = self.cache
        for tid in ids:
            v = core.array([[[[float(tid)]]]])
            kv.update_and_fetch(v, v)
            old = recurrent[0] if recurrent[0] is not None else core.array([0.0])
            recurrent[0] = old * 0.5 + tid
            self.routing = self.routing * 0.75 + tid
        logits = core.arange(8, dtype=core.float32) * (recurrent[0] + self.routing)
        core.eval(logits)
        return logits

    def _checkpoint_family_state(self):
        return {'routing': self.routing}

    def _restore_checkpoint_family(self, state):
        self.routing = state['routing']


def engine(tmp_path, namespace='toy'):
    eng = ToyEngine('', SimpleNamespace(prefill_chunk=4))
    eng.conversation_cache = CheckpointStore(CacheConfig(str(tmp_path), interval=4), namespace)
    return eng


@pytest.mark.parametrize('suffix', [[], [5], [5, 6, 7]])
def test_exact_suffix_restart_logits(tmp_path, suffix):
    a = engine(tmp_path)
    a.generate([1, 2, 3, 4], max_new_tokens=0)
    b = engine(tmp_path)
    b.generate([1, 2, 3, 4] + suffix, max_new_tokens=0)
    assert b.conversation_cache.metrics['reused_tokens'] == 4
    reference = ToyEngine('', SimpleNamespace(prefill_chunk=4))
    reference.prefill([1, 2, 3, 4] + suffix)
    assert core.allclose(b.next_logits(), reference.next_logits()).item()
    assert b.pos == 4 + len(suffix)


def test_eos_limit_and_callback_cancellation(tmp_path):
    eng = engine(tmp_path)
    ids = [1, 2, 3]
    assert eng.generate(ids, GenerationConfig(eos_ids=(7,))) == []
    assert eng.pos == 3
    eng.reset()
    assert eng.generate(ids, max_new_tokens=2) == [7, 7]
    assert eng.pos == 5
    eng.reset()
    def cancel(token):
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError, match='cancelled'):
        eng.generate(ids, max_new_tokens=8, on_token=cancel)
    hit = eng.conversation_cache.restore(ids + [7, 7, 7], read)
    assert len(hit[0]) <= 5
    assert hit[1][3] == len(hit[0])


def test_mid_prefill_boundary(tmp_path):
    eng = engine(tmp_path)
    eng.generate(list(range(1, 11)), max_new_tokens=0)
    branch = engine(tmp_path)
    branch.generate([1, 2, 3, 4, 99], max_new_tokens=0)
    assert branch.conversation_cache.metrics['reused_tokens'] == 4
    ref = ToyEngine('', SimpleNamespace(prefill_chunk=4))
    ref.prefill([1, 2, 3, 4, 99])
    assert core.allclose(branch.next_logits(), ref.next_logits()).item()


@pytest.mark.slow
@pytest.mark.parametrize('tier,env', [('edge0-8b', 'EDGE0_8B_MODEL'), ('edge0-35b', 'EDGE0_35B_MODEL')])
def test_real_weight_continuation(tmp_path, tier, env):
    import os
    from edge0 import AutoEngine
    path = os.environ.get(env)
    if not path:
        pytest.skip(f'{env} unavailable')
    eng = AutoEngine.from_pretrained(path, name=tier,
        conversation_cache=CacheConfig(str(tmp_path), interval=32))
    try:
        ids = eng._tok.encode('Implement a Python function that groups file paths by extension. ' * 8)
        eng.generate(ids, max_new_tokens=0)
        prompt_logits = eng.next_logits()
        tid = int(core.argmax(prompt_logits).item())
        next_logits = eng.step(tid)
        eng.reset()
        eng.generate(ids, max_new_tokens=0)
        assert eng.conversation_cache.metrics.get('reused_tokens') == len(ids)
        assert core.allclose(eng.next_logits(), prompt_logits, rtol=1e-5, atol=1e-5).item()
        restored_next = eng.step(tid)
        assert core.allclose(restored_next, next_logits, rtol=1e-4, atol=1e-4).item()
        # A single token appended to a processed-generation checkpoint must
        # resume from that exact state, with the normal prefill lifecycle.
        eng._save_checkpoint()
        eng.prefill([tid])
        suffix_logits = eng.next_logits()
        eng.reset()
        eng.generate(ids + [tid, tid], max_new_tokens=0)
        assert core.allclose(eng.next_logits(), suffix_logits, rtol=1e-4, atol=1e-4).item()
    finally:
        eng.close()


def test_small_qwen_recurrent_backbone(tmp_path):
    """Actual Qwen attention + gated-delta layers with random small weights."""
    from edge0.backends.mlx._impl.qwen3_5 import TextModel, TextModelArgs
    class SmallQwen(ToyEngine):
        def _build(self):
            self.model = TextModel(TextModelArgs(hidden_size=32, intermediate_size=64,
                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                vocab_size=32, linear_num_value_heads=2, linear_num_key_heads=1,
                linear_key_head_dim=16, linear_value_head_dim=16,
                full_attention_interval=2, head_dim=16))
            self._reset_state()
        def _reset_state(self):
            self.cache = self.model.make_cache()
        def _forward(self, ids, intra_stage=True):
            logits = self.model(core.array(ids)[None, :], cache=self.cache)[0, -1]
            core.eval(logits)
            return logits
        def _checkpoint_family_state(self):
            return {}
        def _restore_checkpoint_family(self, state):
            pass
    model = SmallQwen('', SimpleNamespace(prefill_chunk=4))
    model.conversation_cache = CheckpointStore(CacheConfig(str(tmp_path), interval=4), 'small-qwen')
    model.generate([1, 2, 3, 4, 5], max_new_tokens=0)
    expected = model.step(6)
    model.reset()
    model.generate([1, 2, 3, 4, 5], max_new_tokens=0)
    actual = model.step(6)
    assert core.allclose(expected, actual, rtol=1e-5, atol=1e-5).item()


@pytest.mark.parametrize('family', ['ling', 'qwen'])
def test_family_prerouter_fields_and_staging(tmp_path, family):
    from edge0.engine.checkpoint import capture, restore, FIELDS
    from edge0.prerouter.state import PrerouterState
    from edge0.backends.mlx.checkpoint import write
    block = SimpleNamespace(prev_topk_oh=core.array([2.0]), last_topk=core.array([1]),
        prerouter_m_in=core.array([3.0]), prerouter_oh=core.array([4.0]))
    layer = SimpleNamespace(m_in_cache=core.array([5.0]))
    spec = SimpleNamespace(block_of=lambda model, owner: block,
                           layer_of=lambda model, owner: layer)
    class Expert:
        last_used = [1, 2]
        _staged_state = ((), [], {2, 3})
        _last_prefill_topk = core.array([1, 2])
        def wait_staged(self):
            pass
        def reset(self):
            self._staged_state = None
        def stage_experts(self, ids):
            self._staged_state = ((), [], set(ids))
    eng = engine(tmp_path)
    eng.generate([1, 2], max_new_tokens=0)
    eng.cfg.moe_spec = spec
    eng.model = None
    eng._pg_state = PrerouterState(2, 4, 1, owners=[0])
    for i, key in enumerate(FIELDS):
        setattr(eng._pg_state, key, [core.array([float(i)]), None])
    eng._pg_stager = SimpleNamespace(cur_step=7, pg_cache={1: core.array([8.0])})
    eng._all_stream_layers = {1: Expert()}
    eng._checkpoint_family_state = lambda: capture(eng, family)
    eng.conversation_cache.clear()
    eng.conversation_cache.publish([1, 2], lambda put: write(eng, put))
    state = eng.conversation_cache.restore([1, 2], read)[1][2]
    eng._pg_state.reset()
    eng._pg_stager.pg_cache = {}
    restore(eng, state, family)
    for i, key in enumerate(FIELDS):
        assert getattr(eng._pg_state, key)[0].item() == i
    assert eng._pg_stager.cur_step == 7
    assert eng._all_stream_layers[1]._staged_state[2] == {2, 3}
    if family == 'ling':
        assert eng._pg_stager.pg_cache[1].item() == 8


def test_chat_releases_idle_state_and_reports_usage(tmp_path):
    from edge0.server.chat import ChatSession, ChatRequest, ChatMessage
    eng = engine(tmp_path)
    eng._tok = SimpleNamespace(encode=lambda text: [1, 2, 3], bos_token_id=0)
    request = ChatRequest('toy', [ChatMessage('user', 'hello')], max_tokens=2)
    _, first = ChatSession(eng, request).run()
    assert eng.pos == 0
    assert all(c.empty() for c in eng.cache)
    _, repeated = ChatSession(eng, request).run()
    assert repeated['usage']['prompt_tokens'] == 3
    assert repeated['usage']['total_tokens'] == 5
    assert repeated['usage']['prompt_tokens_details']['cached_tokens'] == 3
    assert eng.pos == 0
    def cancelled(token):
        raise RuntimeError('cancelled')
    with pytest.raises(RuntimeError):
        ChatSession(eng, request).run(on_token=cancelled)
    assert eng.pos == 0
    assert all(c.empty() for c in eng.cache)


def test_sampling_penalty_history_keeps_reused_prompt(tmp_path, monkeypatch):
    eng = engine(tmp_path)
    eng.generate([1, 2, 3, 4], max_new_tokens=0)
    eng.reset()
    seen = []
    def sample(logits, **kwargs):
        seen.append(list(kwargs['history']))
        return 6
    monkeypatch.setattr('edge0.engine.base.sample', sample)
    eng.generate([1, 2, 3, 4, 5], GenerationConfig(
        first_token_greedy=False, repetition_penalty=1.1, max_new_tokens=2))
    assert seen == [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6]]
