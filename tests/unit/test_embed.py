from __future__ import annotations

import pytest

from acharya.rag.embed import cosine


def test_cosine_is_deterministic_and_rejects_mismatched_dimensions() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    with pytest.raises(ValueError, match="dimensions"):
        cosine([1.0], [1.0, 2.0])


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_retrieval_models_use_requested_device(monkeypatch, capsys, tmp_path, device):
    import sys
    from types import SimpleNamespace

    from acharya.rag.embed import BGEDenseEncoder, BGEReranker, ResolvedModel

    calls = []

    def constructor(*args, **kwargs):
        calls.append(kwargs['device'])
        return SimpleNamespace(device=device)

    monkeypatch.setenv('ACHARYA_RETRIEVAL_DEVICE', device)
    monkeypatch.setitem(sys.modules, 'torch',
                        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setitem(sys.modules, 'torch.nn', SimpleNamespace(Identity=lambda: None))
    monkeypatch.setitem(sys.modules, 'sentence_transformers',
                        SimpleNamespace(SentenceTransformer=constructor, CrossEncoder=constructor))
    model = ResolvedModel('test', 'revision', tmp_path, {})
    BGEDenseEncoder(model, '')
    BGEReranker(model)
    assert calls == [device, device]
    assert capsys.readouterr().out.count('retrieval_model_loaded') == 2


def test_cuda_request_fails_without_gpu(monkeypatch):
    import sys
    from types import SimpleNamespace

    from acharya.rag.embed import retrieval_device

    monkeypatch.setenv('ACHARYA_RETRIEVAL_DEVICE', 'cuda')
    monkeypatch.setitem(sys.modules, 'torch',
                        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match='CUDA is unavailable'):
        retrieval_device()


def test_device_mismatch_is_not_silent():
    from types import SimpleNamespace

    from acharya.rag.embed import _report_device

    with pytest.raises(RuntimeError, match='device mismatch'):
        _report_device(SimpleNamespace(device='cpu'), 'reranker', 'cuda')
