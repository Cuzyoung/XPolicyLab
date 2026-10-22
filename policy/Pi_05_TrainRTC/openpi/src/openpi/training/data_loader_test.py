import dataclasses
import threading
import time

import jax
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


class _IndexDataset:
    def __init__(self, size=41, error_at=None):
        self.size = size
        self.error_at = error_at

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if index == self.error_at:
            raise ValueError("bad dataset index")
        return {"index": np.asarray(index, dtype=np.int64)}


def _index_batches(prefetch, *, workers=0, count=23):
    loader = _data_loader.TorchDataLoader(
        _IndexDataset(), local_batch_size=4, num_batches=count,
        num_workers=workers, prefetch_batches=prefetch, seed=0,
        shuffle=True, framework="pytorch",
    )
    try:
        return [batch["index"].numpy().copy() for batch in loader]
    finally:
        loader._cancel_workers()


@pytest.mark.parametrize("count", [0, 1, 10, 11, 23])
def test_host_prefetch_epoch_order_and_finite_limit(count):
    original = _index_batches(0, count=count)
    prefetched = _index_batches(3, count=count)
    assert len(original) == len(prefetched) == count
    for old, new in zip(original, prefetched, strict=True):
        np.testing.assert_array_equal(old, new)


def test_host_prefetch_spawn_epoch_order():
    # Two complete epochs plus three batches; the final sample is dropped in
    # each epoch. Do not replace the original finite-epoch sampler.
    original = _index_batches(0, workers=2)
    prefetched = _index_batches(3, workers=2)
    for old, new in zip(original, prefetched, strict=True):
        np.testing.assert_array_equal(old, new)


def test_host_prefetch_keeps_device_conversion_on_consumer(monkeypatch):
    consumer = threading.get_ident()
    threads = []
    convert = _data_loader.jax.make_array_from_process_local_data
    def record(*args, **kwargs):
        threads.append(threading.get_ident())
        return convert(*args, **kwargs)
    monkeypatch.setattr(_data_loader.jax, "make_array_from_process_local_data", record)
    loader = _data_loader.TorchDataLoader(
        _IndexDataset(), local_batch_size=4, num_batches=3, prefetch_batches=2,
    )
    assert len(list(loader)) == 3
    assert threads and set(threads) == {consumer}


def test_host_prefetch_dataset_error_is_not_swallowed():
    loader = _data_loader.TorchDataLoader(
        _IndexDataset(error_at=8), local_batch_size=4, num_workers=2,
        prefetch_batches=3, framework="pytorch",
    )
    iterator = iter(loader)
    try:
        assert next(iterator)["index"].tolist() == [0, 1, 2, 3]
        assert next(iterator)["index"].tolist() == [4, 5, 6, 7]
        workers = list(loader.torch_loader._iterator._workers)
        with pytest.raises(ValueError, match="bad dataset index"):
            next(iterator)
        assert all(not worker.is_alive() for worker in workers)
    finally:
        iterator.close()


def test_host_prefetch_close_stops_thread_and_workers():
    loader = _data_loader.TorchDataLoader(
        _IndexDataset(), local_batch_size=4, num_workers=2,
        prefetch_batches=2, framework="pytorch",
    )
    iterator = iter(loader)
    next(iterator)
    workers = list(loader.torch_loader._iterator._workers)
    iterator.close()
    assert all(not worker.is_alive() for worker in workers)
    assert not any(thread.name == "openpi-cpu-prefetch" for thread in threading.enumerate())


def test_host_prefetch_bounded_queue_and_error():
    produced = []
    def source():
        for index in range(20):
            produced.append(index)
            yield index
        raise ValueError("prefetch source failed")
    iterator = _data_loader._HostPrefetchIterator(source(), 2, lambda: None)
    try:
        deadline = time.monotonic() + 2
        while iterator._queue.qsize() < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert iterator._queue.qsize() == 2
        assert len(produced) <= 3
        assert [next(iterator) for _ in range(20)] == list(range(20))
        with pytest.raises(ValueError, match="prefetch source failed"):
            next(iterator)
    finally:
        iterator.close()
    assert not iterator._thread.is_alive()


def test_host_prefetch_close_interrupts_waiting_producer():
    entered = threading.Event()
    cancelled = threading.Event()
    def source():
        entered.set()
        assert cancelled.wait(5)
        yield 1
    iterator = _data_loader._HostPrefetchIterator(source(), 2, cancelled.set)
    assert entered.wait(1)
    iterator.close()
    assert cancelled.is_set()
    assert not iterator._thread.is_alive()


def test_host_prefetch_config_default():
    config = _config.TrainConfig(name="prefetch_test", batch_size=4)
    assert config.prefetch_batches == 0
    config = dataclasses.replace(config, prefetch_batches=2, num_workers=0)
    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    assert loader._data_loader._prefetch_batches == 2
    assert len(list(loader)) == 2


class _SlowIndexDataset(_IndexDataset):
    def __getitem__(self, index):
        if index == 8:
            time.sleep(20)
        return super().__getitem__(index)


def test_host_prefetch_close_while_worker_is_loading():
    loader = _data_loader.TorchDataLoader(
        _SlowIndexDataset(), local_batch_size=4, num_workers=2,
        prefetch_batches=2, framework="pytorch",
    )
    iterator = iter(loader)
    next(iterator)
    next(iterator)
    workers = list(loader.torch_loader._iterator._workers)
    # The producer now waits for the slow third batch in a worker process.
    time.sleep(0.1)
    started = time.monotonic()
    iterator.close()
    assert time.monotonic() - started < 15
    assert all(not worker.is_alive() for worker in workers)
    assert not any(thread.name == "openpi-cpu-prefetch" for thread in threading.enumerate())



def test_host_prefetch_process_exits_with_large_pending_batches(tmp_path):
    # Exercise Python's atexit queue finalizers, not just iterator.close().
    import subprocess
    import sys
    import textwrap

    script = tmp_path / "large_batch_exit.py"
    script.write_text(textwrap.dedent(f"""
        import faulthandler
        import importlib.util
        import sys
        import time
        import numpy as np

        spec = importlib.util.spec_from_file_location(
            "openpi.training.data_loader", {_data_loader.__file__!r})
        loader_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = loader_module
        spec.loader.exec_module(loader_module)

        class LargeDataset:
            def __len__(self):
                return 128
            def __getitem__(self, index):
                return {{"data": np.full(29_852_352, index, dtype=np.uint8)}}

        if __name__ == "__main__":
            faulthandler.enable()
            faulthandler.dump_traceback_later(30, exit=True)
            loader = loader_module.TorchDataLoader(
                LargeDataset(), local_batch_size=1, num_workers=2,
                prefetch_batches=2, framework="pytorch")
            iterator = iter(loader)
            next(iterator)
            time.sleep(0.2)
            iterator.close()
            print("ITERATOR_CLOSED", flush=True)
    """))
    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=45, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ITERATOR_CLOSED" in result.stdout
