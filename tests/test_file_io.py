import asyncio
import stat
from pathlib import Path
from threading import Event

import pytest

from xpeech.utils import file_io
from xpeech.utils.file_io import LockedFile


@pytest.mark.asyncio
@pytest.mark.parametrize("shared_instance", [False, True])
async def test_standalone_operations_are_serial_for_same_path(tmp_path, monkeypatch, shared_instance):
    path = tmp_path / "content.txt"
    path.write_text("initial", encoding="utf-8")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(path.name)
    file = LockedFile(path)
    run_file_io = file_io._run_file_io
    active = 0
    maximum = 0

    async def track_operations(operation):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0)
            return await run_file_io(operation)
        finally:
            active -= 1

    monkeypatch.setattr(file_io, "_run_file_io", track_operations)
    methods = [
        ("read_text", ()), ("write_text", ("text",)),
        ("append_text", (" appended",)), ("read_bytes", ()),
        ("write_bytes", (b"bytes",)), ("create_text", ("keep existing",)),
    ]
    calls = [
        getattr(file if shared_instance else LockedFile(alias), method)(*args)
        for method, args in methods
    ]

    results = await asyncio.wait_for(asyncio.gather(*calls), 5)

    assert maximum == 1
    assert results[-1] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("same_instance", [False, True])
async def test_transaction_reuses_lock_but_independent_tasks_wait(tmp_path, same_instance):
    file = LockedFile(tmp_path / "content.txt")
    normal = file if same_instance else LockedFile(file.path)
    start_reader = asyncio.Event()

    async def outside_reader():
        await start_reader.wait()
        return await normal.read_text()

    # 独立任务在事务开始前创建，不继承该事务的持锁状态。
    reader = asyncio.create_task(outside_reader())

    try:
        async with file.transaction() as locked:
            assert locked is file
            await locked.write_text("first")
            start_reader.set()
            await asyncio.sleep(0)
            assert not reader.done()
            async with file.transaction() as nested:
                assert nested is file
                await asyncio.wait_for(asyncio.create_task(nested.append_text(" second")), 5)
            assert not reader.done()
    finally:
        start_reader.set()
        result = await asyncio.wait_for(reader, 5)

    assert result == "first second"


@pytest.mark.asyncio
async def test_cancelled_transaction_releases_shared_lock(tmp_path):
    file = LockedFile(tmp_path / "content.txt")
    entered = asyncio.Event()

    async def hold_context():
        async with file.transaction():
            await file.write_text("first")
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(hold_context())
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(LockedFile(file.path).append_text(" next"), 5)
    assert await file.read_text() == "first next"


@pytest.mark.asyncio
async def test_exception_releases_multi_step_context(tmp_path):
    file = LockedFile(tmp_path / "content.txt")

    with pytest.raises(RuntimeError, match="stop"):
        async with file.transaction():
            await file.write_text("first")
            raise RuntimeError("stop")

    await asyncio.wait_for(file.append_text(" next"), 5)
    assert await file.read_text() == "first next"


@pytest.mark.asyncio
async def test_child_cannot_reuse_transaction_after_it_ends(tmp_path):
    file = LockedFile(tmp_path / "content.txt")
    resume = asyncio.Event()

    async def late_append():
        await resume.wait()
        await file.append_text(" appended")

    async with file.transaction():
        await file.write_text("first")
        child = asyncio.create_task(late_append())

    next_file = LockedFile(file.path)
    try:
        async with next_file.transaction():
            resume.set()
            await asyncio.sleep(0)
            assert not child.done()
            await next_file.write_text("next")
        await asyncio.wait_for(child, 5)
    finally:
        if not child.done():
            child.cancel()
        await asyncio.gather(child, return_exceptions=True)

    assert await file.read_text() == "next appended"


@pytest.mark.asyncio
@pytest.mark.parametrize("reader_name", ["read_text", "read_bytes", "super_read_text"])
async def test_readers_share_the_transaction_lock_but_other_files_do_not_wait(tmp_path, reader_name):
    path = tmp_path / "content.txt"
    path.write_text("before", encoding="utf-8")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(path.name)
    other = tmp_path / "other.txt"
    other.write_text("independent", encoding="utf-8")
    file = LockedFile(path)
    reader = None

    try:
        async with file.transaction():
            read = (
                file_io.super_read_text(file_path=alias)
                if reader_name == "super_read_text"
                else getattr(LockedFile(alias), reader_name)()
            )
            reader = asyncio.create_task(read)
            await asyncio.sleep(0)
            assert not reader.done()
            assert await asyncio.wait_for(LockedFile(other).read_text(), 5) == "independent"
            # 同一文件对象的各个方法均复用事务已经持有的锁。
            assert await file.read_text() == "before"
            await file.write_text("after")
            await file.append_text(" appended")
            assert await file.read_text() == "after appended"
            await file.write_bytes(b"after")
            assert await file.read_bytes() == b"after"
            assert not await file.create_text("must not overwrite")
            assert not reader.done()
    finally:
        if reader is not None:
            result = await asyncio.wait_for(reader, 5)

    if reader_name == "super_read_text":
        assert result[0] == "after"
    else:
        assert result == (b"after" if reader_name == "read_bytes" else "after")


@pytest.mark.asyncio
async def test_readers_see_complete_versions_and_writes_preserve_mode(tmp_path, monkeypatch):
    path = tmp_path / "script.sh"
    path.write_text("old", encoding="utf-8")
    path.chmod(0o750)
    ready = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    replace = file_io.os.replace

    def pause_before_publish(source, target):
        loop.call_soon_threadsafe(ready.set)
        if not release.wait(5):
            raise TimeoutError("test did not release publication")
        replace(source, target)

    monkeypatch.setattr(file_io.os, "replace", pause_before_publish)
    task = asyncio.create_task(LockedFile(path).write_text("new" * 100_000))
    try:
        await asyncio.wait_for(ready.wait(), 5)
        assert path.read_text(encoding="utf-8") == "old"
    finally:
        release.set()
        await task

    assert path.read_text(encoding="utf-8") == "new" * 100_000
    assert stat.S_IMODE(path.stat().st_mode) == 0o750
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.asyncio
@pytest.mark.parametrize("binary", [False, True])
async def test_failed_publication_keeps_original_and_cleans_temporary(tmp_path, monkeypatch, binary):
    path = tmp_path / "content"
    path.write_bytes(b"original")

    def fail_publish(*_args):
        raise OSError("disk error")

    monkeypatch.setattr(file_io.os, "replace", fail_publish)
    with pytest.raises(OSError, match="disk error"):
        if binary:
            await LockedFile(path).write_bytes(b"replacement")
        else:
            await LockedFile(path).write_text("replacement")

    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.asyncio
async def test_cancelled_write_keeps_lock_until_worker_finishes(tmp_path, monkeypatch):
    path = tmp_path / "content"
    path.write_text("old", encoding="utf-8")
    ready = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()
    replace = file_io.os.replace

    def pause_before_publish(source, target):
        loop.call_soon_threadsafe(ready.set)
        if not release.wait(5):
            raise TimeoutError("test did not release publication")
        replace(source, target)

    monkeypatch.setattr(file_io.os, "replace", pause_before_publish)
    first = asyncio.create_task(LockedFile(path).write_text("first"))
    second = None
    try:
        await asyncio.wait_for(ready.wait(), 5)
        first.cancel()
        await asyncio.sleep(0)
        # 重复取消任务也不能提前释放文件锁。
        first.cancel()
        second = asyncio.create_task(LockedFile(path).write_text("second"))
        await asyncio.sleep(0)
        assert not first.done()
        assert not second.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        if second is not None:
            await second

    assert path.read_text(encoding="utf-8") == "second"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.asyncio
async def test_create_if_absent_never_overwrites_a_competing_creator(tmp_path, monkeypatch):
    path = tmp_path / "MEMORY.md"
    link = file_io.os.link

    def competing_create(source, target):
        Path(target).write_text("user content", encoding="utf-8")
        link(source, target)

    monkeypatch.setattr(file_io.os, "link", competing_create)

    assert not await LockedFile(path).create_text("template")
    assert path.read_text(encoding="utf-8") == "user content"
    assert list(tmp_path.iterdir()) == [path]
