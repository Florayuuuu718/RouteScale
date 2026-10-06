from __future__ import annotations

import unittest

from train_deepspeed import initialize_deepspeed_communication


class _FakeComm:
    def __init__(self, world_size: int) -> None:
        self.world_size = world_size

    def get_world_size(self) -> int:
        return self.world_size


class _FakeDeepSpeed:
    def __init__(self, world_size: int) -> None:
        self.comm = _FakeComm(world_size)
        self.calls: list[dict] = []

    def init_distributed(self, **kwargs) -> None:
        self.calls.append(kwargs)


class InitializeDeepSpeedCommunicationTests(unittest.TestCase):
    def test_adopts_existing_process_group_before_zero_init(self) -> None:
        deepspeed = _FakeDeepSpeed(world_size=4)

        initialize_deepspeed_communication(
            deepspeed,
            backend="nccl",
            expected_world_size=4,
        )

        self.assertEqual(
            deepspeed.calls,
            [{"dist_backend": "nccl", "dist_init_required": False}],
        )

    def test_rejects_deepspeed_world_size_mismatch(self) -> None:
        deepspeed = _FakeDeepSpeed(world_size=1)

        with self.assertRaisesRegex(RuntimeError, "1 != 4"):
            initialize_deepspeed_communication(
                deepspeed,
                backend="nccl",
                expected_world_size=4,
            )


if __name__ == "__main__":
    unittest.main()
