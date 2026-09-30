"""Which RealESRGAN weights file load_rrdbnet picks. Needs no weights on disk."""
from hy3dpaint_mlx.realesrgan import RRDBNET_MLX, RRDBNET_NPZ, pick_rrdbnet_weights


def _touch(root, rel):
    f = root / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(b"\0")


def test_prefers_the_downloaded_mlx_copy(tmp_path):
    _touch(tmp_path, RRDBNET_MLX)
    _touch(tmp_path, RRDBNET_NPZ)
    assert pick_rrdbnet_weights(str(tmp_path)).endswith(RRDBNET_MLX)


def test_falls_back_to_the_hand_converted_npz(tmp_path):
    _touch(tmp_path, RRDBNET_NPZ)
    assert pick_rrdbnet_weights(str(tmp_path)).endswith(RRDBNET_NPZ)


def test_none_when_neither_is_there(tmp_path):
    assert pick_rrdbnet_weights(str(tmp_path)) is None
