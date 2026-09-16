"""Open3D visualizer helpers for stable camera during layer switching."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

try:
    import open3d as o3d
except ModuleNotFoundError:  # pragma: no cover - depends on local environment.
    o3d = None

# Shared on-disk view preset used across VisualizerWithKeyCallback viewers.
# Use '[' for save: letter keys are heavily occupied by stage bindings in experiments.
DEFAULT_VIEW_CONFIG_PATH = Path("outputs") / "view_config" / "open3d_view.json"
SAVE_VIEW_CONFIG_KEY = ord("[")
LOAD_VIEW_CONFIG_KEY = ord(" ")


def capture_camera_parameters(vis: "o3d.visualization.Visualizer"):
    """Save the current pinhole camera parameters from a visualizer."""
    ctr = vis.get_view_control()
    return ctr.convert_to_pinhole_camera_parameters()


def restore_camera_parameters(
    vis: "o3d.visualization.Visualizer",
    camera_params,
) -> None:
    """Restore pinhole camera parameters on a visualizer."""
    ctr = vis.get_view_control()
    try:
        ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    except TypeError:
        ctr.convert_from_pinhole_camera_parameters(camera_params)


def preserve_camera_while(
    vis: "o3d.visualization.Visualizer",
    update_fn: Callable[[], Any],
) -> Any:
    """Run ``update_fn`` without changing the current camera viewpoint."""
    camera_params = capture_camera_parameters(vis)
    result = update_fn()
    restore_camera_parameters(vis, camera_params)
    vis.update_renderer()
    return result


def add_geometry_preserve_view(
    vis: "o3d.visualization.Visualizer",
    geometry: "o3d.geometry.Geometry",
    *,
    reset_bounding_box: bool = False,
) -> bool:
    """Add geometry while optionally keeping the current viewpoint."""
    try:
        return bool(vis.add_geometry(geometry, reset_bounding_box=reset_bounding_box))
    except TypeError:
        return bool(vis.add_geometry(geometry))


def remove_geometry_preserve_view(
    vis: "o3d.visualization.Visualizer",
    geometry: "o3d.geometry.Geometry",
) -> bool:
    """Remove geometry without resetting the viewpoint."""
    try:
        return bool(vis.remove_geometry(geometry, reset_bounding_box=False))
    except TypeError:
        return bool(vis.remove_geometry(geometry))


def make_reset_view_callback(message: str = "[View] Camera reset") -> Callable:
    """Build a key callback that resets the visualizer camera."""

    def _callback(visualizer: "o3d.visualization.Visualizer") -> bool:
        visualizer.reset_view_point(True)
        visualizer.poll_events()
        visualizer.update_renderer()
        print(message)
        return False

    return _callback


def register_reset_view_keys(
    vis: "o3d.visualization.Visualizer",
    callback: Callable[["o3d.visualization.Visualizer"], bool] | None = None,
) -> None:
    """Register ``R`` / ``r`` callbacks for camera reset."""
    reset_callback = callback or make_reset_view_callback()
    vis.register_key_callback(ord("R"), reset_callback)
    vis.register_key_callback(ord("r"), reset_callback)


def extract_point_cloud_colors(point_cloud: "o3d.geometry.PointCloud") -> np.ndarray:
    """Copy per-point RGB colors from an Open3D point cloud."""
    return np.asarray(point_cloud.colors, dtype=np.float64).copy()


def preserve_gui_scene_camera(scene, backup_camera, update_fn: Callable[[], Any]) -> Any:
    """Run a GUI scene update while preserving the current camera pose."""
    backup_camera.copy_from(scene.camera)
    result = update_fn()
    scene.camera.copy_from(backup_camera)
    return result


def _as_view_config_path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_VIEW_CONFIG_PATH


def capture_view_config(vis: "o3d.visualization.Visualizer") -> dict[str, Any]:
    """Capture window size, camera pose/zoom, and render point size."""
    if o3d is None:
        raise RuntimeError("Open3D is required to capture view config.")

    camera_params = capture_camera_parameters(vis)
    intrinsic = camera_params.intrinsic
    render_option = vis.get_render_option()
    return {
        "version": 1,
        "window_width": int(intrinsic.width),
        "window_height": int(intrinsic.height),
        "point_size": float(render_option.point_size),
        "intrinsic_matrix": np.asarray(intrinsic.intrinsic_matrix, dtype=np.float64).tolist(),
        "extrinsic": np.asarray(camera_params.extrinsic, dtype=np.float64).tolist(),
    }


def save_view_config(
    vis: "o3d.visualization.Visualizer",
    path: str | Path | None = None,
) -> Path:
    """Serialize the current visualizer view to a JSON file."""
    config_path = _as_view_config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = capture_view_config(vis)
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "[View] Saved view config -> "
        f"{config_path.resolve()} "
        f"(window={config['window_width']}x{config['window_height']}, "
        f"point_size={config['point_size']:.3f})"
    )
    return config_path


def load_view_config(path: str | Path | None = None) -> dict[str, Any] | None:
    """Load a previously saved view config, or return None if missing/invalid."""
    config_path = _as_view_config_path(path)
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[View] Failed to read view config {config_path}: {exc}")
        return None
    if not isinstance(payload, dict):
        print(f"[View] Invalid view config format in {config_path}")
        return None
    return payload


def resolve_view_config_window_size(
    default_width: int = 1280,
    default_height: int = 800,
    path: str | Path | None = None,
) -> tuple[int, int]:
    """Return saved window size when available, otherwise the provided defaults."""
    config = load_view_config(path)
    if config is None:
        return int(default_width), int(default_height)
    try:
        width = int(config.get("window_width", default_width))
        height = int(config.get("window_height", default_height))
    except (TypeError, ValueError):
        return int(default_width), int(default_height)
    if width <= 0 or height <= 0:
        return int(default_width), int(default_height)
    return width, height


def _build_camera_parameters_from_config(config: dict[str, Any]):
    if o3d is None:
        raise RuntimeError("Open3D is required to restore view config.")

    try:
        width = int(config["window_width"])
        height = int(config["window_height"])
        intrinsic_matrix = np.asarray(config["intrinsic_matrix"], dtype=np.float64)
        extrinsic = np.asarray(config["extrinsic"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Incomplete view config: {exc}") from exc

    if intrinsic_matrix.shape != (3, 3) or extrinsic.shape != (4, 4):
        raise ValueError(
            "View config camera matrices must be 3x3 intrinsic and 4x4 extrinsic."
        )

    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(intrinsic_matrix[0, 0]),
        float(intrinsic_matrix[1, 1]),
        float(intrinsic_matrix[0, 2]),
        float(intrinsic_matrix[1, 2]),
    )
    camera_params.extrinsic = extrinsic
    return camera_params


def apply_view_config(
    vis: "o3d.visualization.Visualizer",
    config: dict[str, Any],
) -> None:
    """Apply camera pose/zoom and point size from a loaded view config."""
    camera_params = _build_camera_parameters_from_config(config)
    restore_camera_parameters(vis, camera_params)

    point_size = config.get("point_size")
    if point_size is not None:
        try:
            vis.get_render_option().point_size = float(point_size)
        except (TypeError, ValueError):
            pass

    vis.poll_events()
    vis.update_renderer()


def restore_view_config(
    vis: "o3d.visualization.Visualizer",
    path: str | Path | None = None,
) -> bool:
    """Load view config from disk and apply it. Returns True on success."""
    config_path = _as_view_config_path(path)
    config = load_view_config(config_path)
    if config is None:
        print(f"[View] No view config found at {config_path.resolve()}")
        return False
    try:
        apply_view_config(vis, config)
    except ValueError as exc:
        print(f"[View] Failed to apply view config {config_path}: {exc}")
        return False
    print(
        "[View] Restored view config <- "
        f"{config_path.resolve()} "
        f"(window={config.get('window_width')}x{config.get('window_height')}, "
        f"point_size={config.get('point_size')})"
    )
    print(
        "[View] Note: window size is applied on the next create_window(); "
        "camera pose/zoom and point size are restored immediately."
    )
    return True


def make_save_view_config_callback(
    path: str | Path | None = None,
) -> Callable[["o3d.visualization.Visualizer"], bool]:
    """Build a key callback that saves the current view config."""

    def _callback(visualizer: "o3d.visualization.Visualizer") -> bool:
        save_view_config(visualizer, path)
        return False

    return _callback


def make_load_view_config_callback(
    path: str | Path | None = None,
) -> Callable[["o3d.visualization.Visualizer"], bool]:
    """Build a key callback that restores view config from disk."""

    def _callback(visualizer: "o3d.visualization.Visualizer") -> bool:
        restore_view_config(visualizer, path)
        return False

    return _callback


def register_view_config_keys(
    vis: "o3d.visualization.Visualizer",
    path: str | Path | None = None,
    *,
    save_key: int = SAVE_VIEW_CONFIG_KEY,
    load_key: int = LOAD_VIEW_CONFIG_KEY,
) -> None:
    """Register ``[`` (save) and Space (load) view-config callbacks by default."""
    vis.register_key_callback(save_key, make_save_view_config_callback(path))
    vis.register_key_callback(load_key, make_load_view_config_callback(path))
