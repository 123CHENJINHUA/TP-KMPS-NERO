"""Build and fuse the original TP-KMP model with Nero demonstrations."""

import numpy as np

from .data import common_metadata, load_npz, metadata, poses, read_frames, resample_demo, save_npz
from sim_bridge.grasp import grasp_prediction_fields, grasp_skill_fields


def build_model(demos, frames, samples, components):
    from interactive_incremental_learning import ConfigParams, DataHandler, DataObject, DataObjectType, TPKMP, TPKMPConfigParams

    cfg = ConfigParams()
    handlers = []
    for demo, frame in zip(demos, frames):
        handler = DataHandler(
            demo, cfg.robot_data_structure, DataObjectType.ARRAY_2D,
            DataObject(frame, cfg.frames_data_structure, DataObjectType.ARRAY_2D),
        )
        handler.prepare_for_ml()
        handlers.append(handler)
    model = TPKMP(
        TPKMPConfigParams(k_gmm=components, number_of_test_points=samples, random_state=0),
        handlers, cfg.training_input_data_structure, cfg.training_output_data_structure,
    )
    model.train()
    return model


def train(paths, output, samples=100, components=8):
    if len(paths) < 2 or samples < 3 or not 1 <= components <= samples:
        raise ValueError("Use >= 2 demonstrations, >= 3 samples and 1 <= components <= samples")
    demos, frames, reference = [], [], None
    recordings = []
    synthetic = False
    for path in paths:
        data = load_npz(path)
        recordings.append(data)
        base, tcp = metadata(data)
        names = tuple(data["frame_names"].tolist())
        signature = base, tcp, names
        if reference is not None and signature != reference:
            raise ValueError("All demonstrations must use the same base, TCP and ordered task frames")
        reference = signature
        frame = poses(data["frames"])
        if frame.shape != (len(names), 7) or not names or len(set(names)) != len(names):
            raise ValueError("Invalid task frames")
        demos.append(resample_demo(data["demo"], samples))
        frames.append(frame)
        synthetic |= bool(data["synthetic"])
    demos, frames = np.asarray(demos), np.asarray(frames)
    grasp_fields = grasp_skill_fields(recordings, samples)
    # Same lifecycle as skill_bridge: retain data and rebuild the model on load.
    model = build_model(demos, frames, samples, components)
    preview = fuse(model, frames[0], samples)
    save_npz(output, demonstrations=demos, frames=frames, frame_names=names,
             samples=samples, components=components, preview=preview,
             **grasp_fields,
             **common_metadata(base, tcp, synthetic))
    return preview


def fuse(model, frames, samples):
    from interactive_incremental_learning import ConfigParams, DataObject, DataObjectType

    cfg = ConfigParams()
    means, covariance, epistemic, aleatoric = model.predict(np.linspace(0, 1, samples))
    result, *_ = model.fuse_predictions(
        DataObject(frames, cfg.frames_data_structure, DataObjectType.ARRAY_2D),
        covariance, aleatoric, epistemic, means,
    )
    return poses(result.get_array(cfg.training_output_data_structure))


def predict(skill_path, output, frames_path=None):
    data = load_npz(skill_path)
    grasp_fields = grasp_prediction_fields(data)
    base, tcp = metadata(data)
    demos, frames = np.asarray(data["demonstrations"], dtype=float), poses(data["frames"])
    samples, components = int(data["samples"]), int(data["components"])
    names = data["frame_names"].tolist()
    if (demos.ndim != 3 or demos.shape[0] < 2 or demos.shape[1:] != (samples, 8)
            or frames.shape != (len(demos), len(names), 7) or not names
            or len(set(names)) != len(names) or not 1 <= components <= samples):
        raise ValueError("Invalid skill data shape or model parameters")
    demos = np.asarray([resample_demo(demo, samples) for demo in demos])
    target_frames = frames[0]
    if frames_path:
        new_base, new_tcp, new_names, new_frames = read_frames(frames_path)
        if (new_base, new_tcp) != (base, tcp) or set(new_names) != set(names):
            raise ValueError("Prediction frames must match the recorded base, TCP and task-frame names")
        target_frames = np.asarray([new_frames[new_names.index(name)] for name in names])
    model = build_model(demos, frames, samples, components)
    trajectory = fuse(model, target_frames, samples)
    save_npz(output, trajectory=trajectory, phase=np.linspace(0, 1, samples),
             frames=target_frames, frame_names=names, **grasp_fields,
             **common_metadata(base, tcp, bool(data["synthetic"])))
    return trajectory


def make_example(directory):
    from pathlib import Path

    directory = Path(directory)
    phase = np.linspace(0, 1, 60)
    for index, shift in enumerate((-0.025, 0, 0.025)):
        target = np.array([0.3, shift, 0.3, 1, 0, 0, 0])
        xyz = np.column_stack([0.3 + 0.03 * phase, shift + 0.01 * np.sin(np.pi * phase), 0.35 - 0.05 * phase])
        quat = np.tile([1, 0, 0, 0], (len(phase), 1))
        save_npz(directory / f"demo_{index + 1}.npz", demo=np.column_stack([phase * 8, xyz, quat]),
                 frames=target[None], frame_names=["target"], **common_metadata("base_link", "gripper_base", True))
