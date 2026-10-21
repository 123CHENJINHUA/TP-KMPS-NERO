"""Offline train/predict commands; this process never imports Isaac Sim or ROS."""

import argparse

from nero_bridge.data import load_npz
from nero_bridge.model import predict, train
from .data import require_sim, save_json, shifted_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--demos", nargs="+", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--samples", type=int, default=100)
    fit.add_argument("--components", type=int, default=8)
    pred = sub.add_parser("predict")
    pred.add_argument("--skill", required=True)
    pred.add_argument("--frames")
    pred.add_argument("--output", required=True)
    frames = sub.add_parser("frames", help="Export a demo's target pose, optionally shifted for generalization")
    frames.add_argument("--demo", required=True)
    frames.add_argument("--offset", nargs=3, type=float, default=[0, 0, 0])
    frames.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "train":
        for path in args.demos:
            require_sim(load_npz(path))
        train(args.demos, args.output, args.samples, args.components)
    elif args.command == "predict":
        require_sim(load_npz(args.skill))
        predict(args.skill, args.output, args.frames)
    else:
        save_json(args.output, shifted_frames(load_npz(args.demo), args.offset))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
