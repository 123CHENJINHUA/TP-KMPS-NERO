"""Build a portable Nero URDF from the three model files in nero_ws (no ROS runtime)."""

import argparse
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def prepare(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "meshes").mkdir()
    robot = ET.Element("robot", name="nero")
    for name in ("nero_description.urdf", "nero_with_gripper_flange_description.xacro", "nero_with_gripper_description.xacro"):
        for element in ET.parse(source / "urdf" / name).getroot():
            if element.tag in ("link", "joint") and element.get("name") not in ("world", "world_to_base_link"):
                robot.append(element)
    for mesh in robot.findall(".//mesh"):
        filename = Path(mesh.attrib["filename"]).stem + ".stl"
        shutil.copy2(source / "meshes" / filename, destination / "meshes" / filename)
        mesh.set("filename", "meshes/" + filename)
    ET.indent(robot)
    ET.ElementTree(robot).write(destination / "nero.urdf", encoding="utf-8", xml_declaration=True)
    robot.set("name", "nero2")
    ET.ElementTree(robot).write(destination / "nero_dual_2.urdf", encoding="utf-8", xml_declaration=True)
    print(destination / "nero.urdf")
    print(destination / "nero_dual_2.urdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/home/cjh/Downloads/nero_ws/src/agx_arm_description/agx_arm_urdf/nero")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "assets/nero"))
    args = parser.parse_args()
    prepare(args.source, args.output)
