"""Build the two-arm keel assembly workcell from the saved 0929 scene.

Run with Isaac Sim's Python (which provides pxr). All coordinates are metres
and are local to /World/Environment in 0929_grasp.usd.
"""

from pathlib import Path
import argparse
import math

from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade


ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "assets" / "scenes"
SOURCE = SCENE_DIR / "0929_grasp.usd"
OUTPUT = SCENE_DIR / "keel_assembly.usda"

# Measured from the four circular openings of the saved keel mesh. The two
# left holes are Ø12 mm and the two right holes are Ø10.5 mm.
KEEL_HOLE_X = (-0.0615, 0.0615)
KEEL_HOLE_Y = (-0.034, 0.034)
SCREW_SHAFT_RADIUS = 0.004
WALL_HOLE_RADIUS = 0.0045
RACK_SLOT = 0.0105
WALL_CENTER_XY = (-0.62, 0.0)
WALL_SIZE = (0.36, 0.32, 0.030)  # Width, height, thickness.
WALL_YAW_DEG = 90.0  # Local -Y front faces +X, toward both arm bases.


def color(prim, rgb):
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def translate(prim, xyz):
    UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*xyz))


def cuboid(stage, path, xyz, size, rgb):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    translate(cube.GetPrim(), xyz)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*size))
    color(cube.GetPrim(), rgb)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    return cube


def mesh_from_triangles(stage, path, points, triangles, rgb, collide):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in points])
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr([i for tri in triangles for i in tri])
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    color(mesh.GetPrim(), rgb)
    if collide:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        # A static triangle mesh preserves the open screw holes. A convex hull
        # would cap them and prevent insertion.
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    return mesh


def make_wall(stage, parent, table_top):
    wall = UsdGeom.Xform.Define(stage, parent + "/AssemblyWall")
    wall.GetPrim().SetDocumentation("Fixed tabletop wall with four through holes matching the keel mounting pattern")
    cx, cy = WALL_CENTER_XY
    width, height, thickness = WALL_SIZE
    # Author the plate around its own base, so the Stage Transform controls its
    # placement and orientation without changing the mounting-hole dimensions.
    bottom = 0.0
    base_z = table_top + 0.0005
    translate(wall.GetPrim(), (cx, cy, base_z))
    UsdGeom.Xformable(wall.GetPrim()).AddRotateZOp().Set(WALL_YAW_DEG)
    keel_center_z = 0.165 - 0.0005
    hole_x = list(KEEL_HOLE_X)
    hole_z = [keel_center_z + y for y in KEEL_HOLE_Y]
    xmin, xmax = -width / 2, width / 2
    zmin, zmax = bottom, bottom + height
    front, back = -thickness / 2, thickness / 2
    cell = 0.015
    xs = [xmin, hole_x[0] - cell, hole_x[0] + cell,
          hole_x[1] - cell, hole_x[1] + cell, xmax]
    zs = [zmin, hole_z[0] - cell, hole_z[0] + cell,
          hole_z[1] - cell, hole_z[1] + cell, zmax]
    points, tris, index = [], [], {}

    def vertex(x, y, z):
        key = tuple(round(float(v), 9) for v in (x, y, z))
        if key not in index:
            index[key] = len(points)
            points.append(key)
        return index[key]

    def face(a, b, c, d):
        tris.extend(((a, b, c), (a, c, d)))

    for xi in range(len(xs) - 1):
        for zi in range(len(zs) - 1):
            x0, x1, z0, z1 = xs[xi], xs[xi + 1], zs[zi], zs[zi + 1]
            if xi in (1, 3) and zi in (1, 3):
                hx, hz = hole_x[0 if xi == 1 else 1], hole_z[0 if zi == 1 else 1]
                for n in range(32):
                    a = 2 * math.pi * n / 32
                    b = 2 * math.pi * (n + 1) / 32

                    def outer(angle):
                        dx, dz = math.cos(angle), math.sin(angle)
                        distance = min((x1 - hx if dx >= 0 else hx - x0) / max(abs(dx), 1e-12),
                                       (z1 - hz if dz >= 0 else hz - z0) / max(abs(dz), 1e-12))
                        return hx + dx * distance, hz + dz * distance

                    oa, ob = outer(a), outer(b)
                    ia = (hx + WALL_HOLE_RADIUS * math.cos(a), hz + WALL_HOLE_RADIUS * math.sin(a))
                    ib = (hx + WALL_HOLE_RADIUS * math.cos(b), hz + WALL_HOLE_RADIUS * math.sin(b))
                    for y, reverse in ((front, False), (back, True)):
                        quad = [vertex(*((v[0], y, v[1]))) for v in (oa, ob, ib, ia)]
                        if reverse:
                            quad.reverse()
                        face(*quad)
                    # Interior cylindrical surface, with a real through passage.
                    face(vertex(ia[0], front, ia[1]), vertex(ib[0], front, ib[1]),
                         vertex(ib[0], back, ib[1]), vertex(ia[0], back, ia[1]))
            else:
                for y, reverse in ((front, False), (back, True)):
                    quad = [vertex(x, y, z) for x, z in ((x0, z0), (x1, z0), (x1, z1), (x0, z1))]
                    if reverse:
                        quad.reverse()
                    face(*quad)

    # Close the four outside edges of the plate.
    for a, b in (((xmin, zmin), (xmax, zmin)), ((xmax, zmin), (xmax, zmax)),
                 ((xmax, zmax), (xmin, zmax)), ((xmin, zmax), (xmin, zmin))):
        face(vertex(a[0], front, a[1]), vertex(a[0], back, a[1]),
             vertex(b[0], back, b[1]), vertex(b[0], front, b[1]))
    mesh_from_triangles(stage, str(wall.GetPath()) + "/Plate", points, tris,
                        (0.72, 0.73, 0.72), collide=True)
    angle = math.radians(WALL_YAW_DEG)
    holes = [(cx + hx * math.cos(angle) - front * math.sin(angle),
              cy + hx * math.sin(angle) + front * math.cos(angle), base_z + hz)
             for hx in hole_x for hz in hole_z]
    return {"hole_centers": holes, "size": WALL_SIZE,
            "front_normal": (math.sin(angle), -math.cos(angle), 0.0)}


def make_rack(stage, parent, table_top):
    rack = UsdGeom.Xform.Define(stage, parent + "/ScrewRack")
    rack.GetPrim().SetDocumentation("Fixed four-position holder; square sockets remain open for screw shafts")
    base = str(rack.GetPath())
    cx = -0.35
    ys = (0.115, 0.165, 0.215, 0.265)
    top = table_top + 0.105
    thickness = 0.008
    width = 0.060
    length = 0.200
    ylo, yhi = ys[0] - 0.025, ys[-1] + 0.025
    side = (width - RACK_SLOT) / 2
    metal = (0.28, 0.35, 0.42)
    for side_index, side_x in enumerate((-1, 1), 1):
        cuboid(stage, base + f"/SideRail{side_index}",
               (cx + side_x * (RACK_SLOT + side) / 2, (ylo + yhi) / 2, top - thickness / 2),
               (side, length, thickness), metal)
    segments = [(ylo, ys[0] - RACK_SLOT / 2)]
    segments += [(a + RACK_SLOT / 2, b - RACK_SLOT / 2) for a, b in zip(ys, ys[1:])]
    segments += [(ys[-1] + RACK_SLOT / 2, yhi)]
    for i, (a, b) in enumerate(segments, 1):
        cuboid(stage, base + f"/CrossRail{i}", (cx, (a + b) / 2, top - thickness / 2),
               (RACK_SLOT, b - a, thickness), metal)
    leg_bottom = table_top + 0.0005
    for i, (x, y) in enumerate(((cx + dx, edge) for dx in (-0.022, 0.022)
                               for edge in (ylo + 0.009, yhi - 0.009)), 1):
        cuboid(stage, base + f"/Leg{i}", (x, y, (leg_bottom + top - thickness) / 2),
               (0.012, 0.012, top - thickness - leg_bottom), metal)
    return cx, ys, top


def make_screws(stage, parent, rack, material):
    cx, ys, top = rack
    for i, y in enumerate(ys, 1):
        root = UsdGeom.Xform.Define(stage, parent + f"/Screw{i:02d}")
        prim = root.GetPrim()
        prim.SetDocumentation("Graspable demo bolt: Ø8 mm shaft, Ø18 mm x 14 mm head, no thread model")
        translate(prim, (cx, y, top + 0.001))
        UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(0.025)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
        for name, radius, height, z, rgb in (
            ("Shaft", SCREW_SHAFT_RADIUS, 0.028, -0.014, (0.68, 0.69, 0.70)),
            ("GripHead", 0.009, 0.014, 0.007, (0.76, 0.57, 0.24)),
        ):
            cyl = UsdGeom.Cylinder.Define(stage, str(root.GetPath()) + "/" + name)
            cyl.CreateRadiusAttr(radius)
            cyl.CreateHeightAttr(height)
            cyl.CreateAxisAttr(UsdGeom.Tokens.z)
            translate(cyl.GetPrim(), (0, 0, z))
            color(cyl.GetPrim(), rgb)
            UsdPhysics.CollisionAPI.Apply(cyl.GetPrim()).CreateCollisionEnabledAttr(True)


def build(force=False):
    if OUTPUT.exists() and not force:
        raise FileExistsError(f"{OUTPUT} already exists; pass --force to regenerate it")
    source = Usd.Stage.Open(str(SOURCE))
    if source is None or not source.GetPrimAtPath("/World/Environment/Geometry").IsValid():
        raise RuntimeError(f"Missing keel in source scene: {SOURCE}")
    table = source.GetPrimAtPath("/World/Environment/Cube")
    if not table.IsValid():
        raise RuntimeError("The saved scene has no tabletop at /World/Environment/Cube")
    transform = UsdGeom.Xformable(table)
    values = {op.GetOpType(): op.Get() for op in transform.GetOrderedXformOps()}
    table_top = float(values[UsdGeom.XformOp.TypeTranslate][2]
                      + values[UsdGeom.XformOp.TypeScale][2] / 2)
    if not 0.90 < table_top < 1.05:
        raise RuntimeError(f"Unexpected tabletop height: {table_top}")
    if OUTPUT.exists():
        OUTPUT.unlink()
    stage = Usd.Stage.CreateNew(str(OUTPUT))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    env = UsdGeom.Xform.Define(stage, "/World/Environment")
    env.GetPrim().GetReferences().AddReference(SOURCE.name, "/World/Environment")
    parent = str(env.GetPath())
    material = UsdShade.Material.Get(stage, parent + "/GraspMaterial")
    if not material.GetPrim().IsValid():
        raise RuntimeError("The saved scene has no grasp material")
    wall = make_wall(stage, parent, table_top)
    rack = make_rack(stage, parent, table_top)
    make_screws(stage, parent, rack, material)
    stage.GetRootLayer().Save()
    print(f"Wrote {OUTPUT}")
    print(f"Table top (environment local): {table_top:.6f} m")
    print(f"Wall size: {wall['size']}; front normal: {wall['front_normal']}")
    print(f"Wall hole centers (environment local): {wall['hole_centers']}")
    print(f"Rack screw centers X={rack[0]}, Y={rack[1]}; rack top Z={rack[2]:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace the generated assembly scene")
    build(force=parser.parse_args().force)
