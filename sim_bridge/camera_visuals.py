"""Visible camera housing and axes in the camera's local USD frame."""

DEFAULT_VISUALIZATION = dict(show_body=True, show_axes=True, axis_length_m=.08)


def visualization_options(global_options=None, local_options=None):
    for value in (global_options, local_options):
        if value is not None and not isinstance(value, dict):
            raise ValueError('visualization must be an object')
    options = {**DEFAULT_VISUALIZATION, **(global_options or {}), **(local_options or {})}
    for key in ('show_body', 'show_axes'):
        if type(options[key]) is not bool:
            raise ValueError(f'visualization.{key} must be true or false')
    length = options['axis_length_m']
    if isinstance(length, bool) or not isinstance(length, (int, float)) or not .005 <= length <= 1.:
        raise ValueError('visualization.axis_length_m must be in [0.005, 1.0]')
    return options


def create_camera_visualization(stage, camera_path, options):
    """Attach visual geometry only: no rigid bodies, collision shapes or sensors."""
    from pxr import Gf, UsdGeom, UsdShade, Sdf

    options = visualization_options(local_options=options)
    units = UsdGeom.GetStageMetersPerUnit(stage)
    path = camera_path + '/Visualization'
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().SetDisplayName('Camera housing and local axes')

    def group(name, visible):
        node = UsdGeom.Xform.Define(stage, path + '/' + name)
        node.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited if visible else UsdGeom.Tokens.invisible)
        return node.GetPath().pathString

    materials = {}
    for name, color, metallic, emissive in [
        ('Silver', (.55, .58, .62), .6, False), ('Black', (.018, .02, .025), 0., False),
        ('Lens', (.015, .025, .04), 0., False),
        ('Red', (1., .04, .04), 0., True), ('Green', (.04, 1., .08), 0., True),
        ('Blue', (.05, .2, 1.), 0., True), ('Yellow', (1., .75, .02), 0., True)]:
        material = UsdShade.Material.Define(stage, path + '/Looks/' + name)
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Shader'))
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.35)
        if emissive:
            shader.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
        materials[name] = (material, color)

    def finish(shape, position, rotation, material, scale=None):
        transform = UsdGeom.Xformable(shape.GetPrim())
        translate = transform.GetTranslateOp() or transform.AddTranslateOp()
        rotate = transform.GetRotateXYZOp() or transform.AddRotateXYZOp()
        translate.Set(Gf.Vec3d(*(v / units for v in position)))
        rotate.Set(Gf.Vec3f(*rotation))
        ops = [translate, rotate]
        if scale is not None:
            scaling = transform.GetScaleOp() or transform.AddScaleOp()
            scaling.Set(Gf.Vec3f(*(v / units for v in scale)))
            ops.append(scaling)
        transform.SetXformOpOrder(ops, resetXformStack=False)
        mat, color = materials[material]
        shape.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
        UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(mat)

    body = group('Body', options['show_body'])
    housing = UsdGeom.Cube.Define(stage, body + '/Housing')
    housing.CreateSizeAttr().Set(1.)
    finish(housing, (0., 0., .014), (0., 0., 0.), 'Silver', (.09, .025, .02))
    bezel = UsdGeom.Cube.Define(stage, body + '/FrontPanel')
    bezel.CreateSizeAttr().Set(1.)
    finish(bezel, (0., 0., .003), (0., 0., 0.), 'Black', (.086, .021, .002))
    for index, x in enumerate((-.032, 0., .032)):
        for suffix, radius, height, z, material in [
            ('Ring', .0045, .002, .001, 'Silver'),
            ('Glass', .0035, .0005, -.00035, 'Lens')]:
            lens = UsdGeom.Cylinder.Define(stage, body + f'/Lens{index}{suffix}')
            lens.CreateAxisAttr().Set(UsdGeom.Tokens.z)
            lens.CreateRadiusAttr().Set(radius / units)
            lens.CreateHeightAttr().Set(height / units)
            finish(lens, (x, 0., z), (0., 0., 0.), material)

    axes = group('Axes', options['show_axes'])
    length = options['axis_length_m']

    def arrow(name, direction, rotation, color, origin=(0., 0., 0.)):
        node = UsdGeom.Xform.Define(stage, axes + '/' + name)
        node.GetPrim().SetDisplayName(name)
        shaft = UsdGeom.Cylinder.Define(stage, node.GetPath().AppendChild('Shaft'))
        shaft.CreateAxisAttr().Set(UsdGeom.Tokens.z)
        shaft.CreateRadiusAttr().Set(length * .015 / units)
        shaft.CreateHeightAttr().Set(length * .8 / units)
        finish(shaft, tuple(o + d * length * .4 for o, d in zip(origin, direction)), rotation, color)
        tip = UsdGeom.Cone.Define(stage, node.GetPath().AppendChild('Tip'))
        tip.CreateAxisAttr().Set(UsdGeom.Tokens.z)
        tip.CreateRadiusAttr().Set(length * .05 / units)
        tip.CreateHeightAttr().Set(length * .2 / units)
        finish(tip, tuple(o + d * length * .9 for o, d in zip(origin, direction)), rotation, color)

    arrow('X_Positive', (1., 0., 0.), (0., 90., 0.), 'Red')
    arrow('Y_Positive', (0., 1., 0.), (-90., 0., 0.), 'Green')
    arrow('Z_Positive', (0., 0., 1.), (0., 0., 0.), 'Blue')
    # Offset from the optical axis so the direction arrow is beside the lens.
    arrow('OpticalDirection_MinusZ', (0., 0., -1.), (0., 180., 0.), 'Yellow',
          (max(.06, length * 1.2), 0., 0.))
    return path
