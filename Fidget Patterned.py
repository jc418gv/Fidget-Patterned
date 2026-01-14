"""Minimal Fusion script: create spherical bands and add random non-overlapping circular holes."""

import traceback
import adsk.core
import adsk.fusion
import math
import random

app = adsk.core.Application.get()
ui = app.userInterface

RING_SPECS_MM = [
    (55.0, 48.0),
    (46.0, 39.0),
    (37.0, 30.0),
    (28.0, 21.0),
]
SLICE_HALF_THICK_MM = 10.0

# Random-holes parameters
MIN_HOLE_D_MM = 1.5
MAX_HOLE_D_MM = 6.0
TARGET_HOLES = 20
OUTER_HOOP_WIDTH_MM = 3.5
INNER_EMBOSS_DEPTH_MM = 1.0
GAP_MM = 0.2
HOLE_SCALE = 1.6
EDGE_RIM_FRACTION = 0.18


def _cm(value_mm: float) -> float:
    return value_mm / 10.0


def _create_component(parent: adsk.fusion.Component, name: str) -> adsk.fusion.Component:
    occ = parent.occurrences.addNewComponent(adsk.core.Matrix3D.create())
    occ.component.name = name
    return occ.component


def _make_revolved_sphere(comp: adsk.fusion.Component, radius_cm: float) -> adsk.fusion.BRepBody:
    sketches = comp.sketches
    yz = comp.yZConstructionPlane
    sketch = sketches.add(yz)

    arcs = sketch.sketchCurves.sketchArcs
    lines = sketch.sketchCurves.sketchLines

    center = adsk.core.Point3D.create(0, 0, 0)
    start = adsk.core.Point3D.create(0, radius_cm, 0)
    arc = arcs.addByCenterStartSweep(center, start, -math.pi)
    dia = lines.addByTwoPoints(arc.startSketchPoint, arc.endSketchPoint)

    profile = sketch.profiles.item(0)

    revolves = comp.features.revolveFeatures
    rev_input = revolves.createInput(profile, dia, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    full_angle = adsk.core.ValueInput.createByReal(2 * math.pi)
    try:
        rev_input.setAngleExtent(False, full_angle)
    except Exception:
        try:
            rev_input.setSymmetricAngleExtent(full_angle, True)
        except Exception:
            pass
    rev_feat = revolves.add(rev_input)
    body = rev_feat.bodies.item(0)

    try:
        sketch.deleteMe()
    except Exception:
        pass

    return body


def _make_shell(comp: adsk.fusion.Component, outer_d_mm: float, inner_d_mm: float) -> adsk.fusion.BRepBody:
    outer_radius = _cm(outer_d_mm / 2.0)
    inner_radius = _cm(inner_d_mm / 2.0)

    outer_body = _make_revolved_sphere(comp, outer_radius)
    inner_body = _make_revolved_sphere(comp, inner_radius)

    combine_feats = comp.features.combineFeatures
    tools = adsk.core.ObjectCollection.create()
    tools.add(inner_body)
    combine_input = combine_feats.createInput(outer_body, tools)
    combine_input.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
    combine_feats.add(combine_input)

    return outer_body


def _slice_band(comp: adsk.fusion.Component, body: adsk.fusion.BRepBody, half_thick_mm: float) -> adsk.fusion.BRepBody:
    planes = comp.constructionPlanes
    xy = comp.xYConstructionPlane

    bb = body.boundingBox
    max_pos_z = abs(bb.maxPoint.z)
    max_neg_z = abs(bb.minPoint.z)
    safe_max = 0.95 * min(max_pos_z, max_neg_z)
    desired = _cm(half_thick_mm)
    offset = min(desired, safe_max) if safe_max > 0 else desired

    split_feats = comp.features.splitBodyFeatures

    def make_offset_plane(distance_cm: float):
        p_input = planes.createInput()
        p_input.setByOffset(xy, adsk.core.ValueInput.createByReal(distance_cm))
        return planes.add(p_input)

    def try_split(target_body, plane):
        inp = split_feats.createInput(target_body, plane, True)
        return split_feats.add(inp)

    attempts = 0
    top_feature = None
    top_plane = None
    while attempts < 5 and top_feature is None:
        try:
            if top_plane:
                try:
                    top_plane.deleteMe()
                except Exception:
                    pass
            top_plane = make_offset_plane(offset)
            top_feature = try_split(body, top_plane)
        except Exception:
            top_feature = None
            offset *= 0.9
            attempts += 1

    if top_feature is None:
        try:
            if top_plane:
                try:
                    top_plane.deleteMe()
                except Exception:
                    pass
            top_plane = make_offset_plane(0.0)
            top_feature = try_split(body, top_plane)
        except Exception:
            raise

    top_bodies = [top_feature.bodies.item(i) for i in range(top_feature.bodies.count)]

    def center_z(brep: adsk.fusion.BRepBody) -> float:
        bb2 = brep.boundingBox
        return 0.5 * (bb2.minPoint.z + bb2.maxPoint.z)

    bot_z = -abs(offset)
    target = None
    for cand in top_bodies:
        bb2 = cand.boundingBox
        if bb2.minPoint.z <= bot_z <= bb2.maxPoint.z:
            target = cand
            break
    if target is None:
        target = min(top_bodies, key=lambda b: abs(center_z(b)))

    bottom_plane = make_offset_plane(-abs(offset))
    bottom_feature = try_split(target, bottom_plane)
    result_bodies = [bottom_feature.bodies.item(i) for i in range(bottom_feature.bodies.count)]

    try:
        top_plane.deleteMe()
    except Exception:
        pass
    try:
        bottom_plane.deleteMe()
    except Exception:
        pass

    band = min(result_bodies, key=lambda b: abs(center_z(b)))

    all_bodies = [comp.bRepBodies.item(i) for i in range(comp.bRepBodies.count)]
    for b in all_bodies:
        try:
            b.isVisible = (b == band)
        except Exception:
            pass

    return band


def _outer_face(body: adsk.fusion.BRepBody):
    best = None
    best_area = -1.0
    for face in body.faces:
        try:
            geom = face.geometry
            if hasattr(geom, 'surfaceType') and geom.surfaceType == adsk.core.SurfaceTypes.SphereSurfaceType:
                area = face.area
                if area > best_area:
                    best_area = area
                    best = face
        except Exception:
            continue
    return best


def _emboss_profiles_from_sketch(comp: adsk.fusion.Component, sketch: adsk.fusion.Sketch, face: adsk.fusion.BRepFace, depth_mm: float) -> bool:
    emboss_feats = getattr(comp.features, 'embossFeatures', None)
    if emboss_feats is None:
        return False

    profiles = adsk.core.ObjectCollection.create()
    for prof in sketch.profiles:
        try:
            profiles.add(prof)
        except Exception:
            pass
    if profiles.count == 0:
        return False

    input_types = getattr(adsk.fusion, 'EmbossFeaturesInputTypes', None)
    if input_types is None:
        return False
    deboss_type = getattr(input_types, 'DebossInput', None)
    if deboss_type is None:
        return False

    emboss_input = emboss_feats.createInput(profiles, face, deboss_type)
    depth_mag = adsk.core.ValueInput.createByString(f'{abs(depth_mm)} mm')
    if hasattr(emboss_input, 'depth'):
        emboss_input.depth = depth_mag
    else:
        try:
            emboss_input.setDepth(depth_mag)
        except Exception:
            pass

    try:
        emboss_feats.add(emboss_input)
        return True
    except Exception:
        return False





# The flat-XY hex generator was removed — radial polygons are generated by
# `_generate_hex_polygons` and are cut radially by `_cut_polygons_radially`.



def _generate_hex_polygons(outer_d_mm: float, inner_d_mm: float, outer_hoop_mm: float,
                           target_count: int = TARGET_HOLES):
    """Generate a list of hex polygons (list of vertex (x,y) tuples in mm) covering the annulus.
    This only computes polygon vertex coordinates centered at origin; polygons are filtered per-ring
    when cutting radially."""
    outer_r_mm = outer_d_mm / 2.0
    inner_r_mm = inner_d_mm / 2.0
    outer_limit_mm = outer_r_mm - outer_hoop_mm
    if outer_limit_mm <= inner_r_mm + 1.0:
        return []

    annulus_area = math.pi * (outer_limit_mm ** 2 - inner_r_mm ** 2)
    if annulus_area <= 0:
        return []

    area_per_hex = (3.0 * math.sqrt(3.0) / 2.0)
    est_side = math.sqrt((annulus_area / max(1.0, target_count)) / area_per_hex)
    a = max(MIN_HOLE_D_MM, min(MAX_HOLE_D_MM, est_side * HOLE_SCALE))

    x_spacing = 1.5 * a
    y_spacing = math.sqrt(3.0) * a
    half_y = (math.sqrt(3.0) / 2.0) * a

    polygons = []
    x_min = -outer_limit_mm
    x_max = outer_limit_mm
    i_min = int(math.floor(x_min / x_spacing)) - 1
    i_max = int(math.ceil(x_max / x_spacing)) + 1
    placed = 0
    for i in range(i_min, i_max + 1):
        x = i * x_spacing
        y_offset = 0.0 if (i % 2) == 0 else (y_spacing / 2.0)
        y_min = -outer_limit_mm - y_spacing
        y_max = outer_limit_mm + y_spacing
        j_min = int(math.floor((y_min - y_offset) / y_spacing)) - 1
        j_max = int(math.ceil((y_max - y_offset) / y_spacing)) + 1
        for j in range(j_min, j_max + 1):
            y = j * y_spacing + y_offset
            dist = math.hypot(x, y)
            if dist + a + GAP_MM > outer_limit_mm:
                continue
            if dist - a - GAP_MM < inner_r_mm:
                continue

            pts = []
            pts.append((x + a, y + 0.0))
            pts.append((x + a * 0.5, y + half_y))
            pts.append((x - a * 0.5, y + half_y))
            pts.append((x - a, y + 0.0))
            pts.append((x - a * 0.5, y - half_y))
            pts.append((x + a * 0.5, y - half_y))

            polygons.append(pts)
            placed += 1
            if placed >= target_count:
                break
        if placed >= target_count:
            break

    return polygons


def _create_flat_polygons_sketch(comp: adsk.fusion.Component, polygons, outer_d_mm: float, inner_d_mm: float, rotation_deg: float = 0.0) -> adsk.fusion.Sketch:
    """Create a sketch on the component XY plane and add provided polygon vertex lists (in mm),
    optionally rotated. Only polygons whose centroids fall inside the supplied annulus are added.
    Returns the sketch or None if nothing was added."""
    try:
        sketch = comp.sketches.add(comp.xYConstructionPlane)
        sketch.name = 'HexPattern_Flat'
    except Exception:
        return None

    rot = math.radians(rotation_deg)
    cosr = math.cos(rot)
    sinr = math.sin(rot)

    outer_limit = outer_d_mm / 2.0 - OUTER_HOOP_WIDTH_MM
    inner_r = inner_d_mm / 2.0
    added = 0

    for poly in polygons:
        # compute centroid
        cx = sum([p[0] for p in poly]) / len(poly)
        cy = sum([p[1] for p in poly]) / len(poly)
        r = math.hypot(cx, cy)
        if r + 0.1 > outer_limit or r - 0.1 < inner_r:
            continue

        try:
            pts = []
            for (x, y) in poly:
                rx = x * cosr - y * sinr
                ry = x * sinr + y * cosr
                pts.append(adsk.core.Point3D.create(_cm(rx), _cm(ry), 0))

            prev = pts[0]
            for p in pts[1:]:
                sketch.sketchCurves.sketchLines.addByTwoPoints(prev, p)
                prev = p
            sketch.sketchCurves.sketchLines.addByTwoPoints(prev, pts[0])
            added += 1
        except Exception:
            continue

    if added == 0:
        try:
            sketch.deleteMe()
        except Exception:
            pass
        return None

    return sketch


def _cut_polygons_radially(comp: adsk.fusion.Component, polygons, outer_d_mm: float, inner_d_mm: float, band: adsk.fusion.BRepBody, max_polygons: int = 200) -> bool:
    """For each polygon whose centroid lies within the ring annulus, create a flat sketch on XY,
    extrude a prism as a new body, then use Combine Cut with the band. Returns True if at least
    one cut was made."""
    extrudes = comp.features.extrudeFeatures
    combine_feats = comp.features.combineFeatures

    outer_limit = outer_d_mm / 2.0 - OUTER_HOOP_WIDTH_MM
    inner_r = inner_d_mm / 2.0

    cuts_made = 0
    for poly in polygons:
        if cuts_made >= max_polygons:
            break

        # centroid
        cx = sum([p[0] for p in poly]) / len(poly)
        cy = sum([p[1] for p in poly]) / len(poly)
        r = math.hypot(cx, cy)
        if r + 0.1 > outer_limit or r - 0.1 < inner_r or r < 1e-6:
            continue

        try:
            # create flat sketch and draw polygon (in XY)
            sketch = comp.sketches.add(comp.xYConstructionPlane)
            sketch.name = f'PolyTool_{int(cx)}_{int(cy)}'
            pts3d = [adsk.core.Point3D.create(_cm(x), _cm(y), 0) for (x, y) in poly]
            prev = pts3d[0]
            for p in pts3d[1:]:
                sketch.sketchCurves.sketchLines.addByTwoPoints(prev, p)
                prev = p
            sketch.sketchCurves.sketchLines.addByTwoPoints(prev, pts3d[0])

            profiles = adsk.core.ObjectCollection.create()
            for prof in sketch.profiles:
                try:
                    profiles.add(prof)
                except Exception:
                    pass
            if profiles.count == 0:
                try:
                    sketch.deleteMe()
                except Exception:
                    pass
                continue

            # extrude as a new body (tool)
            cut_distance_cm = 6.0 * _cm(outer_d_mm / 2.0)
            ext_input = extrudes.createInput(profiles, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            distance = adsk.core.ValueInput.createByReal(cut_distance_cm)
            try:
                ext_input.setDistanceExtent(True, distance)
            except Exception:
                ext_input.setDistanceExtent(False, distance)
            extrudes.add(ext_input)

            # the new tool body will be the last body added to the component
            tool_body = comp.bRepBodies.item(comp.bRepBodies.count - 1)
            tools = adsk.core.ObjectCollection.create()
            tools.add(tool_body)

            # combine (cut) the tool from the band
            combine_input = combine_feats.createInput(band, tools)
            combine_input.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
            combine_input.isKeepToolBodies = False
            combine_feats.add(combine_input)

            cuts_made += 1
            try:
                sketch.deleteMe()
            except Exception:
                pass
        except Exception:
            try:
                sketch.deleteMe()
            except Exception:
                pass
            continue

    return cuts_made > 0


def _cut_centered_pyramids(comp: adsk.fusion.Component, band: adsk.fusion.BRepBody,
                           outer_mm: float, count: int, band_thickness_mm: float) -> bool:
    """Cut alternating pyramids with apex at the origin and bases on the XY plane."""

    sketches = comp.sketches
    extrudes = comp.features.extrudeFeatures
    combine_feats = comp.features.combineFeatures
    xy_plane = comp.xYConstructionPlane

    if count <= 0:
        return False

    outer_radius = outer_mm / 2.0
    base_radius_mm = max(outer_radius + 2.0, 60.0)

    effective_thickness = max(band_thickness_mm, 2.0 * SLICE_HALF_THICK_MM)
    half_thickness = 0.5 * effective_thickness
    rim_mm = min(half_thickness - 0.25, EDGE_RIM_FRACTION * effective_thickness)
    if rim_mm < 0:
        rim_mm = 0.0
    core_half_mm = max(0.25, half_thickness - rim_mm)
    if core_half_mm <= 1e-3:
        return False

    extent = adsk.core.ValueInput.createByReal(_cm(core_half_mm))

    sector = 2.0 * math.pi / float(count)
    half_angle = 0.45 * sector

    cuts = 0
    for i in range(count):
        # Alternate: cut every other wedge so triangles appear point-up/point-down along the ring
        if (i % 2) != 0:
            continue

        center_angle = i * sector
        a1 = center_angle - half_angle
        a2 = center_angle + half_angle

        apex = adsk.core.Point3D.create(0, 0, 0)
        base1 = adsk.core.Point3D.create(_cm(base_radius_mm * math.cos(a1)),
                                         _cm(base_radius_mm * math.sin(a1)), 0)
        base2 = adsk.core.Point3D.create(_cm(base_radius_mm * math.cos(a2)),
                                         _cm(base_radius_mm * math.sin(a2)), 0)

        sketch = None
        try:
            sketch = sketches.add(xy_plane)
            sketch.name = f'CenterPyramid_{i}'
            lines = sketch.sketchCurves.sketchLines
            lines.addByTwoPoints(apex, base1)
            lines.addByTwoPoints(base1, base2)
            lines.addByTwoPoints(base2, apex)

            profiles = adsk.core.ObjectCollection.create()
            for prof in sketch.profiles:
                try:
                    profiles.add(prof)
                except Exception:
                    pass
            if profiles.count == 0:
                continue

            ext_input = extrudes.createInput(profiles, adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            try:
                ext_input.setSymmetricExtent(extent, True)
            except Exception:
                ext_input.setDistanceExtent(False, extent)
            extrude = extrudes.add(ext_input)

            tool_body = extrude.bodies.item(0)
            tools = adsk.core.ObjectCollection.create()
            tools.add(tool_body)

            combine_input = combine_feats.createInput(band, tools)
            combine_input.operation = adsk.fusion.FeatureOperations.CutFeatureOperation
            combine_input.isKeepToolBodies = False
            combine_feats.add(combine_input)

            cuts += 1
        finally:
            if sketch:
                try:
                    sketch.deleteMe()
                except Exception:
                    pass

    return cuts > 0


    






def _cut_profiles_through_band(comp: adsk.fusion.Component, sketch: adsk.fusion.Sketch, outer_d_mm: float) -> bool:
    profiles = adsk.core.ObjectCollection.create()
    for prof in sketch.profiles:
        try:
            profiles.add(prof)
        except Exception:
            pass
    if profiles.count == 0:
        return False

    extrudes = comp.features.extrudeFeatures
    cut_distance_cm = 2.0 * _cm(outer_d_mm / 2.0) + 1.0
    try:
        ext_input = extrudes.createInput(profiles, adsk.fusion.FeatureOperations.CutFeatureOperation)
        distance = adsk.core.ValueInput.createByReal(cut_distance_cm)
        ext_input.setDistanceExtent(False, distance)
        ext_input.isSolid = True
        extrudes.add(ext_input)
        return True
    except Exception:
        return False


def _make_band(comp: adsk.fusion.Component, outer_d_mm: float, inner_d_mm: float, make_solid: bool = False) -> adsk.fusion.BRepBody:
    base_body = _make_revolved_sphere(comp, _cm(outer_d_mm / 2.0)) if make_solid else _make_shell(comp, outer_d_mm, inner_d_mm)
    return _slice_band(comp, base_body, SLICE_HALF_THICK_MM)


def run(_context: str):
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox('No active Fusion design.')
            return

        root = design.rootComponent
        ring_count = len(RING_SPECS_MM)

        # Precompute a single hex pattern (polygons in mm) using the outermost ring extents.
        outermost_outer_mm, _ = RING_SPECS_MM[0]
        innermost_inner_mm = RING_SPECS_MM[-1][1]
        pattern_polygons = _generate_hex_polygons(outermost_outer_mm, innermost_inner_mm, OUTER_HOOP_WIDTH_MM, target_count=TARGET_HOLES * 2)
        # Choose number of radial rotations to replicate the pattern (creates denser lattice)
        ROTATIONS = 6

        for index in range(ring_count):
            outer_mm, inner_mm = RING_SPECS_MM[index]

            comp = _create_component(root, f'Ring {index + 1}')
            make_solid = (index == (ring_count - 1))
            band = _make_band(comp, outer_mm, inner_mm, make_solid=make_solid)
            face = _outer_face(band)
            if not face:
                continue

            # Perform radial cuts for polygons whose centroids lie inside this ring's annulus.
            try:
                bb = band.boundingBox
                band_thickness_mm = 10.0 * (bb.maxPoint.z - bb.minPoint.z)
            except Exception:
                band_thickness_mm = SLICE_HALF_THICK_MM * 2.0

            # Skip pattern application for the innermost ring (keep the core solid)
            if index == (ring_count - 1):
                continue

            # For rings 1-3 (outermost three), apply center-based pyramids to remove large wedges quickly
            if index <= 2:
                counts = [24, 18, 12]
                count = counts[index] if index < len(counts) else 12
                _cut_centered_pyramids(comp, band, outer_mm, count, band_thickness_mm)
            else:
                # Maximum polygons per ring (safeguard for performance)
                MAX_POLYS_PER_RING = 120
                ok = _cut_polygons_radially(comp, pattern_polygons, outer_mm, inner_mm, band, max_polygons=MAX_POLYS_PER_RING)
                if not ok:
                    # No cuts were applied for this ring; fall back to embossing a flat pattern onto the spherical face
                    depth_mm = 0.75 * band_thickness_mm
                    flat_sk = _create_flat_polygons_sketch(comp, pattern_polygons, outer_mm, inner_mm, rotation_deg=0.0)
                    if flat_sk is not None:
                        _emboss_profiles_from_sketch(comp, flat_sk, face, depth_mm)
                    else:
                        # nothing to apply for this ring
                        pass


        ui.messageBox('Fidget rings created.')
    except Exception:
        if ui:
            ui.messageBox(f'Failed:\n{traceback.format_exc()}')
