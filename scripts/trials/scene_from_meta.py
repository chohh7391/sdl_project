#!/usr/bin/env python3
"""Write the MuJoCo scene a recorded trajectory assumes, from its meta JSON.

``export_trajectory.py`` records the layout a trial was planned against, because
the replay is position-controlled and senses nothing: put the vessels somewhere
else and the pour misses. This turns that recorded layout into an FR5 MuJoCo
scene, so a replay can be watched against the geometry it was planned for
rather than against the one hard-coded cell.

    scripts/trials/scene_from_meta.py \
        _2026__IEEE_Access/revision/analysis/data/trajectory/transfer_seed5_meta.json \
        --output ~/ros2_ws/src/cho_robot_project/cho_description_fr5/xml/scene_ag95_sdl_seed5.xml

The output has to land in ``cho_description_fr5/xml/`` because that is the only
directory the bringup will look in: ``fr5.ros2_control.xacro`` builds the model
path as ``$(find cho_description_fr5)/xml/${mujoco_scene}``.

NO APRILTAGS. ``scene_ag95_sdl.xml`` is the perception cell and carries two tag
plates at a fixed +x offset; the Isaac cell this meta came from instead picks
each tag's mounting ANGLE per seed, so that every plate clears the other objects
(``isaacsim/scripts/standalone/task.py``, ``_clear_tag_angle``). Reproducing a
seed's tag layout means running that solver, and a plate written at a guessed
angle would be a perception cell that exists nowhere. A blind replay does not
need one, so this scene has vessels and obstacles only. For perception work, use
``scene_ag95_sdl.xml``.
"""

import argparse
import json
import math
import os
import sys

#: Footprint and height [m] per entity, from ``TAMP/tamp/src/envs/utils.py``
#: ENTITIES, which is what the planner collision-checks against. Written down
#: rather than imported because that module pulls in curobo/cutamp, and a scene
#: generator should not need a GPU stack. Keep in step with ENTITIES.
#:
#: ``shape`` is how the entity is DRAWN here. The planner models every one of
#: them as a box of the dims below; a cylinder is drawn where the object is
#: round, exactly as ``scene_ag95_sdl.xml`` draws its two vessels.
ENTITY_DIMS = {
    'beaker':  {'dims': [0.05, 0.05, 0.135],  'shape': 'cylinder'},
    'flask':   {'dims': [0.07, 0.07, 0.12],   'shape': 'cylinder'},
    'magnet':  {'dims': [0.010, 0.010, 0.035], 'shape': 'cylinder'},
    'box':     {'dims': [0.108, 0.108, 0.08], 'shape': 'box'},
    'stirrer': {'dims': [0.18, 0.18, 0.09],   'shape': 'box'},
}

#: Drawn colour per entity. Glassware for the vessels, so the two that matter
#: read the same as they do in the perception cell.
ENTITY_MATERIAL = {
    'beaker': 'glassware',
    'flask': 'glassware',
    'magnet': 'hardware',
    'box': 'hardware',
    'stirrer': 'hardware',
}

#: Half the table's edge [m]. ``scene_ag95_sdl.xml``: 1.5 m square, top face at
#: z = 0, robot base at the origin.
TABLE_HALF_M = 0.75


def yaw_quat(yaw_deg):
    """MuJoCo ``quat`` (w x y z) for a rotation of *yaw_deg* about +z."""
    half = math.radians(yaw_deg) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def geom_for(name, entry):
    """One ``<geom>`` line for *name* at the meta's pose, resting on the table.

    The centre height is half the entity's own height, which is what "standing
    on the table top" means for every one of these and what
    ``mujoco_cell_static_poses.yaml`` already records for the magnet (0.0175)
    and the stirrer (0.045).
    """
    spec = ENTITY_DIMS[name]
    dims = spec['dims']
    x, y = entry['xy']
    yaw_deg = float(entry.get('yaw_deg', 0.0))
    centre_z = dims[2] / 2.0

    if spec['shape'] == 'cylinder':
        # MuJoCo cylinder size is (radius, half-height); the footprint above is
        # a diameter, so the radius is half of it.
        size = '%.6g %.6g' % (dims[0] / 2.0, dims[2] / 2.0)
    else:
        size = '%.6g %.6g %.6g' % (dims[0] / 2.0, dims[1] / 2.0, dims[2] / 2.0)

    quat = yaw_quat(yaw_deg)
    return (
        '    <geom name="%s" type="%s" pos="%.6g %.6g %.6g"\n'
        '          quat="%.9g %.9g %.9g %.9g" size="%s" material="%s"/>'
        % (name, spec['shape'], x, y, centre_z, quat[0], quat[1], quat[2], quat[3],
           size, ENTITY_MATERIAL[name])
    )


def check_layout(layout):
    """Complain about anything the scene cannot honestly draw."""
    problems = []
    for name, entry in layout.items():
        if name not in ENTITY_DIMS:
            problems.append(
                '%s has no entry in ENTITY_DIMS; add its ENTITIES dims before '
                'drawing it' % name)
            continue
        x, y = entry['xy']
        if abs(x) > TABLE_HALF_M or abs(y) > TABLE_HALF_M:
            problems.append(
                '%s at (%.4f, %.4f) is off the %.2f m table' % (name, x, y, 2 * TABLE_HALF_M))
    return problems


def render(meta, meta_path):
    layout = meta['layout_the_trajectory_assumes']
    problems = check_layout(layout)
    if problems:
        raise SystemExit('refusing to write a scene:\n  ' + '\n  '.join(problems))

    geoms = '\n'.join(geom_for(name, layout[name]) for name in sorted(layout))
    listed = '\n'.join(
        '         %-8s (%8.4f, %8.4f)  yaw %7.1f deg' % (
            name, layout[name]['xy'][0], layout[name]['xy'][1],
            layout[name].get('yaw_deg', 0.0))
        for name in sorted(layout))

    return '''<mujoco model="fr5 sdl cell {seed}">
  <!-- GENERATED by TAMP/tamp/scripts/trials/scene_from_meta.py from
       {meta_name}
       Do not hand-edit: regenerate it from the meta instead.

       The layout ONE RECORDED TRIAL was planned against (seed {seed}, tool
       {tool}, recorded {recorded}). The replay of that trial is
       position-controlled and senses nothing, so the vessels have to be here or
       the pour misses - which is the whole reason this scene exists rather than
       the trial being watched against the cell in scene_ag95_sdl.xml, whose
       layout is a different one:

{listed}

       Heights are each entity's own, resting on the table top at z = 0, and the
       dims are ENTITIES' (TAMP/tamp/src/envs/utils.py) - what the planner
       collision-checked this trajectory against.

       NO APRILTAGS, deliberately. See the generator's docstring: the Isaac cell
       picks each tag's mounting angle per seed, so a plate drawn here at a
       guessed angle would describe a cell that does not exist. Use
       scene_ag95_sdl.xml for anything that puts perception in the loop.

       THE OBJECTS ARE STATIC, for the same two reasons scene_ag95_sdl.xml gives:
       free joints would change nq and invalidate the home/home1 keyframes the
       bringup selects by name, and every geom in fr5_ag95.xml is
       contype/conaffinity 0, so nothing could be grasped here anyway. This
       scene shows a recorded trajectory against the geometry it assumed. It
       does not test grasping. -->

  <include file="fr5_ag95.xml"/>
  <include file="scene_common.xml"/>

  <asset>
    <material name="bench" rgba="0.76 0.76 0.78 1" specular="0.1" shininess="0.1" reflectance="0"/>
    <material name="glassware" rgba="0.82 0.88 0.92 1" specular="0.3" shininess="0.3" reflectance="0"/>
    <material name="hardware" rgba="0.35 0.37 0.40 1" specular="0.1" shininess="0.1" reflectance="0"/>
  </asset>

  <worldbody>
    <!-- Table. Top face at z = 0, {edge} m square, as scene_ag95_sdl.xml. -->
    <geom name="bench" type="box" pos="0 0 -0.01" size="{half} {half} 0.01" material="bench"/>

{geoms}
  </worldbody>
</mujoco>
'''.format(
        seed=meta.get('seed', '?'),
        tool=meta.get('tool', '?'),
        recorded=meta.get('recorded_utc', '?'),
        meta_name=os.path.basename(meta_path),
        listed=listed,
        geoms=geoms,
        edge='%.4g' % (2 * TABLE_HALF_M),
        half='%.4g' % TABLE_HALF_M,
    )


def render_layout(meta, meta_path):
    """The same layout as a cho_task_manager replay-gate file.

    Written by the same run that writes the scene, so the cell the gate checks
    against and the cell MuJoCo simulates cannot drift apart - which they would
    the first time one of the two was regenerated alone.

    ``yaw_deg`` is emitted only for entities whose orientation is observable. A
    cylinder has no yaw to match, and declaring one would make the gate refuse a
    cell that is in fact correct.
    """
    layout = meta['layout_the_trajectory_assumes']
    lines = [
        '# GENERATED by scripts/trials/scene_from_meta.py (sdl_project) from',
        '#   %s' % os.path.basename(meta_path),
        '# Do not hand-edit: regenerate it together with the scene it describes,',
        '#   cho_description_fr5/xml/scene_ag95_sdl_seed%s.xml' % meta.get('seed', '?'),
        '#',
        '# The cell layout the trajectory-replay gate checks a recording against.',
        '# yaw_deg is omitted for the round vessels: they have no yaw to match.',
        'layout:',
    ]
    for name in sorted(layout):
        entry = layout[name]
        lines.append('  %s:' % name)
        lines.append('    xy: [%.6g, %.6g]' % (entry['xy'][0], entry['xy'][1]))
        if ENTITY_DIMS.get(name, {}).get('shape') != 'cylinder':
            lines.append('    yaw_deg: %.6g' % float(entry.get('yaw_deg', 0.0)))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('meta', help='trajectory meta JSON written by export_trajectory.py')
    parser.add_argument('--output', required=True, help='scene XML to write')
    parser.add_argument('--layout-output', default=None,
                        help='also write the cho_task_manager replay-gate layout YAML here')
    args = parser.parse_args(argv)

    with open(args.meta) as handle:
        meta = json.load(handle)

    scene = render(meta, args.meta)
    with open(args.output, 'w') as handle:
        handle.write(scene)

    if args.layout_output:
        with open(args.layout_output, 'w') as handle:
            handle.write(render_layout(meta, args.meta))
        print('wrote %s' % args.layout_output)

    print('wrote %s' % args.output)
    for name, entry in sorted(meta['layout_the_trajectory_assumes'].items()):
        print('  %-8s (%8.4f, %8.4f) yaw %7.1f deg  %s'
              % (name, entry['xy'][0], entry['xy'][1], entry.get('yaw_deg', 0.0),
                 ENTITY_DIMS[name]['dims']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
