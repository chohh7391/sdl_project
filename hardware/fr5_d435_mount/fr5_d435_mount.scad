// FR5 wrist mount for an Intel RealSense D435 (eye-in-hand), 3D printable.
//
// The rig already has a metal adapter plate between the FR5 flange and the
// AG-95. This part goes UNDER that adapter: flange -> this ring -> the existing
// adapter -> gripper, all held by the same four flange screws (longer by
// `ring_t`). The camera arm lives in the `ring_t` gap and reaches out past the
// gripper, then rises once it is clear of the adapter's footprint.
//
// THICKNESS IS THE WHOLE GAME. Everything this ring adds under the adapter is
// added to the TCP, so the clamped part is kept at `ring_t` = 3 mm and the
// stiffness is bought back outboard of `clamp_r`, where there is free air.
//
// DIMENSION PROVENANCE -- read README.md before printing.
//   measured  flange bore / OD, from meshes/robot/fr5/wrist3_link.STL
//   measured  gripper envelope, from meshes/gripper/ag95/base_link.STL
//   standard  Oe50 bolt circle, 4x M6 (ISO 9409-1-50-4-M6)
//   datasheet D435 body 90 x 25 x 25 mm, 1/4"-20 UNC on the bottom face
//   UNKNOWN   everything about the existing adapter plate -> `clamp_r`, the
//             wing notch, and the bolt clocking. Print gauge_flange first: it
//             is the real stack-up test, at the real thickness.

// Which body to render. Export one STL per value.
part = "mount";          // "mount" | "gauge_flange" | "gauge_camera"

$fn = 96;

/* ------------------------------------------------------------------ *
 *  Robot interface -- FR5 tool flange, ISO 9409-1-50-4-M6
 * ------------------------------------------------------------------ */
bolt_bcd      = 50;      // bolt circle diameter
bolt_n        = 4;
bolt_hole_d   = 6.6;     // M6 clearance
bolt_clock    = 45;      // azimuth of the first screw, measured from the arm.
                         // 45 or 0: pick whichever puts the arm where you want
                         // it, since four screws only allow 90 deg steps.
pin_clock     = [0, 90, 180, 270];  // clearance for a locating pin, wherever
pin_hole_d    = 6.6;                // the adapter puts one (cheap insurance)

/* ------------------------------------------------------------------ *
 *  Sandwich ring -- this is the TCP shift, keep it small
 * ------------------------------------------------------------------ */
ring_t    = 4;           // ADDS EXACTLY THIS MUCH TO THE TCP. 4 rather than 3
                         // because the camera now sits on a 130 mm stalk: at
                         // 3 mm this span bends 0.5 deg under its own weight
                         // when the wrist is horizontal (3.5 mm of pointing
                         // error at 0.4 m), at 4 mm it is 0.14 deg / 1.0 mm.
ring_od   = 62;          // flange OD is 63 (measured); held 1 mm under it so
                         // the printed edge cannot stand proud of the flange
ring_bore = 32;          // clears the flange bore (Oe31.5). If the adapter has a
                         // bigger spigot on its underside, open this up -- but
                         // not past 42, where it would meet the bolt holes.

/* ------------------------------------------------------------------ *
 *  Where the sandwich ends and the arm may grow
 * ------------------------------------------------------------------ */
clamp_x = 50;            // How far the adapter's wings reach along the arm's
                         // direction (measured on the rig: r = 93/2 = 46.5),
                         // plus 3.5 mm. Inboard of it the arm stays `ring_t`
                         // thin; outboard it ramps up. A half-space, not a
                         // radius: these obstacles are rectangles whose corners
                         // sit further out than their faces.

// The adapter's wings, measured on the rig. They stand 20 mm proud of the
// adapter's underside, on the gripper's narrow axis -- which is also the axis
// the camera wants. The arm goes under them (it lives in the ring_t gap) and
// the camera goes over them, which is what sets cam_z below.
wing_r = 46.5;
wing_h = 20;
ramp_h  = 10;            // arm thickness once it is outboard of clamp_x
arm_w   = 90;            // width of the flat arm where it crosses the
                         // clamped span -- this is what carries the
                         // stalk's moment, so it is wide on purpose
tower_w = [66, 44];      // stalk width at the ramp, and at the plate
tower_t = 9;             // stalk thickness at the ramp (it fans out to meet
                         // the plate's own footprint at the top)
window_d = 24;           // lightening window through the stalk, 0 = solid. It
                         // is centred, so it can only ever eat into the middle
                         // and never opens the side rails.

// If the adapter's wings hang below its mounting face AND you have to put the
// arm on the same azimuth, cut a channel so they pass. Off by default: putting
// the arm 90 deg away from the wings is always the better answer.
wing_notch   = false;
wing_notch_w  = 34;      // channel width
wing_notch_r  = [34, 56];// from this radius to that one

/* ------------------------------------------------------------------ *
 *  Camera -- Intel RealSense D435
 * ------------------------------------------------------------------ */
cam_w  = 90;             // long axis (stereo baseline direction)
cam_d  = 25;             // front (lens) to back (USB-C)
cam_h  = 25;             // tripod face to top
cam_fit = 0.8;           // total slack between the two end walls

// THE IMAGERS ARE NOT ON THE BODY'S CENTRELINE. realsense2_description
// (_d435.urdf.xacro) puts the depth / infra1 origin 17.5 mm off the body centre
// along the width, and the colour imager 32.5 mm (infra2 sits at -32.5). Centre
// the *body* and the picture comes out shifted sideways by that much. So the
// cradle is offset by this amount instead, which puts the chosen imager in the
// tool's symmetry plane. Set it to the stream the tag detector actually runs on.
cam_lens_dy = 32.5;      // 32.5 = colour, 17.5 = depth / infra1 (IR).
                         // 32.5 is what the reference Panda mount uses: its
                         // camera plate is centred 32.87 mm off the flange axis
                         // (plate -62..-4, body -78..+12, 16 mm of overhang each
                         // side; its two M3 holes sit 44.6 mm apart, symmetric
                         // about that centre), which lands the colour imager on
                         // the axis to within half a millimetre.
tripod_d        = 6.8;   // 1/4"-20 close clearance (screw major dia 6.35)
tripod_travel   = 0;     // 0 = a plain hole. Any slot here is play the camera
                         // can drift in, so the pose is fixed by the print.
tripod_cb_d     = 13;    // head counterbore
tripod_cb_depth = 3;
tripod_grip     = 4;     // MATERIAL LEFT UNDER THE SCREW HEAD. The screw reaches
                         // (screw length under head - tripod_grip) into the
                         // camera, so raise this if the screw bottoms out in the
                         // camera's thread before the head clamps, and lower it
                         // if there is too little thread engagement left.

/* ------------------------------------------------------------------ *
 *  Pose of the camera in the tool frame (+Z = tool axis, +X = arm)
 * ------------------------------------------------------------------ */
tilt      = 8;           // inward tilt of the optical axis, degrees. With
                         // cam_z below this puts the tool axis within a few
                         // percent of the image centre over 350-450 mm.
cam_clear = 38;          // tool axis -> nearest camera face. On the gripper's
                         // narrow axis: 38 = 33.5 + 4.5 of clearance. Only the
                         // wings are wider than that on this axis, and the
                         // camera clears them by sitting above wing_h.
cam_z     = 120;         // camera centre height above the flange face. This is
                         // the number that decides how much gripper ends up in
                         // frame: at 44 the AG-95 ate the top 56 % of the
                         // image (measured against a real wrist frame), at 130
                         // it nicks the top 4 % and the tool axis sits at the
                         // centre. The camera still stops short of the
                         // fingertips (~183), so it never becomes the closest
                         // thing to the bench.

/* ------------------------------------------------------------------ *
 *  Cradle
 * ------------------------------------------------------------------ */
plate_t    = tripod_grip + tripod_cb_depth;   // 7: the head still sinks into the
                         // counterbore, the plate is simply thick enough that
                         // tripod_grip is left under it
plate_back = 4;          // how far the plate reaches behind the camera
wall_t     = 4;          // end wall thickness
wall_h     = 10;         // how far the end walls stand proud of the plate
wall_len   = 12.5;       // full-height length of each end wall, measured back
                         // from the lens face. Behind that it ramps away at
                         // 45 deg and is gone before the camera's back face, so
                         // nothing stands beside the USB-C connector.
tie_slot   = [3, 9];     // cable tie slot, width x length
label      = true;

/* ------------------------------------------------------------------ *
 *  Derived
 * ------------------------------------------------------------------ */
s = sin(tilt);
c = cos(tilt);

plate_w = cam_w + cam_fit + 2 * wall_t;
wall_x  = (cam_w + cam_fit + wall_t) / 2;

cam_x = cam_clear + (cam_h / 2) * c;   // camera centre, distance from tool axis

// Camera local axes (u,v,n) -> tool frame. u = width, v = optical axis,
// n = height (+n points at the tool axis, so the tripod face looks outward
// and the fixing screw stays reachable with the gripper fitted).
cam_M = [[  0, -s, -c, cam_x],
         [ -1,  0,  0, 0    ],
         [  0,  c, -s, cam_z],
         [  0,  0,  0, 1    ]];

function w(p) = [ cam_x - s * p[1] - c * p[2],
                  -p[0],
                  cam_z + c * p[1] - s * p[2] ];

py0 = -(cam_d / 2 + plate_back);   // plate back edge, local v
py1 =   cam_d / 2;                 // plate front edge: flush with the lens face,
                                   // so nothing of the mount enters the FOV
pz  = -cam_h / 2;                  // plate inner surface, local n

p_back_in  = w([0, py0, pz]);            // plate back edge, camera side
p_back_out = w([0, py0, pz - plate_t]);  // plate back edge, outboard side

tip_x   = (p_back_in[0] - 4 + p_back_out[0]) / 2;
tip_len = p_back_out[0] - p_back_in[0] + 4;

/* ------------------------------------------------------------------ *
 *  Bodies
 * ------------------------------------------------------------------ */

// The sandwich itself: ring plus the flat arm that crosses the clamped zone.
// hull() against the ring blends the arm in on the ring's tangents instead of
// meeting it in a notch.
module ring_and_arm() {
    hull() {
        cylinder(d = ring_od, h = ring_t);
        translate([tip_x, 0, ring_t / 2])
            cube([tip_len, arm_w, ring_t], center = true);
    }
}

// Outboard of clamp_r there is nothing above the arm until the gripper, so the
// arm thickens into a ramp. This is where the bending stiffness comes from --
// the clamped span stays thin on purpose.
module arm_ramp() {
    difference() {
        hull() {
            cylinder(d = ring_od, h = ring_t);
            translate([tip_x, 0, ramp_h / 2])
                cube([tip_len, arm_w, ramp_h], center = true);
        }
        translate([clamp_x - 500, -500, -1]) cube([500, 1000, ramp_h + 2]);
    }
}

// The stalk: from the ramp up to the underside of the backing plate, tapering
// as it goes. Its outboard face is vertical, which keeps this the widest point
// of the part, and it stays outboard of both the wings (46.5) and the gripper
// (33.5) the whole way up, so it needs no steps. The window is there to keep
// the mass down; it is centred so it cannot open the side edges.
module tower() {
    difference() {
        hull() {
            translate([tip_x, 0, ramp_h / 2])
                cube([tip_len, tower_w[0], ramp_h], center = true);
            multmatrix(cam_M)
                translate([cam_lens_dy, py0 + 5, pz - plate_t / 2])
                    cube([tower_w[1], 10, plate_t], center = true);
        }
        if (window_d > 0)
            hull() for (zz = [ramp_h + 26, cam_z - 48])
                translate([tip_x - 30, 0, zz])
                    rotate([0, 90, 0]) cylinder(d = window_d, h = 80);
    }
}

// Backing plate + the two end walls that stop the camera yawing, in camera
// local coordinates so the flat gauge can reuse them.
module cradle_local(thickness = plate_t, walls = true) {
    translate([cam_lens_dy, 0, 0]) union() {
        translate([0, (py0 + py1) / 2, pz - thickness / 2])
            cube([plate_w, py1 - py0, thickness], center = true);
        if (walls)
            for (sx = [-1, 1])
                // Full height over `wall_len`, then a 45 deg ramp back down to
                // the plate. Both are clear of the camera's back face.
                hull() {
                    translate([sx * wall_x, py1 - wall_len - wall_h + 0.005, pz + 0.005])
                        cube([wall_t, 0.01, 0.01], center = true);
                    translate([sx * wall_x, py1 - wall_len / 2, pz + wall_h / 2])
                        cube([wall_t, wall_len, wall_h], center = true);
                }
    }
}

// 1/4"-20 through slot with a counterbore, cut from the outboard face.
module tripod_cut_local(thickness = plate_t, counterbore = true) {
    translate([cam_lens_dy, 0, 0]) union() {
    hull() for (dy = [-tripod_travel / 2, tripod_travel / 2])
        translate([0, dy, pz - thickness - 2])
            cylinder(d = tripod_d, h = thickness + 4);
    if (counterbore)
        hull() for (dy = [-tripod_travel / 2, tripod_travel / 2])
            translate([0, dy, pz - thickness - 0.01])
                cylinder(d = tripod_cb_d, h = tripod_cb_depth + 0.01);
    }
}

module flange_cuts(h = ring_t) {
    translate([0, 0, -1]) cylinder(d = ring_bore, h = h + 2);
    for (i = [0 : bolt_n - 1])
        rotate([0, 0, bolt_clock + i * 360 / bolt_n])
            translate([bolt_bcd / 2, 0, -1])
                cylinder(d = bolt_hole_d, h = h + 2);
    for (a = pin_clock)
        rotate([0, 0, a])
            translate([bolt_bcd / 2, 0, -1])
                cylinder(d = pin_hole_d, h = h + 2);
}

// Keep the arm flat where a downward-hanging adapter wing has to cross it.
module wing_cut() {
    if (wing_notch)
        translate([(wing_notch_r[0] + wing_notch_r[1]) / 2, 0,
                   ring_t + (ramp_h + plate_t) / 2])
            cube([wing_notch_r[1] - wing_notch_r[0], wing_notch_w, ramp_h + plate_t],
                 center = true);
}

module tie_slots() {
    for (sy = [-1, 1])
        translate([52, sy * 22, -1])
            hull() for (dx = [-1, 1])
                translate([dx * (tie_slot[1] - tie_slot[0]) / 2, 0, 0])
                    cylinder(d = tie_slot[0], h = ramp_h + 2);
}

// Engraved on the outboard face of the backing plate -- the one surface that is
// flat, visible with everything assembled, and does no work.
module engraving() {
    if (label)
        multmatrix(cam_M)
            translate([cam_lens_dy - 26, 0, pz - plate_t - 0.01])
                rotate([0, 0, 90]) mirror([1, 0, 0])
                    linear_extrude(0.7)
                        text(str("D435 t", tilt, " L", cam_lens_dy), size = 5,
                             halign = "center", valign = "center");
}

module mount() {
    difference() {
        union() {
            ring_and_arm();
            arm_ramp();
            tower();
            multmatrix(cam_M) cradle_local();
        }
        multmatrix(cam_M) tripod_cut_local();
        flange_cuts();
        wing_cut();
        tie_slots();
        engraving();
    }
}

// Stack gauge: the ring at the REAL thickness, with four stub ribs marking the
// four azimuths the arm could take. Bolt it between the flange and the existing
// adapter and it answers everything at once -- bolt circle, whether the screws
// are long enough, whether the adapter still seats flat, whether a pin is in
// the way, and which of the four arm directions the adapter's wings leave free.
module gauge_flange() {
    difference() {
        union() {
            cylinder(d = ring_od, h = ring_t);
            for (i = [0 : 3])
                rotate([0, 0, 90 * i])
                    translate([(18 + 55) / 2, 0, ring_t / 2])
                        cube([55 - 18, i == 0 ? 20 : 12, ring_t], center = true);
        }
        flange_cuts();
    }
}

// Camera gauge: the cradle alone, flat and thin. Confirms the end wall spacing
// and that the 1/4"-20 lines up, in about half an hour of printing.
// Camera gauge: the cradle exactly as it is on the mount -- same plate, same
// counterbore, same material under the head -- so the screw either clamps here
// or it will not clamp there either. Lies flat on the bed as printed.
module gauge_camera() {
    translate([0, 0, cam_h / 2 + plate_t])
        difference() {
            cradle_local();
            tripod_cut_local();
        }
}

if (part == "mount")             mount();
else if (part == "gauge_flange") gauge_flange();
else if (part == "gauge_camera") gauge_camera();
else                             echo("unknown part", part);
