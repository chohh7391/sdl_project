// Copyright 2026 Hyunho Cho
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <string>

namespace sdl_pour_controller {
namespace pour {

/**
 * What is being poured. The two classes do not share a flow law.
 *
 * A liquid's outflow is a continuous monotone function of tilt and remaining
 * volume, so a flow rate can be regulated. A granular medium arrives in
 * avalanches whose size is set by the aperture and the packing: below some tilt
 * nothing moves, above it a lump goes. Regulating its flow rate is not possible
 * and the controller only meters the avalanches.
 */
enum class MaterialClass {
    Liquid = 0,
    Granular = 1,
};

/**
 * One endpoint of a class: how a material at that extreme of flow_index pours.
 *
 * Both endpoints of both classes come from configuration, never from this
 * header. Only the LIQUID/free endpoint has measured backing (water on the
 * HS-AA, 2026-09-16): 6-11 g/s continuous, 1.0 s to settle after the flow
 * stops, <=0.21 g of tail. Every other endpoint is a commissioning seed.
 */
struct PourLimits {
    //: Bulk-phase outflow ceiling [g/s]. The tilt is regulated to hold this.
    double max_flow_rate{0.0};
    //: Outflow a trim pulse is metered at [g/s]. Low on purpose: the stop error
    //: is proportional to the flow rate at the moment the tilt stops, so the
    //: last grams have to arrive slowly however fast the bulk phase ran.
    double trim_flow_rate{0.0};
    //: Shortest pulse that still breaks the material loose [s]. With
    //: trim_flow_rate this sets the smallest dose the controller can meter.
    double trim_pulse_sec{0.0};
    //: Seconds between material leaving the lip and the scale reporting it.
    //: Its uncertainty, times the flow rate, dominates the stop error.
    double transport_delay{0.0};
    //: Prior for what still arrives after the tilt stops [g]. Re-measured from
    //: the pour itself and replaced for every trim pulse after the first.
    double afterflow_grams{0.0};
    //: How long the reading must hold still, with the indicator reporting
    //: stable, before it is believed [s].
    double settle_hold_sec{0.0};
    //: Tilt rate for the bulk and trim phases [rad/s].
    double tilt_rate{0.0};
    //: Tilt rate while hunting for the angle at which flow starts [rad/s].
    //: Slower than tilt_rate: overshooting the onset of a thick or cohesive
    //: material dumps it.
    double seek_tilt_rate{0.0};
    //: Smallest amount that can arrive as one piece [g]. A tolerance below this
    //: is not achievable by any control law -- for water broken into drops it is
    //: the drop, measured at 1.2 g on the test rig -- so it floors the goal's
    //: tolerance instead of being chased.
    double dose_quantum{0.0};

    //: How far under the angle it started flowing at a material has to be
    //: tipped back before it stops [rad]. Water stops at its onset; a thick
    //: liquid that is already running keeps running below it -- oligosaccharide
    //: syrup on the rig (2026-10-03) poured 1.2 g/s parked 0.1 deg under its
    //: onset and still dripped 10 deg under it -- so a stop parks this much
    //: further back. Zero for water and for granular media.
    double park_drop{0.0};

    //: How many of this pour's own (tilt, flow) pairs the flow model needs
    //: before a stop is decided from its forecast; until then it is decided
    //: from the flow on the scale times the delay. The model is fitted from the
    //: onset on, so its first pair is the first mass to land -- and a thick
    //: liquid lets go of what it held at the lip as one lump. On the rig
    //: (2026-10-03 and -04) syrup landed 0.82-0.86 g inside one 0.2 s sample,
    //: the model fitted on it forecast 9 g still to come, and the pour stopped
    //: at 0.86 g and spent the next 100 s in trim pulses. Zero for water: in
    //: the sim, waiting made its stops worse (flow_model.min_pairs). The larger
    //: of this and flow_model.min_pairs applies.
    double forecast_min_pairs{0.0};

    //: Goals smaller than this [g] seek their onset slower, in proportion, down
    //: to PlannerConfig::slow_seek_min_fraction of seek_tilt_rate. The seek
    //: goes on turning for a whole transport delay after the first of the
    //: stream leaves the lip, so it arrives that far past the onset; water on
    //: the rig (2026-10-04) was running ~5 g/s by the time its 10 g goals saw
    //: it, stopped at 3-5 g on 3-6 g forecast in the air, and trimmed the rest
    //: over by 0.8-2.7 g, every one of ten. A larger goal has the room to meter
    //: that stream down. 0 is off.
    double slow_seek_below_grams{0.0};

    //: What the flow on the scale times the delay is believed of, as the mass
    //: in the air, while the stop waits for forecast_min_pairs. Right after the
    //: onset that flow is the rush off a wet lip, not the stream: water on the
    //: rig (2026-10-04) read 5-7 g/s there, the stops it called at 2-3 g of a
    //: 10 g goal forecast 5-7 g in the air, and 2.3-2.8 g came, tail and all.
    //: Below 1 the pour carries on through the rush and is metered by the tilt
    //: law instead of stopped and trimmed. 1 believes it whole. In (0, 1].
    double early_flight_factor{1.0};

    //: How far above its onset the bulk keeps the vessel when it steps the tilt
    //: down for too much flow [rad]: the stream is throttled, not broken. A
    //: broken stream of water starts again only past where it stopped, all at
    //: once off the wet lip (2026-09-24, 2026-10-04: 1.3-2.3 g in one go), so
    //: metering it back down to nothing and up again costs more than it saves.
    //: The one stop that breaks it is the stop at the target. 0 is off: the
    //: law may step down as far as its lead takes it.
    double keep_flowing_margin{0.0};

    //: What the forecast of what still comes after a stop is believed of, at
    //: the stop and in the settle's grading of it. Water on the rig (2026-10-04,
    //: ten 10 g stops, the stream a second from the scale) let 0.21-1.52x of the
    //: forecast come, median 0.7, the larger forecasts the most over: stops came
    //: early and the trims that followed overpoured. 1 believes it whole.
    double stop_forecast_scale{1.0};

    //: Trim pulses this material may take, or -1 for the planner's
    //: max_trim_pulses. A stopped stream of water starts again off the wet lip
    //: all at once, 2-4 g in the air before the scale sees any (2026-10-04:
    //: every trim of a 10 g goal overpoured, by 1.5-2.2 g), so it is not
    //: trimmed: short of the target it ends short. Not interpolated -- the
    //: index's nearer endpoint's.
    double max_trim_pulses{-1.0};

    //: The last grams before the target over which the bulk no longer tilts
    //: further after the flow [g]: a stream thinning there is not taken up
    //: (see final_approach_rise for what it does follow). On the rig
    //: (2026-10-04, ten 10 g water pours) the stream thinned 2-3 g out as the
    //: beaker emptied, the law climbed 0.9-2.3 deg in the last 2 s to keep the
    //: flow, and with the scale a second behind, that climb landed as a 2.4 g/s
    //: surge at the stop: 0.9-1.9 g came after it, +1.0 and +1.2 g over. Only
    //: a stream the scale has shown stopped outright is climbed after. 0 is off.
    double final_approach_grams{0.0};

    //: How far the onset rises per gram that leaves the lip [rad/g], which the
    //: final approach follows: it tilts on at this times the flow on the scale
    //: instead of holding. Held flat, the rig's beaker (2026-10-04) let two of
    //: five streams die 2.6 and ~1 g out and end short; its onset rose 0.0090
    //: rad/g (0.515 deg/g, fifteen steps 0-40 g, sd 0.71 deg), faster full
    //: than nearly empty. 0 holds.
    double final_approach_rise{0.0};

    //: How far below the highest tilt this pour has reached the bulk may step
    //: the vessel back for too much flow [rad]; -1 leaves it to the law's lead.
    //: Water broke four times in twelve 30 and 50 g pours on the rig
    //: (2026-10-04), each after the bulk stepped back 1.6-2.8 deg to slow down
    //: 6-10 g out, and came back only 3-6 deg higher, all at once, 2-3 s later:
    //: the stop called on that rush left one pour 2.26 g short. The eighteen
    //: that never stepped back never broke for longer than 0.8 s. 0 never
    //: steps back: the flow slows as the vessel empties, and the stop is left
    //: to the forecast.
    double max_step_down{-1.0};

    //: The taps in the final approach [rad, s], in place of shake_amplitude /
    //: shake_period there, and kept up there whether or not the flow is short.
    //: Held at its tilt and tapped 0.012 rad every 0.08 s, the rig's sugar
    //: (2026-10-05) did not trickle: nothing came for 6-7 s, the creep after a
    //: stopped stream went on 2.6-5 deg, and the bed let go 1.6-5 g at once.
    //: Larger, slower taps are to tip its surface past the angle of repose a
    //: little at a time. 0 keeps the ordinary taps.
    double final_approach_shake_amplitude{0.0};
    double final_approach_shake_period{0.0};

    //: The fastest the final approach may tilt [rad/s], below the material's
    //: own rate; 0 leaves it to the law. A stopped stream there is crept after
    //: at the lead-bounded climb, which on the rig's sugar (2026-10-05, five
    //: 10 g steps) came to 0.38-0.62 deg/s: nothing landed for 5-7 s while it
    //: went on 2.6-3.7 deg, and then 2.1-3.7 g let go at once, with another
    //: degree still climbed before the scale showed it.
    double final_approach_tilt_rate{0.0};
    // The three below may be zero: zero is what a liquid has.

    //: How much later than a liquid this material reaches the lip [rad]. A
    //: granular surface stands at its angle of repose instead of levelling, so
    //: from the same fill it has to tip about that much further. The seek
    //: covers seek_fast_until plus this at tilt_rate, because nothing can pour
    //: below it.
    double repose_angle{0.0};
    //: Taps that shake a granular bed loose [rad of the pour joint], and how
    //: often one starts [s]. Zero amplitude is off. See Shaker.
    double shake_amplitude{0.0};
    double shake_period{0.0};
};

/**
 * The two endpoints of one material class, and the interpolation between them.
 *
 * flow_index is the single number a caller supplies to say how reluctantly its
 * material flows within its class, and this is what turns that number into
 * every rate, delay and bound at once. One index moves the whole profile
 * together -- a higher one pours slower, waits longer for the stream, expects a
 * larger tail, and holds still longer before believing the scale -- which is
 * the point: those quantities are not independent, and letting a caller set
 * them separately invites a combination no material actually has.
 */
struct MaterialProfile {
    //: flow_index = 0. Water; dry salt or sugar.
    PourLimits free;
    //: flow_index = 1. Honey or syrup; damp clumping powder.
    PourLimits resistant;

    //: Linear on every field. flow_index outside [0, 1] is clamped rather than
    //: extrapolated: past the endpoints these are guesses about a material
    //: nobody characterised, and a guess should not be allowed to keep growing.
    [[nodiscard]] PourLimits at(double flow_index) const;

    //: Every field of both endpoints must be finite and positive -- except
    //: repose_angle and the shake, which may be zero -- and the resistant
    //: endpoint must actually be the slower one. Checked at configure
    //: so a transposed pair is refused with a vessel still on the bench rather
    //: than discovered with one in the gripper.
    [[nodiscard]] bool validate(const std::string & label, std::string & why) const;
};

//: Parse "liquid" / "granular". Returns false for anything else.
bool parse_material_class(const std::string & text, MaterialClass & out);

//: For log lines and abort messages.
const char * to_string(MaterialClass material);

} // namespace pour
} // namespace sdl_pour_controller
