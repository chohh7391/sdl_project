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

#include <algorithm>
#include <cmath>

namespace sdl_pour_controller {
namespace pour {

/**
 * One coordinate of the pour, moved with bounded speed AND acceleration.
 *
 * Every motion the pour makes -- the tilt, the lip's run along the align path,
 * the way back -- used to be a constant rate that started, stopped and reversed
 * inside one control cycle: a trim pulse's creep at 0.26 deg/s became an 8.6
 * deg/s retract from one cycle to the next. The velocity is now continuous and
 * the acceleration bounded, so each move is a trapezoid (a triangle when it is
 * too short to reach its speed), and a move to a target brakes to arrive at
 * rest instead of stopping dead on it.
 *
 * With a jerk bound set, the acceleration is continuous too: each corner of the
 * trapezoid becomes an S-curve. On the rig (2026-10-03) the trapezoid's corners
 * -- every start, stop and reversal of a syrup pour -- read as knocks.
 *
 * No clock and no ROS: a velocity (and an acceleration) it carries between
 * calls, and a step it hands back for the caller to take.
 */
class Ramp
{
public:
    [[nodiscard]] double velocity() const { return v_; }
    [[nodiscard]] double acceleration() const { return a_; }

    void reset(double velocity = 0.0)
    {
        v_ = std::isfinite(velocity) ? velocity : 0.0;
        a_ = 0.0;
    }

    //: How fast the acceleration may change [unit/s^3]. 0 (the default) leaves
    //: the acceleration free to jump, which is the plain trapezoid.
    void set_jerk(double jerk) { jerk_ = (std::isfinite(jerk) && jerk > 0.0) ? jerk : 0.0; }
    [[nodiscard]] double jerk() const { return jerk_; }

    //: Follow a commanded rate, reaching it no faster than `accel`. Returns the
    //: step to take this cycle.
    double follow(double rate, double accel, double dt)
    {
        if (!std::isfinite(rate) || !(accel > 0.0) || !(dt > 0.0)) {
            v_ = 0.0;
            a_ = 0.0;
            return 0.0;
        }
        if (jerk_ > 0.0) {
            return follow_jerk(rate, accel, dt);
        }
        const double dv = accel * dt;
        const double before = v_;
        v_ += std::clamp(rate - v_, -dv, dv);
        a_ = (v_ - before) / dt;
        return v_ * dt;
    }

    //: Head for `target` at up to `max_rate`, braking to arrive there at rest.
    //: Returns the step to take this cycle; the last one lands exactly on the
    //: target, so a caller can test for arrival with ==.
    double reach(double position, double target, double max_rate, double accel, double dt)
    {
        const double error = target - position;
        if (!std::isfinite(error) || !(accel > 0.0) || !(dt > 0.0)) {
            v_ = 0.0;
            return 0.0;
        }
        // A cap of zero is "stop", and stopping is braked like anything else:
        // a caller throttling the speed down to nothing gets a deceleration,
        // not a step to rest.
        const double cap = std::isfinite(max_rate) ? std::max(0.0, max_rate) : 0.0;
        // The speed from which `accel` just stops the coordinate on the target.
        // With a jerk bound the stop also has to build its deceleration up and
        // let it go again, which costs v a / (2 j) more: v^2/(2a) + v a/(2j) = |e|.
        double braking = std::sqrt(2.0 * accel * std::abs(error));
        if (jerk_ > 0.0) {
            const double a2j = accel * accel / jerk_;
            braking = 0.5 * (-a2j + std::sqrt(a2j * a2j + 8.0 * accel * std::abs(error)));
        }
        const double step = follow(std::copysign(std::min(cap, braking), error), accel, dt);
        if (std::abs(error) <= std::abs(step) + 1e-12 && step * error >= 0.0) {
            v_ = 0.0;
            a_ = 0.0;
            return error;
        }
        return step;
    }

    //: Slow to rest at `accel`, the jerk bound set aside: the stop for a path
    //: the arm has fallen behind, which used to be reset() -- the speed gone in
    //: one cycle, a knock -- and is still a deceleration, only a hard one.
    //: Never reverses. Returns the step to take this cycle.
    double brake(double accel, double dt)
    {
        if (!(accel > 0.0) || !(dt > 0.0)) {
            v_ = 0.0;
            a_ = 0.0;
            return 0.0;
        }
        const double before = v_;
        v_ -= std::copysign(std::min(std::abs(v_), accel * dt), v_);
        a_ = (v_ - before) / dt;
        return v_ * dt;
    }

    //: What the coordinate actually did this cycle, when something downstream --
    //: a clamp, a lagging IK, a path that holds the tilt back -- took less of
    //: the step than was handed out. The next cycle accelerates from THAT.
    void observed(double step, double dt)
    {
        const double v = (dt > 0.0 && std::isfinite(step)) ? step / dt : 0.0;
        if (std::abs(v - v_) > 1e-9) {
            // Cut short downstream: whatever acceleration it had is gone too.
            a_ = 0.0;
        }
        v_ = v;
    }

private:
    //: The velocity still gained if the acceleration `a` is shed at the jerk
    //: bound from the next cycle on -- summed cycle by cycle, as it will be
    //: taken, so the shedding starts exactly when it has to.
    [[nodiscard]] double coast(double a, double dt) const
    {
        const double jd = jerk_ * dt;
        const double s = std::abs(a);
        const double n = std::floor(s / jd);
        return std::copysign(dt * (n * s - jd * n * (n + 1.0) / 2.0), a);
    }

    //: One cycle toward `rate` with the acceleration moved by at most jerk x dt:
    //: raise it, hold it, or shed it -- the first of those from which shedding
    //: afterwards still lands on the rate without passing it.
    double follow_jerk(double rate, double accel, double dt)
    {
        const double jd = jerk_ * dt;
        const double e = rate - v_;
        if (std::abs(e) <= jd * dt && std::abs(a_) <= jd) {
            // Less than one cycle's worth from the rate with next to no
            // acceleration left: land on it.
            v_ = rate;
            a_ = 0.0;
            return v_ * dt;
        }
        const double toward = e >= 0.0 ? 1.0 : -1.0;
        double next = std::clamp(a_ - toward * jd, -accel, accel);
        for (const double candidate : {a_ + toward * jd, a_}) {
            const double a = std::clamp(candidate, -accel, accel);
            if (toward * (e - a * dt - coast(a, dt)) >= -1e-12) {
                next = a;
                break;
            }
        }
        a_ = next;
        double dv = a_ * dt;
        if (toward * (dv - e) > 0.0) {
            dv = e;
            a_ = 0.0;
        }
        v_ += dv;
        return v_ * dt;
    }

    double v_{0.0};
    double a_{0.0};
    double jerk_{0.0};
};

} // namespace pour
} // namespace sdl_pour_controller
