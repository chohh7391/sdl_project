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

#include "sdl_pour_controller/pour/pour_types.hpp"

namespace sdl_pour_controller {
namespace pour {

const char * to_string(PourPhase phase)
{
    switch (phase) {
        case PourPhase::Verify:  return "verify";
        case PourPhase::Seek:    return "seek";
        case PourPhase::Bulk:    return "bulk";
        case PourPhase::Retract: return "retract";
        case PourPhase::Settle:  return "settle";
        case PourPhase::Trim:    return "trim";
        case PourPhase::Done:    return "done";
    }
    return "?";
}

} // namespace pour
} // namespace sdl_pour_controller
