import xml.etree.ElementTree as ET

VESSELS = {"beaker_A", "beaker_B", "flask_A", "flask_B"}
OBJECTS = {"box_A", "bottle_A"}
PLATES = {"plate_A", "plate_B"}
REAGENTS = {"water", "ethanol"}
VOLUMES = {"10 mL", "25 mL", "50 mL"}
STIR_TIMES = {"30 s", "1 min", "2 min"}
TEMPS = {"0 C", "25 C", "60 C"}

ALLOWED_TAGS = {
    "Add", "Stir", "HeatChill", "Transfer", "CleanVessel", "Move"
}

# Attributes each operator may carry. Undefined TAGS were already rejected, but
# an undefined ATTRIBUTE on an otherwise well-formed step passed silently --
# <Add vessel reagent volume foo='bar' /> and <Move object place speed='fast' />
# both validated -- so the "undefined tag or attribute" check was only half
# implemented. None of the 100 generations of Section 4.2.1 carries an
# out-of-schema attribute, so adding this does not change that measurement.
# Vessel capacities, from the glassware actually purchased for the physical
# cell (author, 2026-09-11): 100 mL beakers, 300 mL flasks. The manuscript
# claims the validator covers vessel capacity; it did not, and this is the
# information that check needs. The 100-instruction test set never exercises it
# -- the largest fill any of those procedures reaches is 50 mL -- so this check
# is validated by the injected-error benchmark rather than by that set.
CAPACITY_ML = {
    "beaker_A": 100.0, "beaker_B": 100.0,
    "flask_A": 300.0, "flask_B": 300.0,
}

ALLOWED_ATTRS = {
    "Add": {"vessel", "reagent", "volume"},
    "Stir": {"vessel", "time"},
    "HeatChill": {"vessel", "temp", "active"},
    "Transfer": {"from_vessel", "to_vessel", "volume"},
    "CleanVessel": {"vessel"},
    "Move": {"object", "place"},
}

class ProcedureValidationError(Exception):
    pass


def _volume_ml(text):
    """Millilitres from a volume attribute, or None if it is not one."""
    try:
        return float(str(text).split()[0])
    except (ValueError, AttributeError, IndexError):
        return None


class ProcedureValidator:
    def __init__(self):
        self.vessel_state = {v: "empty" for v in VESSELS}
        # Millilitres currently in each vessel, for the capacity check.
        self.vessel_volume = {v: 0.0 for v in VESSELS}
        # Which object occupies each plate, for the device-placement check.
        self.plate_occupant = {p: None for p in PLATES}
        # The reagent each vessel currently holds, for the contamination check.
        # None means empty or cleaned.
        self.vessel_content = {v: None for v in VESSELS}

    def validate(self, xml_str: str):
        try:
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            raise ProcedureValidationError(f"Invalid XML format: {e}")

        if root.tag != "procedure":
            raise ProcedureValidationError("Root must be <procedure>")

        steps = list(root)
        if len(steps) == 0:
            raise ProcedureValidationError("Procedure is empty")

        for step in steps:
            self.validate_step(step)

        return True

    def validate_step(self, step):
        tag = step.tag

        if tag not in ALLOWED_TAGS:
            raise ProcedureValidationError(f"Invalid tag: {tag}")

        undefined = set(step.attrib) - ALLOWED_ATTRS[tag]
        if undefined:
            raise ProcedureValidationError(
                f"Invalid attribute in {tag}: {sorted(undefined)}")

        if tag == "Add":
            self.validate_add(step)
        elif tag == "Stir":
            self.validate_stir(step)
        elif tag == "HeatChill":
            self.validate_heatchill(step)
        elif tag == "Transfer":
            self.validate_transfer(step)
        elif tag == "CleanVessel":
            self.validate_clean(step)
        elif tag == "Move":
            self.validate_move(step)

    def validate_add(self, step):
        vessel = step.attrib.get("vessel")
        reagent = step.attrib.get("reagent")
        volume = step.attrib.get("volume")

        if vessel not in VESSELS:
            raise ProcedureValidationError(f"Invalid vessel in Add: {vessel}")
        if reagent not in REAGENTS:
            raise ProcedureValidationError(f"Invalid reagent: {reagent}")
        if volume not in VOLUMES:
            raise ProcedureValidationError(f"Invalid volume: {volume}")

        # Contamination: a vessel already holding a different reagent has to be
        # cleaned first. This uses only state the validator already tracks, and
        # is the same class of physical precondition as pouring from an empty
        # vessel -- not chemical knowledge, which stays out of scope.
        held = self.vessel_content.get(vessel)
        if held is not None and held != reagent:
            raise ProcedureValidationError(
                f"Contamination: {reagent} into {vessel}, which holds {held}, "
                f"without an intervening CleanVessel")
        self.vessel_content[vessel] = reagent

        added = _volume_ml(volume)
        cap = CAPACITY_ML.get(vessel)
        if added is not None and cap is not None:
            if self.vessel_volume[vessel] + added > cap:
                raise ProcedureValidationError(
                    f"Add overfills {vessel}: "
                    f"{self.vessel_volume[vessel] + added:.0f} mL into a "
                    f"{cap:.0f} mL vessel")
            self.vessel_volume[vessel] += added

        self.vessel_state[vessel] = "filled"

    def validate_stir(self, step):
        vessel = step.attrib.get("vessel")
        time = step.attrib.get("time")

        if vessel not in VESSELS:
            raise ProcedureValidationError(f"Invalid vessel in Stir: {vessel}")
        if time not in STIR_TIMES:
            raise ProcedureValidationError(f"Invalid stir time: {time}")
        if self.vessel_state[vessel] != "filled":
            raise ProcedureValidationError(f"Cannot Stir empty vessel: {vessel}")

    def validate_heatchill(self, step):
        vessel = step.attrib.get("vessel")
        temp = step.attrib.get("temp")
        active = step.attrib.get("active")

        if vessel not in VESSELS:
            raise ProcedureValidationError(f"Invalid vessel in HeatChill: {vessel}")
        if temp not in TEMPS:
            raise ProcedureValidationError(f"Invalid temperature: {temp}")
        if active != "true":
            raise ProcedureValidationError("HeatChill must have active='true'")
        if self.vessel_state[vessel] != "filled":
            raise ProcedureValidationError(f"Cannot HeatChill empty vessel: {vessel}")

    def validate_transfer(self, step):
        from_v = step.attrib.get("from_vessel")
        to_v = step.attrib.get("to_vessel")
        volume = step.attrib.get("volume")

        if from_v not in VESSELS or to_v not in VESSELS:
            raise ProcedureValidationError("Invalid vessel in Transfer")
        if from_v == to_v:
            raise ProcedureValidationError("Transfer from_vessel and to_vessel must differ")
        if volume not in VOLUMES:
            raise ProcedureValidationError(f"Invalid volume in Transfer: {volume}")
        if self.vessel_state[from_v] != "filled":
            raise ProcedureValidationError(f"Cannot Transfer from empty vessel: {from_v}")

        src_content = self.vessel_content.get(from_v)
        dst_content = self.vessel_content.get(to_v)
        if (src_content is not None and dst_content is not None
                and src_content != dst_content):
            raise ProcedureValidationError(
                f"Contamination: transferring {src_content} into {to_v}, which "
                f"holds {dst_content}, without an intervening CleanVessel")
        if src_content is not None:
            self.vessel_content[to_v] = src_content

        moved = _volume_ml(volume)
        cap = CAPACITY_ML.get(to_v)
        if moved is not None:
            if moved > self.vessel_volume[from_v] + 1e-9:
                raise ProcedureValidationError(
                    f"Cannot Transfer {moved:.0f} mL from {from_v}, which holds "
                    f"{self.vessel_volume[from_v]:.0f} mL")
            if cap is not None and self.vessel_volume[to_v] + moved > cap:
                raise ProcedureValidationError(
                    f"Transfer overfills {to_v}: "
                    f"{self.vessel_volume[to_v] + moved:.0f} mL into a "
                    f"{cap:.0f} mL vessel")
            self.vessel_volume[from_v] -= moved
            self.vessel_volume[to_v] += moved
            if self.vessel_volume[from_v] <= 1e-9:
                self.vessel_state[from_v] = "empty"

        self.vessel_state[to_v] = "filled"

    def validate_clean(self, step):
        vessel = step.attrib.get("vessel")

        if vessel not in VESSELS:
            raise ProcedureValidationError(f"Invalid vessel in CleanVessel: {vessel}")

        self.vessel_state[vessel] = "empty"
        self.vessel_volume[vessel] = 0.0
        self.vessel_content[vessel] = None

    def validate_move(self, step):
        obj = step.attrib.get("object")
        place = step.attrib.get("place")

        if obj not in OBJECTS:
            raise ProcedureValidationError(f"Invalid object in Move: {obj}")
        if place not in PLATES:
            raise ProcedureValidationError(f"Invalid place in Move: {place}")

        # Device placement: a plate holds one object. Moving onto an occupied
        # plate is a placement conflict; moving an object off its current plate
        # frees that plate.
        occupant = self.plate_occupant[place]
        if occupant is not None and occupant != obj:
            raise ProcedureValidationError(
                f"Device placement conflict: {place} already holds {occupant}")
        for p, who in self.plate_occupant.items():
            if who == obj:
                self.plate_occupant[p] = None
        self.plate_occupant[place] = obj
