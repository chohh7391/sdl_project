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

class ProcedureValidationError(Exception):
    pass


class ProcedureValidator:
    def __init__(self):
        self.vessel_state = {v: "empty" for v in VESSELS}

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

        self.vessel_state[to_v] = "filled"

    def validate_clean(self, step):
        vessel = step.attrib.get("vessel")

        if vessel not in VESSELS:
            raise ProcedureValidationError(f"Invalid vessel in CleanVessel: {vessel}")

        self.vessel_state[vessel] = "empty"

    def validate_move(self, step):
        obj = step.attrib.get("object")
        place = step.attrib.get("place")

        if obj not in OBJECTS:
            raise ProcedureValidationError(f"Invalid object in Move: {obj}")
        if place not in PLATES:
            raise ProcedureValidationError(f"Invalid place in Move: {place}")
