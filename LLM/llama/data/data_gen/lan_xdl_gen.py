import random
import os
import xml.etree.ElementTree as ET

# -----------------------------
# Output path
# -----------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "pre_xdl_data")
OUTPUT_FILE = "procedure_with_instruction_v1.xml"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Configuration
# -----------------------------

VESSELS = ["beaker_A", "beaker_B", "flask_A", "flask_B"]
OBJECTS = ["box_A", "bottle_A"]
PLATES = ["plate_A", "plate_B"]

REAGENTS = ["water", "ethanol"]
VOLUMES = ["10 mL", "25 mL", "50 mL"]
STIR_TIMES = ["30 s", "1 min", "2 min"]
TEMPS = ["0 C", "25 C", "60 C"]

NUM_SAMPLES = 10

# -----------------------------
# Natural Language Templates (English, diverse)
# -----------------------------

def nl_add(vessel, reagent, volume):
    templates = [
        f"Add {volume} of {reagent} to {vessel}.",
        f"Pour {volume} of {reagent} into {vessel}.",
        f"Introduce {volume} of {reagent} into the {vessel}.",
        f"Place {volume} of {reagent} into {vessel}."
    ]
    return random.choice(templates)

def nl_stir(vessel, time):
    templates = [
        f"Stir {vessel} for {time}.",
        f"Mix the contents of {vessel} for {time}.",
        f"Agitate {vessel} for {time}.",
        f"Stir the solution in {vessel} for {time}."
    ]
    return random.choice(templates)

def nl_heatchill(vessel, temp):
    templates = [
        f"Heat {vessel} to {temp}.",
        f"Set the temperature of {vessel} to {temp}.",
        f"Adjust {vessel} to {temp}.",
        f"Bring {vessel} to {temp}."
    ]
    return random.choice(templates)

def nl_transfer(from_vessel, to_vessel, volume):
    templates = [
        f"Transfer {volume} from {from_vessel} to {to_vessel}.",
        f"Move {volume} of liquid from {from_vessel} into {to_vessel}.",
        f"Pour {volume} from {from_vessel} into {to_vessel}.",
        f"Send {volume} from {from_vessel} to {to_vessel}."
    ]
    return random.choice(templates)

def nl_clean(vessel):
    templates = [
        f"Clean {vessel}.",
        f"Wash {vessel}.",
        f"Rinse and clean {vessel}.",
        f"Perform cleaning on {vessel}."
    ]
    return random.choice(templates)

def nl_move(obj, place):
    templates = [
        f"Move {obj} to {place}.",
        f"Place {obj} on {place}.",
        f"Relocate {obj} onto {place}.",
        f"Put {obj} on top of {place}."
    ]
    return random.choice(templates)

# -----------------------------
# XML Indent (correct version)
# -----------------------------

def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i + "  "
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    else:
        if not elem.tail or not elem.tail.strip():
            elem.tail = i

# -----------------------------
# Generate one procedure + instruction
# -----------------------------

def generate_procedure_with_instruction():
    procedure = ET.Element("procedure")
    instruction_steps = []

    vessel = random.choice(VESSELS)

    # Add (always first)
    reagent = random.choice(REAGENTS)
    volume = random.choice(VOLUMES)
    ET.SubElement(procedure, "Add",
                  vessel=vessel,
                  reagent=reagent,
                  volume=volume)
    instruction_steps.append(nl_add(vessel, reagent, volume))

    # Stir
    if random.random() < 0.7:
        t = random.choice(STIR_TIMES)
        ET.SubElement(procedure, "Stir", vessel=vessel, time=t)
        instruction_steps.append(nl_stir(vessel, t))

    # HeatChill
    if random.random() < 0.5:
        temp = random.choice(TEMPS)
        ET.SubElement(procedure, "HeatChill",
                      vessel=vessel,
                      temp=temp,
                      active="true")
        instruction_steps.append(nl_heatchill(vessel, temp))

    # Transfer
    if random.random() < 0.5:
        target = random.choice([v for v in VESSELS if v != vessel])
        vol = random.choice(VOLUMES)
        ET.SubElement(procedure, "Transfer",
                      from_vessel=vessel,
                      to_vessel=target,
                      volume=vol)
        instruction_steps.append(nl_transfer(vessel, target, vol))
        vessel = target

    # CleanVessel
    if random.random() < 0.3:
        ET.SubElement(procedure, "CleanVessel", vessel=vessel)
        instruction_steps.append(nl_clean(vessel))

    # Move object (independent)
    if random.random() < 0.5:
        obj = random.choice(OBJECTS)
        place = random.choice(PLATES)
        ET.SubElement(procedure, "Move", object=obj, place=place)
        instruction_steps.append(nl_move(obj, place))

    # Instruction element
    instruction = ET.Element("instruction")
    instruction.text = " ".join(instruction_steps)

    return instruction, procedure

# -----------------------------
# Write XML (no root wrapper)
# -----------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    # f.write('<?xml version="1.0" encoding="utf-8"?>\n\n')

    for _ in range(NUM_SAMPLES):
        instruction, procedure = generate_procedure_with_instruction()
        indent(instruction)
        indent(procedure)

        f.write(ET.tostring(instruction, encoding="unicode"))
        f.write("\n")
        f.write(ET.tostring(procedure, encoding="unicode"))
        f.write("\n\n")

print("Saved procedure + instruction dataset to:")
print(OUTPUT_PATH)
