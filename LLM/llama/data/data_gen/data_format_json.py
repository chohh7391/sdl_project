import random
import os
import json
import xml.etree.ElementTree as ET

# -----------------------------
# Output path
# -----------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "pre_xdl_data")
OUTPUT_FILE = "procedure_instruction_v1.jsonl"
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

NUM_SAMPLES = 5000   # 늘릴 때 여기만 바꾸면 됨

# -----------------------------
# Natural Language Templates (English, diverse)
# -----------------------------

def nl_add(vessel, reagent, volume):
    return random.choice([
        f"Add {volume} of {reagent} to {vessel}.",
        f"Pour {volume} of {reagent} into {vessel}.",
        f"Introduce {volume} of {reagent} into the {vessel}.",
        f"Place {volume} of {reagent} into {vessel}."
    ])

def nl_stir(vessel, time):
    return random.choice([
        f"Stir {vessel} for {time}.",
        f"Mix the contents of {vessel} for {time}.",
        f"Agitate {vessel} for {time}.",
        f"Stir the solution in {vessel} for {time}."
    ])

def nl_heatchill(vessel, temp):
    return random.choice([
        f"Heat {vessel} to {temp}.",
        f"Set the temperature of {vessel} to {temp}.",
        f"Adjust {vessel} to {temp}.",
        f"Bring {vessel} to {temp}."
    ])

def nl_transfer(fv, tv, volume):
    return random.choice([
        f"Transfer {volume} from {fv} to {tv}.",
        f"Move {volume} of liquid from {fv} into {tv}.",
        f"Pour {volume} from {fv} into {tv}.",
        f"Send {volume} from {fv} to {tv}."
    ])

def nl_clean(vessel):
    return random.choice([
        f"Clean {vessel}.",
        f"Wash {vessel}.",
        f"Rinse and clean {vessel}.",
        f"Perform cleaning on {vessel}."
    ])

def nl_move(obj, place):
    return random.choice([
        f"Move {obj} to {place}.",
        f"Place {obj} on {place}.",
        f"Relocate {obj} onto {place}.",
        f"Put {obj} on top of {place}."
    ])

# -----------------------------
# XML Indent (stable)
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
# Generate one sample
# -----------------------------

def generate_sample():
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

    # Clean
    if random.random() < 0.3:
        ET.SubElement(procedure, "CleanVessel", vessel=vessel)
        instruction_steps.append(nl_clean(vessel))

    # Move object
    if random.random() < 0.5:
        obj = random.choice(OBJECTS)
        place = random.choice(PLATES)
        ET.SubElement(procedure, "Move", object=obj, place=place)
        instruction_steps.append(nl_move(obj, place))

    # Pretty XML
    indent(procedure)
    xml_str = ET.tostring(procedure, encoding="unicode")

    return {
        "instruction": " ".join(instruction_steps),
        "output": xml_str
    }

# -----------------------------
# Generate JSONL dataset
# -----------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for _ in range(NUM_SAMPLES):
        # random.seed(_)
        sample = generate_sample()
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

print("Saved JSONL dataset to:")
print(OUTPUT_PATH)
