import random
import os
import xml.etree.ElementTree as ET

# -----------------------------
# Output path
# -----------------------------

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset", "pre_xdl_data")
OUTPUT_FILE = "xdl_dataset_v1.xml"
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
# XDL Element Generators
# -----------------------------

def add_elem(parent, tag, **attrs):
    elem = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        elem.set(k, v)
    return elem

def gen_add(parent, vessel):
    add_elem(parent, "Add",
             vessel=vessel,
             reagent=random.choice(REAGENTS),
             volume=random.choice(VOLUMES))

def gen_transfer(parent, from_vessel, to_vessel):
    add_elem(parent, "Transfer",
             from_vessel=from_vessel,
             to_vessel=to_vessel,
             volume=random.choice(VOLUMES))

def gen_stir(parent, vessel):
    add_elem(parent, "Stir",
             vessel=vessel,
             time=random.choice(STIR_TIMES))

def gen_heatchill(parent, vessel):
    add_elem(parent, "HeatChill",
             vessel=vessel,
             temp=random.choice(TEMPS),
             active="true")

def gen_clean(parent, vessel):
    add_elem(parent, "CleanVessel",
             vessel=vessel)

def gen_move(parent):
    add_elem(parent, "Move",
             object=random.choice(OBJECTS),
             place=random.choice(PLATES))

# -----------------------------
# Procedure Generator
# -----------------------------

def generate_procedure_element():
    procedure = ET.Element("procedure")
    vessel = random.choice(VESSELS)

    gen_add(procedure, vessel)

    if random.random() < 0.7:
        gen_stir(procedure, vessel)

    if random.random() < 0.5:
        gen_heatchill(procedure, vessel)

    if random.random() < 0.5:
        target = random.choice([v for v in VESSELS if v != vessel])
        gen_transfer(procedure, vessel, target)
        vessel = target

    if random.random() < 0.3:
        gen_clean(procedure, vessel)

    if random.random() < 0.5:
        gen_move(procedure)

    return procedure

# -----------------------------
# Pretty-print single procedure
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
        # 마지막 child 뒤에는 부모 레벨로 복귀
        if not elem[-1].tail or not elem[-1].tail.strip():
            elem[-1].tail = i
    else:
        if not elem.tail or not elem.tail.strip():
            elem.tail = i

# -----------------------------
# Write procedures (NO root)
# -----------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    # f.write('<?xml version="1.0" encoding="utf-8"?>\n')

    for _ in range(NUM_SAMPLES):
        proc = generate_procedure_element()
        indent(proc)

        xml_str = ET.tostring(proc, encoding="unicode")
        f.write(xml_str)
        f.write("\n")  

print("Saved procedure-only XML (no root tag) to:")
print(OUTPUT_PATH)
