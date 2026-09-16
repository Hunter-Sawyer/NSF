#Cleans network files from more recent versions of sumo with unsupported vehicle classes
#the HPC is on 1.5.0 and the UNSUPPORTED set all cause issues.




from pathlib import Path
from collections import Counter
import argparse
import re

UNSUPPORTED = {
    "subway",
    "aircraft",
    "wheelchair",
    "scooter",
    "drone",
    "container",
    "cable_car",
}

ATTRIBUTE_PATTERN = re.compile(
    r'(?P<prefix>\b(?:allow|disallow)\s*=\s*")'
    r'(?P<values>[^"]*)'
    r'(?P<suffix>")'
)

removed = Counter()


def clean_attribute(match):
    values = match.group("values").split()
    kept = []

    for value in values:
        if value in UNSUPPORTED:
            removed[value] += 1
        else:
            kept.append(value)

    return (
        match.group("prefix")
        + " ".join(kept)
        + match.group("suffix")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_network")
    parser.add_argument("output_network")
    args = parser.parse_args()

    source = Path(args.input_network).resolve()
    output = Path(args.output_network).resolve()

    if source == output:
        raise ValueError("Output must be different from input")

    text = source.read_text(encoding="utf-8")
    cleaned = ATTRIBUTE_PATTERN.sub(clean_attribute, text)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(cleaned, encoding="utf-8")

    print(f"Wrote cleaned network: {output}")
    print("Removed vehicle classes:")
    for vehicle_class, count in sorted(removed.items()):
        print(f"  {vehicle_class}: {count}")


if __name__ == "__main__":
    main()