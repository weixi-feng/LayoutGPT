import os
import argparse
import re
import cssutils
from tqdm import tqdm
from string import digits

from utils import *


def parse_layout(string, canvas_size=64, no_integer=False):
    idx = string.find(' {')
    category = re.sub(r'[0-9]', '', string[:idx].replace(' ', '-'))
    string = category + string[idx:]
    try:
        sheet = cssutils.parseString(string)
        rule = sheet.cssRules[0]
        bbox = [b.strip().strip(";").strip() for b in rule.style.cssText.split("\n")]
        text = rule.selectorText
    except:
        try:
            text, bbox = string.split("{")
            bbox = bbox.strip().strip("}").strip().strip(";").split(";")
            assert len(bbox) == 4
            bbox = [b.strip().strip(";").strip() for b in bbox]
        except:
            return None, None

    category = text.strip()
    parsed_category = category.translate(category.maketrans('', '', digits)).strip()

    bbox = [b.split(":") for b in bbox]
    if no_integer:
        bbox = {k.strip():float(v.lstrip().rstrip()) for k, v in bbox}
    else:
        bbox = {k.strip():int(v.lstrip().rstrip("px")) for k, v in bbox}

    if sorted(bbox.keys()) != ['height', 'left', 'top', 'width']:
        print(string)
        bbox = [0] * 4
        return parsed_category, bbox

    bbox = [bbox['left'], bbox['top'], min(bbox['left']+bbox['width'], canvas_size), min(bbox['top']+bbox['height'], canvas_size)]
    if bbox[0] >= canvas_size or bbox[1] >= canvas_size:
        return None, None
    bbox = [float(b)/canvas_size for b in bbox]

    return parsed_category, bbox


def parse_3D_layout(string, unit='m'):
    # cannot use cssutils due to self-defined properties
    try:
        text, bbox = string.split("{")
        bbox = bbox.strip().strip("}").strip().strip(";").split(";")
        assert len(bbox) == 7
        bbox = [b.strip().strip(";").rstrip("degrees").strip() for b in bbox]
    except:
        return None, None

    category = text.strip()
    parsed_category = category.translate(category.maketrans('', '', digits)).strip()

    bbox = [b.split(":") for b in bbox]

    bbox = {k.strip():float(v.lstrip().rstrip(unit)) for k, v in bbox}
    if sorted(bbox.keys()) != sorted(['height', 'width', 'length', 'orientation', 'left', 'top', 'depth']):
        print(string)
        bbox = {k:0 for k in bbox.keys()}
        return parsed_category, bbox

    return parsed_category, bbox



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--files", nargs="+")
    args = parser.parse_args()
    
    for fname in args.files:
        basename = os.path.basename(fname)
        dirname = os.path.dirname(fname)

        assert "raw" not in basename
        response = load_json(fname)

        print(f"Parsing {basename}")
        for r in tqdm(response):
            layout = r['text'].strip().strip("\n").strip().split("\n")
            assert len(layout) >= 2
            r['objects'] = []

            for elm in layout:
                selector_text, bbox = parse_layout(elm)
                if selector_text == None:
                    continue
                if sum(bbox) == 0:
                    print("Failed")
                r['objects'].append([selector_text, bbox])
        
        write_json(os.path.join(dirname, "parsed_"+basename), response)
                

    
    