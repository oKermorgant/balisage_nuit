#!/usr/bin/env python3

import cv2
import argparse
from balisage import Light, GPS, Boat, dist
from pynput.keyboard import Listener
from numpy import vstack
import yaml

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.description = 'Balisage de nuit'
parser.add_argument('-p', '--pattern', type=str, help='Motif', default='')
parser.add_argument('-d', '--drift', type=float, help='Dérive (sud) due au vent en nœud', default=0)
parser.add_argument('-f', '--file', type=str, help='Configuration',default='zones/default.yaml')

args = parser.parse_args()

with open(args.file) as f:
    config = yaml.safe_load(f.read())

if args.pattern:
    config[args.pattern] = config.pop('Fl.3s')


gps = GPS(config)
top_base = gps.image()

lights = [Light.build(pat, geom) for pat,geom in config.items() if pat != 'start']

# write base top image
sectors = sum([light.sectors for light in lights],start=[])
sectors.sort(key = lambda s: -ord(s.color))

for sector in sectors:
    sector.write(top_base)
for sector in sectors:
    sector.write_borders(top_base)

for light in lights:
    light.display(top_base)


boat = Boat(config['start'], args.drift)
view_base = boat.image()


with Listener(on_press=boat.on_press, on_release=boat.on_release) as listener:

    while True:

        top = top_base.copy()
        view = view_base.copy()
        boat.move()
        boat.display(top)

        for light in sorted(lights, key = lambda light: -dist(light, boat)):
            light.seen_from(boat, view)

        cv2.imshow('View', vstack((view, top)))
        cv2.waitKey(1)
