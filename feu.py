#!/usr/bin/env python3

import cv2
import argparse
from balisage import Light, GPS, Boat, angle
from pynput.keyboard import Listener
from numpy import vstack, median
import yaml

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.description = 'Balisage de nuit'
parser.add_argument('-p', '--pattern', type=str, help='Motif', default='')
parser.add_argument('-d', '--drift', type=float, help='Dérive (sud) due au vent en nœud', default=0)
parser.add_argument('-f', '--file', type=str, help='Configuration',default='zones/default.yaml')
parser.add_argument('-o', '--obs', type=float, help='Distance pour vitesse réduite',default=5.)
parser.add_argument('-r', '--reflexion', action='store_true', default = False)
parser.add_argument('-v', '--visi', type=float, default = 0.75, help='Pourcentage de la portée du feu où on le voit brillant')

args = parser.parse_args()

with open(args.file) as f:
    config = yaml.safe_load(f.read())

cv2.destroyAllWindows()

if args.pattern:
    config[args.pattern] = config.pop('Fl.3s')
    winname = args.pattern
else:
    import os.path
    winname = os.path.splitext(os.path.basename(args.file))[0]

cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
cv2.resizeWindow(winname, 1000, 700)
cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)
cv2.waitKey(1)

gps = GPS(config)
top_base = gps.image()

boat = config.pop('start')

lights = []
for pat, geom in config.items():
    if isinstance(geom, list):
        lights += [Light.build(pat, sub) for sub in geom]
    else:
        lights.append(Light.build(pat, geom))
Light.reflexion = args.reflexion
Light.visi = 1.-args.visi

# coord of all lights
center = median([light.c for light in lights], 0)

boat = Boat(boat, theta = angle(boat, center),
            drift = args.drift, obs = args.obs)

# write base top image
sectors = sum([light.sectors for light in lights],start=[])
sectors.sort(key = lambda s: -ord(s.color))

for sector in sectors:
    sector.write(top_base)
for sector in sectors:
    sector.write_borders(top_base)

for light in lights:
    light.display(top_base)

view_base = boat.image()

with Listener(on_press=boat.on_press, on_release=boat.on_release) as listener:

    while boat.running:

        top = top_base.copy()
        view = view_base.copy()
        boat.adapt_speed(lights)
        boat.move()
        boat.display(top, view)

        for light in lights:
            light.seen_from(boat, view)
        boat.draw_hull(view)

        cv2.imshow(winname, vstack((view, top)))
        cv2.waitKey(1)
