#!/usr/bin/env python3

import cv2
import argparse
from balisage import Light, GPS, Boat, angle
from evdev import InputDevice, categorize, ecodes
import os
from numpy import vstack, median
import yaml
from threading import Thread

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.description = 'Balisage de nuit'
parser.add_argument('-p', '--pattern', type=str, help='Motif', default='')
parser.add_argument('-d', '--drift', type=float, help='Dérive (sud) due au vent en nœud', default=0)
parser.add_argument('-f', '--file', type=str, help='Configuration',default='zones/Cardinales.yaml')
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


class Listener:
    def __init__(self, on_press = None, on_release = None):

        self.on_press = on_press
        self.on_release = on_release

        # identify keyboard
        self.src = None
        inputs = '/dev/input/by-path'
        kbs = [dev for dev in os.listdir(inputs) if dev.endswith('-kbd')]

        self.threads = []

        for kb in kbs:
            self.threads.append(Thread(target=self.listen, args=(f'{inputs}/{kb}',)))

        for t in self.threads:
            t.start()

    def stop(self):
        self.src = 'stop'

    def listen(self, dev):

        keyboard_dev = InputDevice(dev)

        # stop listening to this one if another was used
        event = None
        while self.src in (None, dev):

            event = keyboard_dev.read_one()
            if event is None:
                continue

            if event.type != ecodes.EV_KEY:
                continue

            if event.value == 1:  # Key press, identify this keyboard
                if self.src is None:
                    self.src = dev
                    break
        else:
            return

        # main loop for source keyboard
        for event in keyboard_dev.read_loop():

            if self.src != dev:
                return

            if event.type != ecodes.EV_KEY or event.value == 2:
                continue

            key = categorize(event).keycode[4:]

            if event.value == 1:  # Key press
                if self.on_press is not None:
                    self.on_press(key)

            elif event.value == 0:   # release
                if self.on_release is not None:
                    self.on_release(key)


listener = Listener(on_press=boat.on_press, on_release=boat.on_release)

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

cv2.destroyAllWindows()
listener.stop()
