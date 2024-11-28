import cv2
import numpy as np
from pynput.keyboard import Key
from scipy import ndimage
from time import time, sleep

dt = 0.025
sec = int(1/dt)

# target image
W = 1000
H = int(400*W/800)
view_h = W//4
hor = 3*view_h//4

# sector margin to blurr
margin = 2*np.pi/180


def load_im(filename, target_height):
    im = cv2.imread(f'images/{filename}', -1)
    Y,X = im.shape[:2]
    ratio = target_height/Y
    im = cv2.resize(im, (int(X*ratio),int(Y*ratio)))
    return im


# Boat sprite
bto = load_im('bto.png', W/20)

# Light sprite
star = load_im('feu.png',30)


def paste(src, dst, pos, rotation = 0.):
    rotated = ndimage.rotate(src, -rotation*180/np.pi, reshape=True)

    x, y = pos

    x1 = int(x)-rotated.shape[1]//2
    y1 = int(y)-rotated.shape[0]//2
    x2 = x1 + rotated.shape[1]
    y2 = y1 + rotated.shape[0]

    x0 = 0
    y0 = 0
    yf,xf = rotated.shape[:2]

    if x1 < 0:
        x0 -= x1
        x1 = 0
    if x2 >= dst.shape[1]:
        xf -= x2-dst.shape[1]
        x2 = dst.shape[1]
    if y1 < 0:
        y0 -= y1
        y1 = 0
    if y2 >= dst.shape[0]:
        yf -= y2-dst.shape[0]
        y2 = dst.shape[0]

    mask = rotated[y0:yf, x0:xf, 3]/255

    try:
        for c in range(3):
            dst[y1:y2, x1:x2,c] = mask*rotated[y0:yf, x0:xf, c]/255 + (1-mask)*dst[y1:y2, x1:x2,c]
    except ValueError:
        return


def degrees(wsg):
    # lat,lon = wsg.split(',')

    def read(coord):
        coord, axis = coord.rsplit(' ',1)
        numbers = ''
        for c in coord:
            if c.isdigit():
                numbers += c
            elif c in '.,':
                numbers += '.'
            else:
                numbers += ' '
        deg = sum(c/60**i for i,c in enumerate(map(float,numbers.split())))
        if axis.strip() in ('N','E'):
            return deg
        return -deg
    return list(map(read, wsg.split(',')))


def to_pi(a, positive = False):

    if positive:
        return to_pi(a-np.pi) + np.pi
    return (a+np.pi) % (2*np.pi) - np.pi


def parse_coord(config, miles = None):

    coordinates = []
    cur = 0
    for key in config:
        if isinstance(config[key], dict):
            for sub in config[key]:
                if miles is None:
                    coordinates.append(degrees(config[key][sub]))
                else:
                    config[key][sub] = miles[cur]
                    cur += 1
        elif isinstance(config[key], list):
            for i in range(len(config[key])):
                if miles is None:
                    coordinates.append(degrees(config[key][i]))
                else:
                    config[key][i] = miles[cur]
                    cur += 1
        elif miles is None:
            coordinates.append(degrees(config[key]))
        else:
            config[key] = miles[cur]
            cur += 1
    return coordinates


class GPS:

    gps = None

    def __init__(self, config):

        GPS.gps = self

        # gather all coordinates
        coordinates = np.array(parse_coord(config))

        # convert to nautical miles
        center = .5*(np.amax(coordinates,0) + np.amin(coordinates,0))

        dx = 1.
        to_rad = np.pi/180

        # TODO improve this one, use proper WSG84
        dy = 6366/1.852*2*np.arcsin(abs(np.cos(center[0]*to_rad)*(np.sin(to_rad/2))))
        lat = center[0]*to_rad
        dy = np.arccos(np.sin(lat)**2+np.cos(lat)**2*np.cos(to_rad))*6366/1.852

        dy = np.cos(center[0]*to_rad)*1.6

        dxy = np.array([[dx,dy]])/60.

        miles = (coordinates-center)/dxy
        parse_coord(config, miles)

        # get span
        dx,dy = np.amax(miles,0) - np.amin(miles,0)
        dx = max(dx,.5)
        dy = max(dy,.5)

        # target image 800x600

        if dx/dy < H/W:
            # increase lat span
            dx = H/W*dy
        else:
            # increase long span
            dy = W/H*dx

        dx *= 1.1
        dy *= 1.1

        self.K = np.array([-H/dx, W/dy])
        self.P0 = np.array([H/2, W/2])

        if 'start' not in config:
            config['start'] = [0.,0.]

    def image(self):
        return np.zeros((H,W,3))

    def pixels(self, p):
        px = np.flip((self.P0 + self.K*p).astype(np.int32).reshape(-1,2),1)
        return px


def dist(o1, o2):
    return np.linalg.norm(o1.c-o2.c)


def angle(o,x):
    return np.angle(np.dot(x-o,np.array([1,1j])))


def bgr(color, v = 1.):
    bgr = {'W': [1.,1.,1.], 'R': [0,0,1.], 'G': [0,1.,0], 'Y': [0,1.,1.], 'B': [.3,0.,0.]}
    return np.array([float(v*c) for c in bgr[color]])


class Boat:

    def __init__(self, start, drift = 0., obs = 1.):
        self.vtarget = 200/min(abs(GPS.gps.K))
        self.nearest = 100.
        self.vx = 0.
        self.vy = 0.
        self.drift = drift
        self.w = 0.
        self.vc = 1.
        self.obs = obs
        self.c = start
        self.theta = 0.
        self.t0 = None
        self.fwd = 0.

    def image(self):

        im = np.zeros((view_h,W,3))
        im[:hor] = bgr('B')

        # for x in (W//4,3*W//4):
        #     for y in range(0,view_h):
        #         if y % 10 < 5:
        #             cv2.circle(im, [x,y],1,[1.,1.,1.])
        return im

    def draw_hull(self, im):
        mid = W//2
        w = W//20
        if self.fwd:
            rad = int(2*view_h)
            cv2.circle(im, [mid, view_h-W//30+rad], rad, [.5,.5,.5], -1)
        else:
            cv2.fillPoly(im, [np.array([[mid, view_h-W//16],
                                        [mid-w,view_h-3],[mid+w,view_h-3]])],[.5,.5,.5])
        im[-2:] = [1.,1.,1.]

    def on_press(self,key: Key):

        w = 1.

        if key == Key.up:
            self.vx = self.vtarget
        elif key == Key.down:
            self.vx = -self.vtarget
        elif key == Key.left:
            self.w = -w
        elif key == Key.right:
            self.w = w
        elif key == Key.ctrl_r:
            self.fwd = np.pi-self.fwd
        if not hasattr(key, 'char'):
            return
        if key.char == '[':
            self.vy = -self.vtarget/10
        elif key.char == ']':
            self.vy = self.vtarget/10

    def on_release(self,key):
        if key in (Key.up, Key.down):
            self.vx = 0.
        elif key in (Key.left, Key.right):
            self.w = 0.
        try:
            if hasattr(key, 'char') and key.char in '[]':
                self.vy = 0
        except TypeError:
            return

    def move(self):

        if self.t0 is None:
            self.t0 = time()
        elapsed = time() - self.t0
        if elapsed < dt:
            sleep(dt - elapsed)
        self.t0 = time()

        c,s = np.cos(self.theta), np.sin(self.theta)
        R = np.array([[c,-s],[s,c]])
        self.c += np.dot(R,[[self.vx*self.vc],[self.vy*self.vc]]).flatten() * dt
        self.c[0] -= self.drift*self.vc*dt
        self.theta += self.w*np.sqrt(self.vc)*dt

    def display(self, top, view):
        paste(bto, top, GPS.gps.pixels(self.c)[0], self.theta)

        cv2.putText(view, f'{self.vtarget*self.vc:.02f} kn',
                    [W//10,int(.95*view_h)], cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [1.,1.,1.], 2, cv2.LINE_AA, False)

        cv2.putText(view, f'Cap {to_pi(self.theta, True)*180/np.pi:.01f}',
                    [3*W//4,int(.95*view_h)], cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [1.,1.,1.], 2, cv2.LINE_AA, False)

    def adapt_speed(self, lights):

        for light in lights:
            light.boat = {'d': dist(light, self), 'a': to_pi(angle(light.c, self.c), True)}

        lights.sort(key = lambda light: -light.boat['d'])

        c = 1.
        cmin = .1
        self.nearest = 100.
        for light in lights:
            d = light.boat['d']
            self.nearest = min(self.nearest, d)
            if d < self.obs:
                c = min(c, np.sqrt(d/self.obs))
                # speed down wrt nearest sector
                if light.sectors:
                    c = min(c, min([abs(sector.rel(light.boat['a'])) for sector in light.sectors])/margin)
            if c < cmin:
                break

        self.vc = .5*(self.vc + max(c, cmin))


class Sector:

    def __init__(self, center, color, start, end, height):
        self.c = center
        self.color = color
        self.rng = np.clip(height, 3, 10)

        self.start = to_pi(start, True)
        self.span = to_pi(end - start, True)

    def write(self, im, start = None, span = None):

        if self.color == 'B':
            return

        if start is None or span is None:
            start = self.start
            span = self.span

        steps = max(2,int(abs(span)*180/np.pi))
        points = np.vstack([self.c] + [self.c + self.rng*np.array([np.cos(a),np.sin(a)])
                                        for a in np.linspace(start, start+span, steps)])

        cv2.fillPoly(im, [GPS.gps.pixels(points)], bgr(self.color,.5))

    def write_borders(self, im):

        if self.color == 'W':
            self.write(im)
            return

        span = 1.
        for a,da in ((self.start, 1), (self.start+self.span, -1)):
            self.write(im, a, da*span*np.pi/180)

    def rel(self, a):
        return to_pi(a - self.start)


def parse_pattern(pat):
    if pat.startswith('N'):
        pat = 'Q'
    elif pat.startswith('E'):
        pat = 'Q(3).15s'
    elif pat.startswith('S'):
        pat = 'Q(6)+LFl.15s'
    elif pat.startswith('W'):
        pat = 'Q(9).15s'

    # default values
    meta = {'m': 3., 's': 4., 'M': 5.}
    if '.' in pat:
        pat,info = pat.split('.')
        avail = ''.join(c for c in info if c.isalpha())
        for c in 'smM':
            if c in avail:
                idx = info.index(c)
                meta[c] = float(info[:idx].replace(',','.'))
                info = info[idx+1:]

    colors = ''.join([c for c in pat if c in 'WRGYB'])
    if not colors:
        colors = 'W'
    pat = pat.replace(colors, '').strip('.')

    offset = hash((pat, meta['s']))

    if len(colors) == 2:
        other = ''.join([c for c in colors if c != 'W'])
        colors = f'{other}W{other}'
    elif len(colors) == 3:
        colors = 'RWG'
    if colors == 'B':
        pat = 'Iso'
    return pat, colors, meta, offset


class Light:

    visi = 0.75

    @staticmethod
    def build(pat, geom):

        if pat.startswith('Oc'):
            ret = Light(pat.replace('Oc','Fl'), geom)
            ret.on = [not v for v in ret.on]
            return ret
        return Light(pat, geom)

    def __init__(self, pat, geom):

        pat, self.colors, meta, offset = parse_pattern(pat)
        self.height = meta['m']
        self.rng = meta['M']
        self.on = [False for _ in range(int(meta['s']/dt))]
        self.cur = offset % len(self.on)
        # parse sectors
        self.sectors = []
        if isinstance(geom, dict):
            self.c = geom['pos']

            sectors = sorted([(angle(self.c,point), color[0]) for color,point in geom.items() if color != 'pos'])
            for i, (start, color) in enumerate(sectors):
                end = sectors[(i+1) % len(sectors)][0]
                self.sectors.append(Sector(self.c, color, start, end, meta['M']))
        else:
            self.c = geom

        # no need for time pattern
        if pat == 'Iso':
            self.fill(meta['s']/2, meta['s'])
            return

        bonus = None
        other = None
        dur = {'Q': .5, 'VQ': .25, 'Fl': 1., 'LFl': 2.5}

        if '(' not in pat:
            if 'Q' in pat:
                self.cur = -1
                n = meta['s'] // dur[pat]
            else:
                n = 1
        else:
            pat,other = pat.split(')')
            pat,n = pat.split('(')
            if '+' in n:
                n,n2 = n.split('+')
                bonus = int(n2)
        end = int(n)*2*dur[pat]

        self.fill(dur[pat],end)

        # wait 2 sec and add last one
        if bonus:
            start = int((end+2.)/dt)
            for i in range(int(dur[pat]/dt)):
                self.on[start+i] = True
            end += 2.+dur[pat]

        if other:
            # LFl
            other = other.strip('+')
            start = int(end/dt)
            for i in range(int(dur[other]/dt)):
                self.on[start+i] = True

    def fill(self, lit, end):

        end = int(end/dt)
        cycle = int(lit/dt)
        whole = 2*cycle

        for i in range(len(self.on)):

            if i == end:
                return

            where = i % whole
            if where < cycle:
                self.on[i] = True

    def display(self, im):

        if self.sectors:
            paste(star, im, GPS.gps.pixels(self.c)[0])
        else:
            cv2.circle(im, GPS.gps.pixels(self.c)[0], 5, bgr(self.colors[0]),-1)

    def seen_from(self, boat, im):

        # update time step
        self.cur = (self.cur+1) % len(self.on)

        a = to_pi(self.boat['a']+np.pi - (boat.theta+boat.fwd))
        if abs(a) > np.pi/2:
            return

        d = self.boat['d']
        x = int(a*W/np.pi + W/2) % W

        # seen height
        h = 30*np.log((self.height-2.)/d)

        y = int(hor - np.clip(h, 2, 2*hor/3))
        rad = max(2, min(6,int(2*self.height/d)))
        wtop = int(.6*rad)
        wbot = int(1.5*rad)
        # display pole anyway
        pole = np.array([[x+wtop,y],[x-wtop,y],[x-wbot,hor],[x+wbot,hor]])
        if not self.on[self.cur] or d > self.rng:
            cv2.fillPoly(im, [pole], [0,0,0])
            return

        color = bgr(self.colors[0])
        if self.sectors:
            # find nearest sector border
            a = self.boat['a']
            start = np.argmin([abs(sector.rel(a)) for sector in self.sectors])
            rel = np.clip((self.sectors[start].rel(a)/margin+1)/2, 0,1)
            # blend corresponding color
            cn = bgr(self.sectors[start].color)
            cp = bgr(self.sectors[start-1].color)
            color = rel*cn + (1-rel)*cp

        fade = min(.2, d/5)  # .2*(self.rng-d)/self.rng

        if (color != bgr('B')).any():
            cv2.fillPoly(im, [pole], fade**2*color)
            visi = min(1, (1-d/self.rng)/self.visi)
            cv2.circle(im, [x,y], rad, visi*color, -1)
            if Light.reflexion and d < 1.:
                cv2.line(im, [x,hor],[W//2,2*view_h],fade*color,2)
        else:
            cv2.fillPoly(im, [pole], [0,0,0])

