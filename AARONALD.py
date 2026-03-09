#!/usr/bin/env python3
"""
AARONALD

Inspired by Harold Cohen's AARON.
Generates 32x32 pixel art scenes and a poetic caption.
 Press 'y' to generate and display, 'n' to quit.
 
 (sorry its a bit rushed)
"""

import random
import sys
import colorama
colorama.init()

# ------------------------------------------------------------
# ansi helpers

def set_fg(r, g, b):
    return f'\033[38;2;{r};{g};{b}m'

def set_bg(r, g, b):
    return f'\033[48;2;{r};{g};{b}m'

RESET = '\033[0m'

# ------------------------------------------------------------
# palette

palette = {
    'sand': (230, 200, 170, 255),
    'grass': (120, 150, 100, 255),
    'autumnGrass': (170, 140, 70, 255),
    'winterGround': (220, 230, 240, 255),
    'skyDay': (180, 210, 240, 255),
    'skySunset1': (250, 190, 140, 255),
    'skySunset2': (200, 120, 160, 255),
    'skyNight': (30, 40, 70, 255),
    'darkSoil': (100, 80, 60, 255),
    'plantGreen': (90, 130, 70, 255),
    'autumnLeaf': (200, 120, 40, 255),
    'potBrown': (140, 100, 70, 255),
    'skin': (230, 190, 150, 255),
    'animalFur': (170, 130, 90, 255),
    'catOrange': (210, 140, 70, 255),
    'catBlack': (40, 40, 40, 255),
    'catWhite': (240, 240, 240, 255),
    'catCalico1': (210, 140, 70, 255),   # orange
    'catCalico2': (40, 40, 40, 255),     # black
    'catCalico3': (240, 240, 240, 255),  # white
    'dogBrown': (150, 100, 70, 255),
    'dogBlack': (50, 50, 50, 255),
    'dogTan': (200, 160, 120, 255),
    'dogWhite': (250, 250, 240, 255),
    'birdYellow': (230, 200, 50, 255),
    'rockGrey': (130, 120, 110, 255),
    'treeTrunk': (120, 80, 50, 255),
    'leafGreen': (70, 120, 60, 255),
    'evergreen': (40, 80, 50, 255),
    'roofRed': (180, 80, 60, 255),
    'wallLight': (220, 200, 160, 255),
    'windowBlue': (130, 180, 210, 255),
    'white': (245, 245, 235, 255),
    'black': (30, 30, 30, 255),
    'starColor': (255, 255, 220, 255),
    'cloudLight': (240, 240, 250, 0),    # alpha dynamic
    'cloudDark': (210, 210, 230, 0),
    'sunColor': (255, 230, 150, 255),
    'moonColor': (245, 245, 210, 255),
    'carBody': (200, 50, 50, 255),
    'carWheel': (40, 40, 40, 255),
}


def rand_float(a=0.0, b=1.0):
    return random.random() * (b - a) + a

def rand_int(a, b):
    return random.randint(a, b)

def choice(seq):
    return random.choice(seq)

def indefinite_article(word):
    if not word:
        return 'a'
    first = word[0].lower()
    return 'an' if first in 'aeiou' else 'a'

def combine_phrases(phrases):
    if not phrases:
        return ''
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return f"{phrases[0]} and {phrases[1]}"
    return f"{', '.join(phrases[:-1])}, and {phrases[-1]}"

# ------------------------------------------------------------
# drawing primitives (modify img in place)

def set_pixel(img, x, y, col):
    if 0 <= x < 32 and 0 <= y < 32:
        img[y][x] = col

def draw_circle(img, cx, cy, r, col):
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if dx*dx + dy*dy <= r*r + 0.5:
                x, y = cx + dx, cy + dy
                if 0 <= x < 32 and 0 <= y < 32:
                    img[y][x] = col

def fill_rect(img, x1, y1, x2, y2, col):
    for y in range(max(0, y1), min(32, y2+1)):
        for x in range(max(0, x1), min(32, x2+1)):
            img[y][x] = col

def h_line(img, y, x1, x2, col):
    if 0 <= y < 32:
        for x in range(max(0, x1), min(32, x2+1)):
            img[y][x] = col

def v_line(img, x, y1, y2, col):
    if 0 <= x < 32:
        for y in range(max(0, y1), min(32, y2+1)):
            img[y][x] = col

def draw_shadow(img, x, y, w, h, dark_factor):
    # darken pixels below and to the right
    for dy in range(1, 3):
        for dx in range(1, w+1):
            sx, sy = x + dx, y + h - 1 + dy
            if 0 <= sx < 32 and 0 <= sy < 32:
                orig = img[sy][sx]
                if orig:
                    img[sy][sx] = (
                        max(0, orig[0] - 40),
                        max(0, orig[1] - 40),
                        max(0, orig[2] - 40),
                        255
                    )

# ------------------------------------------------------------
# scene generator

def generate_scene():
    
    seasons = ['summer', 'autumn', 'winter', 'spring']
    season = choice(seasons)
    is_autumn = (season == 'autumn')
    is_winter = (season == 'winter')

    # time of day
    times = [
        {'name': 'morning', 'light': 1.0, 'shadowDark': 0.6},
        {'name': 'noon', 'light': 1.2, 'shadowDark': 0.5},
        {'name': 'evening', 'light': 0.9, 'shadowDark': 0.7},
        {'name': 'night', 'light': 0.5, 'shadowDark': 0.9}
    ]
    time_idx = rand_int(0, len(times)-1)
    time = times[time_idx]
    is_night = (time['name'] == 'night')
    has_sun = (not is_night) and (random.random() < 0.5)
    has_moon = (is_night or random.random() < 0.3) and not has_sun
    moon_phase = choice(['new', 'crescent', 'quarter', 'gibbous', 'full']) if has_moon else None
    has_stars = is_night and random.random() < 0.8
    has_clouds = random.random() < 0.4

    # ground horizon
    horizon = rand_int(14, 22)

    # ground colour
    if is_winter:
        ground_col = palette['winterGround']
    elif is_autumn:
        ground_col = palette['autumnGrass']
    else:
        ground_col = choice([palette['grass'], palette['sand'], (180, 160, 140, 255)])

    # sky color
    if is_night:
        base_sky = palette['skyNight']
    elif time['name'] == 'morning':
        base_sky = (210, 220, 235, 255)
    else:
        base_sky = palette['skyDay']

    # init image
    img = [[[0, 0, 0, 255] for _ in range(32)] for _ in range(32)]
    for y in range(32):
        for x in range(32):
            if y < horizon:
                img[y][x] = base_sky
            else:
                img[y][x] = ground_col

    # sky objects
    if has_sun:
        sun_x = rand_int(5, 27)
        sun_y = rand_int(2, horizon-4)
        draw_circle(img, sun_x, sun_y, 3, palette['sunColor'])
        for _ in range(4):
            dx, dy = rand_int(-2, 2), rand_int(-2, 2)
            if dx != 0 or dy != 0:
                set_pixel(img, sun_x + dx*2, sun_y + dy*2, palette['sunColor'])

    if has_moon:
        moon_x = rand_int(5, 27)
        moon_y = rand_int(2, horizon-4)
        draw_circle(img, moon_x, moon_y, 3, palette['moonColor'])
        if moon_phase == 'crescent':
            for dy in range(-2, 3):
                for dx in range(0, 4):
                    x, y = moon_x + dx, moon_y + dy
                    if 0 <= x < 32 and 0 <= y < 32 and (dx-1.5)**2 + dy**2 <= 2.5**2:
                        if random.random() > 0.3:
                            img[y][x] = base_sky
        elif moon_phase == 'quarter':
            for dy in range(-2, 3):
                for dx in range(0, 3):
                    x, y = moon_x + dx, moon_y + dy
                    if 0 <= x < 32 and 0 <= y < 32:
                        img[y][x] = base_sky
        elif moon_phase == 'gibbous':
            for dy in range(-1, 2):
                for dx in range(2, 4):
                    x, y = moon_x + dx, moon_y + dy
                    if 0 <= x < 32 and 0 <= y < 32:
                        img[y][x] = base_sky

    if has_stars:
        for _ in range(rand_int(5, 15)):
            sx, sy = rand_int(0, 31), rand_int(0, horizon-2)
            bright = rand_int(200, 255)
            set_pixel(img, sx, sy, (bright, bright, bright, 255))
            if random.random() > 0.8:
                set_pixel(img, sx+1, sy, (bright-30, bright-30, bright-30, 255))

    # clouds
    if has_clouds:
        for _ in range(rand_int(1, 3)):
            cx = rand_int(5, 27)
            cy = rand_int(2, horizon-5)
            alpha = rand_int(150, 255)
            col_light = (240, 240, 250, alpha)
            col_dark = (210, 210, 230, alpha)
            col = choice([col_light, col_dark])
            draw_circle(img, cx, cy, 3, col)
            draw_circle(img, cx-2, cy-1, 2, col)
            draw_circle(img, cx+2, cy, 2, col)

    # ground elements flags
    has_plant = random.random() < 0.5
    person_count = rand_int(0, 2) if random.random() < 0.4 else 0  # 0,1,2
    animal_count = rand_int(1, 2) if random.random() < 0.5 else 0
    animal_types = []
    animal_colors = []
    heterochromia = False

    for _ in range(animal_count):
        typ = choice(['cat', 'dog', 'bird'])
        animal_types.append(typ)
        if typ == 'cat':
            col = choice(['orange', 'black', 'calico'])
            animal_colors.append(col)
        elif typ == 'dog':
            col = choice(['brown', 'black', 'tan', 'white'])
            animal_colors.append(col)
        else:
            animal_colors.append('yellow')  # bird

    # heterochromia 
    if 'cat' in animal_types and random.random() < 0.1:
        heterochromia = True

    has_tree = random.random() < 0.3
    has_rock = random.random() < 0.4
    has_house = random.random() < 0.2
    has_fence = random.random() < 0.25
    has_frame = random.random() < 0.3
    has_car = random.random() < 0.15

    # rock
    if has_rock:
        for _ in range(rand_int(1, 3)):
            x = rand_int(4, 28)
            y = rand_int(horizon, 28)
            size = rand_int(2, 4)
            for _ in range(size):
                draw_circle(img, x + rand_int(-1, 1), y + rand_int(-1, 1), size-1, palette['rockGrey'])
            draw_shadow(img, x-1, y-1, size, size, time['shadowDark'])

    # tree
    if has_tree:
        trunk_x = rand_int(8, 24)
        trunk_y = rand_int(horizon, 26)
        trunk_h = rand_int(4, 7)
        fill_rect(img, trunk_x-1, trunk_y-trunk_h, trunk_x+1, trunk_y, palette['treeTrunk'])
        if is_winter and random.random() < 0.7:
            for _ in range(3):
                bx = trunk_x + rand_int(-2, 2)
                by = trunk_y - trunk_h - rand_int(1, 3)
                set_pixel(img, bx, by, (100, 70, 40, 255))
        else:
            leaf_col = palette['autumnLeaf'] if is_autumn else palette['leafGreen']
            draw_circle(img, trunk_x, trunk_y - trunk_h - 2, 4, leaf_col)
            if is_winter and random.random() < 0.3:
                draw_circle(img, trunk_x, trunk_y - trunk_h - 2, 4, palette['evergreen'])
        draw_shadow(img, trunk_x-3, trunk_y-trunk_h-4, 6, 2, time['shadowDark'])

    # house
    if has_house:
        hx = rand_int(6, 22)
        hy = rand_int(horizon, 26)
        w = rand_int(5, 8)
        h = rand_int(4, 6)
        fill_rect(img, hx, hy-w, hx+w, hy, palette['wallLight'])
        for i in range(w):
            yoffs = i // 2
            v_line(img, hx+i, hy-w-yoffs, hy-w, palette['roofRed'])
        fill_rect(img, hx+2, hy-4, hx+3, hy-3, palette['windowBlue'])
        draw_shadow(img, hx, hy-h, w, h, time['shadowDark'])

    # fence
    if has_fence:
        fy = rand_int(horizon+1, horizon+4)
        for fx in range(2, 30, 5):
            v_line(img, fx, fy-2, fy+2, (160, 130, 100, 255))
        h_line(img, fy-1, 0, 31, (140, 110, 80, 255))

    # plant
    if has_plant:
        px = rand_int(6, 26)
        py = rand_int(horizon+2, 28)
        fill_rect(img, px-2, py, px+2, py+3, palette['potBrown'])
        draw_circle(img, px, py-2, 3, palette['plantGreen'])
        draw_circle(img, px-1, py-3, 2, palette['plantGreen'])
        draw_circle(img, px+2, py-3, 2, palette['plantGreen'])
        draw_shadow(img, px-3, py-2, 5, 4, time['shadowDark'])

    # car
    if has_car:
        cx = rand_int(6, 24)
        cy = rand_int(horizon+2, 28)
        # random car colour
        col = (rand_int(100, 230), rand_int(100, 200), rand_int(100, 200), 255)
        fill_rect(img, cx-3, cy-2, cx+3, cy, col)
        draw_circle(img, cx-2, cy+1, 2, palette['carWheel'])
        draw_circle(img, cx+2, cy+1, 2, palette['carWheel'])
        draw_shadow(img, cx-3, cy-2, 6, 2, time['shadowDark'])

    # animal
    for i in range(animal_count):
        ax = rand_int(8, 24) + i*2
        ay = rand_int(horizon+2, 28)
        typ = animal_types[i]
        color = animal_colors[i]
        if typ == 'cat':
            if color == 'orange':
                draw_circle(img, ax, ay-2, 3, palette['catOrange'])
                fill_rect(img, ax-2, ay, ax+2, ay+3, palette['catOrange'])
            elif color == 'black':
                draw_circle(img, ax, ay-2, 3, palette['catBlack'])
                fill_rect(img, ax-2, ay, ax+2, ay+3, palette['catBlack'])
            elif color == 'calico':
                for dy in range(-2, 4):
                    for dx in range(-2, 3):
                        x, y = ax+dx, ay+dy
                        if 0 <= x < 32 and 0 <= y < 32:
                            rnd = random.random()
                            if rnd < 0.33:
                                set_pixel(img, x, y, palette['catCalico1'])
                            elif rnd < 0.66:
                                set_pixel(img, x, y, palette['catCalico2'])
                            else:
                                set_pixel(img, x, y, palette['catCalico3'])
            # eye
            eye_y = ay-3
            left_x, right_x = ax-1, ax+1 
            if heterochromia:
                set_pixel(img, left_x, eye_y, (100, 150, 255, 255))   # blue
                set_pixel(img, right_x, eye_y, (100, 255, 100, 255))  # green
            else:
                eye_col = choice([
                    (100, 150, 255, 255),  # blue
                    (255, 255, 100, 255),  # yellow
                    (100, 255, 100, 255)   # green
                ])
                set_pixel(img, left_x, eye_y, eye_col)
                set_pixel(img, right_x, eye_y, eye_col)
            draw_shadow(img, ax-3, ay-2, 5, 5, time['shadowDark'])
        elif typ == 'dog':
            if color == 'brown':
                dogcol = palette['dogBrown']
            elif color == 'black':
                dogcol = palette['dogBlack']
            elif color == 'tan':
                dogcol = palette['dogTan']
            else:
                dogcol = palette['dogWhite']
            draw_circle(img, ax, ay-2, 3, dogcol)
            fill_rect(img, ax-2, ay, ax+2, ay+4, dogcol)
            set_pixel(img, ax-1, ay-3, palette['black'])
            set_pixel(img, ax+1, ay-3, palette['black'])
            draw_shadow(img, ax-2, ay-1, 4, 5, time['shadowDark'])
        elif typ == 'bird':
            draw_circle(img, ax, ay-1, 2, palette['birdYellow'])
            set_pixel(img, ax+2, ay-2, (255, 140, 0, 255))
            fill_rect(img, ax-1, ay-1, ax, ay, (200, 170, 0, 255))
            draw_shadow(img, ax-2, ay-2, 3, 2, time['shadowDark'])

    # people
    for p in range(person_count):
        px = rand_int(8, 24) + p*3
        py = rand_int(horizon+2, 28)
        draw_circle(img, px, py-4, 2, palette['skin'])
        shirt = (rand_int(80, 220), rand_int(80, 200), rand_int(80, 220), 255)
        v_line(img, px, py-2, py+2, shirt)
        h_line(img, py-1, px-2, px+2, shirt)
        pants = (rand_int(40, 150), rand_int(40, 150), rand_int(40, 150), 255)
        v_line(img, px-1, py+3, py+5, pants)
        v_line(img, px+1, py+3, py+5, pants)
        draw_shadow(img, px-2, py-4, 4, 8, time['shadowDark'])

    # frame
    if has_frame:
        frame_col = choice([
            (200, 120, 100, 255),
            (100, 150, 180, 255),
            (180, 150, 80, 255)
        ])
        for x in range(32):
            for y in range(2):
                img[y][x] = frame_col
                img[31-y][x] = frame_col
        for y in range(32):
            img[y][0] = frame_col
            img[y][1] = frame_col
            img[y][30] = frame_col
            img[y][31] = frame_col

    # build caption =================
    ground_phrases = []

    if person_count == 1:
        ground_phrases.append('a person')
    elif person_count == 2:
        ground_phrases.append('two figures')

    if animal_count > 0:
        animal_phrases = []
        for i in range(animal_count):
            typ = animal_types[i]
            col = animal_colors[i]
            if typ == 'cat' and col == 'calico':
                animal_phrases.append('a calico cat')
            elif typ == 'cat' and col == 'orange':
                animal_phrases.append('an orange cat')
            elif typ == 'cat' and col == 'black':
                animal_phrases.append('a black cat')
            elif typ == 'dog':
                animal_phrases.append(f'a {col} dog')
            else:
                animal_phrases.append('a bird')
        animal_str = combine_phrases(animal_phrases)
        if person_count > 0:
            ground_phrases.append(f'with {animal_str}')
        else:
            ground_phrases.append(animal_str)

    if heterochromia and 'cat' in animal_types:
        ground_phrases.append('with heterochromia')

    if has_plant:
        ground_phrases.append('a plant')
    if has_tree:
        if is_autumn:
            ground_phrases.append('an autumn tree')
        elif is_winter:
            ground_phrases.append('a winter tree')
        else:
            ground_phrases.append('a tree')
    if has_house:
        ground_phrases.append('a house')
    if has_rock:
        ground_phrases.append('rocks')
    if has_fence:
        ground_phrases.append('a fence')
    if has_car:
        ground_phrases.append('a car')

    # sky 
    sky_desc = []
    if has_sun:
        sky_desc.append('sunny')
    if has_moon:
        sky_desc.append(f'{moon_phase} moon')
    if has_stars:
        sky_desc.append('starry')
    if has_clouds:
        sky_desc.append('cloudy')
    if not sky_desc:
        sky_desc.append(time['name'])

    sky_phrase = combine_phrases(sky_desc)
    if not (sky_phrase.endswith('night') or sky_phrase.endswith('moon') or sky_phrase.endswith('sky')):
        sky_phrase += ' sky'

    article = indefinite_article(sky_phrase.split()[0])

    if not ground_phrases:
        caption = f"A serene landscape under {article} {sky_phrase}"
    else:
        ground_part = combine_phrases(ground_phrases)
        ground_part = ground_part[0].upper() + ground_part[1:]
        caption = f"{ground_part} under {article} {sky_phrase}"

    caption = ' '.join(caption.split())
    return img, caption

# ------------------------------------------------------------
# Terminal display

def print_image(img):
    # 32x32 image using ANSI truecolor.
    for y in range(32):
        line = ''
        for x in range(32):
            r, g, b, a = img[y][x]
            # background color with two spaces for squareish pixels
            line += f'{set_bg(r, g, b)}  '
        line += RESET
        print(line)

# ------------------------------------------------------------
# Main loop

def main():
    print("AARONALD")
    print("Press 'y' to generate an image, 'n' to quit.\n")
    while True:
        try:
            inp = input("Generate? (y/n): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if inp.startswith('n'):
            print("Exiting.")
            break
        elif inp.startswith('y'):
            img, caption = generate_scene()
            print("\n" + caption)
            print_image(img)
            print()  # blank line
        else:
            print("Please enter 'y' or 'n'.")

if __name__ == "__main__":
    main()