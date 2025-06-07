import pygame
import math
pygame.init()
pygame.font.init()
font = pygame.font.SysFont('Arial', 30)
screen = pygame.display.set_mode((512,512))
x = 0
y = 0
t = 0
xm = 0
ym = 0
xv = 0
yv = 0
running = True
while running:
    dist = abs(abs(abs(xm) - abs(x)) + abs(abs(ym) - abs(y)))
    sdist = int(round(dist))
    


    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((0,0,0))
    pygame.draw.circle(screen,(255,127,63),(x,y),10)
    # line
    x = x + xv/5
    y = y + yv/5
    yv += 0.01
    pygame.draw.line(screen,(63,127,255),(x,y),(xm,ym),int(dist/10))
    
    # dist
    text = font.render(str(int(round(dist))), 0, (255, 255, 255))
    csad = pygame.transform.scale(text,(math.sqrt(sdist/20)*10+64,math.sqrt(sdist/20)*10+32))
    rectfont = text.get_rect()
    rectfont.topleft = (0, 0)
    screen.blit(csad, rectfont)
    
    # dont touch
    pygame.display.update()
    t += 1
    xm, ym = pygame.mouse.get_pos()
    if t > 100:
        if x > xm:
            xv += (xm - x)/100
        else :
            xv += (xm - x)/100

        if y > ym:
            yv += (ym - y)/100
        else :
            yv += (ym - y)/100
        t = 0
        xv = xv / 1.1
        yv = yv / 1.1