import pygame
import math
pygame.init()
screen = pygame.display.set_mode((500,500))
running = True
velx = 32
vely = 2
tempx = velx
tempy = vely
x = 40
y = 250
velx1 = -1
vely1 = -10
tempx1 = velx1
tempy1 = vely1
x1 = 480
y1 = 250
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((0,0,0))
    pygame.draw.circle(screen,(255,255,0),(x,y),40)
    pygame.draw.circle(screen,(255,0,255),(x1,y1),20)
    pygame.display.update()
    x += velx/100
    y += vely/100
    x1 += velx1/100
    y1 += vely1/100
    tempx1 = velx1
    tempx = velx
    tempy1 = vely1
    tempy = vely
    if abs(abs(abs(x1) - abs(x))+abs(abs(y1) - abs(y)))<=80:
        velx = tempx1
        velx1 = tempx
        vely = tempy1
        vely1 = tempy
    
    
    if x1 > 480:
        velx1 = -velx1
    if x1 < 20:
        velx1 = -velx1
    
    if y1 > 480:
        vely1 = -vely1
    if y1 < 20:
        vely1 = -vely1
    

    if x > 460:
        velx = -velx
    if x < 40:
        velx = -velx
    
    if y > 460:
        vely = -vely
    if y < 40:
        vely = -vely