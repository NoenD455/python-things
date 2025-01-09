import pygame
import random
import math
pygame.init 
screen = pygame.display.set_mode((500,500))
posy = 0
posx = 0
running = True
col = 0
boardr = [random.randint(0, 100) for _ in range(1024)]
boardg = [random.randint(0, 100) for _ in range(1024)]
boardb = [random.randint(0, 100) for _ in range(1024)]
read = 0
cursorx = 0
cursory = 0
cursorposx = 0
cursorposy = 0

while True :
    col = col%255 + 1
    if posx > 499 :
        posx = 0
        posy = posy + 15.625
    if posy > 500 :
        posy = 0
        posx = 0
        read = 0    
    posx = posx + 15.625
    read = read + 1
    if read > 1023 :
        read = 0
        boardg[round(cursory/32)+33] = 255
        boardb[round(cursorx/32)+32] = 255
    cursorposx, cursorposy = pygame.mouse.get_pos()
    cursorx = round(cursorposx/16)*16
    cursory = round(cursorposy/16)*16

    
    for event in pygame.event.get():
        if event.type == pygame.quit :
            running = False
    pygame.draw.rect(screen,(boardr[read],boardg[read],boardb[read]),pygame.Rect(posx-15.625,posy-15.625,16,16))
    pygame.display.flip()
    print(round(cursory/32)+32)

    
