import pygame
import random
import math
pygame.init 
screen = pygame.display.set_mode((500,500))
posy = 0
posx = 0
running = True
col = 0
boardr = [random.randint(0, 100) for _ in range(1025)]
boardg = [random.randint(0, 100) for _ in range(1025)]
boardb = [random.randint(0, 100) for _ in range(1025)]
read = 0
cursorx = 0
cursory = 0
cursorposx = 0
cursorposy = 0
cursorx1 = 0
cursory1 = 0

def drawred(x,y):
   boardr[(round(cursorx1/32*2-x)+(round(cursory1/32*2-y)*16*2+33))%1024]= 255
def drawgreen(x,y):
   boardg[(round(cursorx1/32*2-x)+(round(cursory1/32*2-y)*16*2+33))%1024]= 255  
def drawblue(x,y):
   boardb[(round(cursorx1/32*2-x)+(round(cursory1/32*2-y)*16*2+33))%1024]= 255

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
    if read > 1024 :
        read = 0
        if cursory <= 480 :
         boardg[(round(cursory1/32*2)*16*2+33)%1024] = 255
         boardb[round(cursorx1/32*2)+33] = 255
         drawred(1,1)
         drawred(-1,1)
         drawred(-1,-1)
         drawred(0,-1)
         drawred(1,-1)
        if cursory > 480 :
         boardg[(round(cursory1/32*2)*16*2+33)%1024-1] = 255
         boardb[round(cursorx1/32*2)+33] = 255
         boardr[(round(cursorx1/32*2)+(round(cursory1/32*2)*16*2+33))%1024-1] = 255
    cursorposx, cursorposy = pygame.mouse.get_pos()
    cursorx = round(cursorposx/16)*16
    cursory = round(cursorposy/16)*16
    cursorx1 = round(random.randrange(480)/16)*16
    cursory1 = round(random.randrange(480)/16)*16
    
    for event in pygame.event.get():
        if event.type == pygame.quit :
            running = False
    pygame.draw.rect(screen,(boardr[read],boardg[read],boardb[read]),pygame.Rect(posx-15.625,posy-15.625,16,16))
    pygame.display.flip()
    
    
    print()
    
