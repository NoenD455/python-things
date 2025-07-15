import pygame
pygame.init()
screen = pygame.display.set_mode((500,500))
x= 0 
y= 0
c = 0
board = [0] * 2500
board[123] = 1
board[223] = 2
def getb(x,y):
    return  board[(y%50)*50+(x%50)]
def setb(x,y,z):
    board[(y%50)*50+(x%50)] = z
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if c == 0:
        x = x+1
        if x>49:
            y = y+1
            x= 0
        if y > 49:
            y = 0
            x = 0
            c= 1
        
    
    
        if board[y*50+x] == 0:
            pygame.draw.rect(screen,(128,128,0),(x*10,y*10,10,10))
        if board[y*50+x] == 1:
            pygame.draw.rect(screen,(128,0,0),(x*10,y*10,10,10))
        if board[y*50+x] == 2:
            pygame.draw.rect(screen,(128,128,128),(x*10,y*10,10,10))
    if c == 1:
        x = x+1
        if x>49:
            y = y+1
            x= 0
        if y > 49:
            y = 0
            x = 0
            c= 0
        print(board[y*50+x])
        if getb(x,y) == 1 and getb(x,y+1) == 0:
            setb(x,y,0)
            setb(x,y+1,1)
    pygame.display.flip()