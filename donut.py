# Simple pygame program

# Import and initialize the pygame library
import pygame
import math 
pygame.init()
posx = 1
posy = 1
time = 100
cy = 1
cx = 1
count = 1
# Set up the drawing window
screen = pygame.display.set_mode([500, 500])

# Run until the user asks to quit
running = True


while running:
    if count > 99 :
        cx, cy = pygame.mouse.get_pos()
    print(count)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if count > 100 :
        screen.fill((0,0,0))
        count = 0
    pygame.draw.circle(screen, (0, 0, (round((posy%255)/20)*20)%255), (posx, posy), posy/10)
    count = count + 1

    # Flip the display
    pygame.display.flip()
    posx = math.cos(time)*100+250*cx/200
    posy = math.sin(time)*100+250*cy/200
    time = time + 1


# Done! Time to quit.
