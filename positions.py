import pygame
import random
import numpy

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("balls")

# Initialize positions
velx = [random.randint(-50, 50) for _ in range(10)]  # Ensure circles are within boundaries
vely = [random.randint(1, 2) for _ in range(10)]
posx = [random.randint(10, 490) for _ in range(10)]  # Ensure circles are within boundaries
posy = [random.randint(10, 490) for _ in range(10)]
read = 0

# Main loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update position and draw
    read = read + 1
    if read > 9:
        read = 0
        pygame.display.flip()

        screen.fill((0, 0, 0))  # Clear the screen
        
    # Update position with bounds checking
    velx[read] = velx[read]
    vely[read] = vely[read] + 0.1
    posy[read] += vely[read]
    posx[read] += velx[read]

    velx[read] = velx[read] / 1.005
    vely[read] = vely[read] / 1.005
    # Ensure circles stay within the screen boundaries
    posx[read] = max(10, min(490, posx[read]))
    posy[read] = max(10, min(490, posy[read]))
    
    if posy[read] > 489 :
        vely[read] = 0 - vely[read]
    
    if posx[read] == 490 :
        velx[read] = 0 - velx[read]
    if posx[read] == 10 :
        velx[read] = 0 - velx[read]
    

    # Draw the current circle
    pygame.draw.circle(screen, (255, 255, 255), (posx[read], posy[read]), 10)

    # Update the display
    

    # Control the frame rate
    pygame.time.delay(1)  # Add a short delay to control the speed of the animation

# Quit Pygame
pygame.quit()