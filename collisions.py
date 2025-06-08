import pygame
import math
import time
wait = 0
c = 1
pygame.font.init()
font = pygame.font.SysFont('Impact', 30)
pygame.init()
def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)
screen = pygame.display.set_mode((500,500))
running = True
velx = 0
vely = 0
tempx = velx
tempy = vely
x = 40
y = 250
velx1 = 0
vely1 = 0
tempx1 = velx1
tempy1 = vely1
x1 = 480
y1 = 250
reload = 3
string = "READY " + str(round(reload, 2))
time = 0
mx, my = pygame.mouse.get_pos()
angle_rad = math.atan2(vely, velx)
larium = 0
circle_x = x + 40 * math.cos(angle_rad)
circle_y = y + 40 * math.sin(angle_rad)

#   dont touch
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    time = time + 0.001
    screen.fill((0,0,0))
    pygame.draw.circle(screen,(255,255,0),(x,y),40)
    pygame.draw.circle(screen,(255,0,255),(x1,y1),20)
    
    angle_rad = math.atan2(vely, velx)
    circle_x = x + 40 * -math.cos(larium)
    circle_y = y + 40 * -math.sin(larium)
    pygame.draw.circle(screen,(255,0,0),(circle_x,circle_y),10)

    pygame.draw.circle(screen,(255,32,64),(250,250),wait*100)

    text = font.render(string, 0, ((math.sin(time)*255)%255, (math.sin(time*2)*255)%255, (math.sin(time*3)*255)%255))
    csad = pygame.transform.scale(text,(abs(math.sqrt(abs(velx+velx1)/20)*10+256),abs(math.sqrt(abs(vely+vely1)/2)*10+32)))
    rectfont = text.get_rect()
    rectfont.topleft = (0, 0)

    screen.blit(csad, rectfont)
    
    if larium < angle_rad:
        larium = larium + 0.0025
    if larium > angle_rad:
        larium = larium - 0.0025
    
    pygame.display.update()
    x += velx/100
    y += vely/100
    x1 += velx1/100
    y1 += vely1/100
    tempx1 = velx1
    tempx = velx
    tempy1 = vely1
    tempy = vely
    
    #   collision
    if euclidean_distance((x,y), (x1,y1))<=60:
        x = x + tempx1/100
        velx = tempx1 /reload
        
        x1 = x1 + tempx/100
        velx1 = tempx /1
        
        y = y  + tempy1/100
        vely = tempy1 /reload
        
        y1 = y1 + tempy/100
        vely1 = tempy /1

    if euclidean_distance((circle_x,circle_y), (x1,y1))<=30: 
        string = "game over"
        c = 0
    if c == 0:
        if wait < 5:
            wait = wait + 0.01
            string = "game over"
            print(wait)
        string = "game over"
        if wait > 4:
            running = False
            string = "game over"
            running = False
            print(3)
        print(13)
    
    mx, my = pygame.mouse.get_pos()
    #   bound
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


    
    #   tiny
    
    keys = pygame.key.get_pressed()

    
    if keys[pygame.K_LEFT]:
        velx1 = velx1 - 0.025
    elif keys[pygame.K_RIGHT]:
        velx1 = velx1 + 0.025

    if keys[pygame.K_UP]:
        vely1 = vely1 - 0.025
    elif keys[pygame.K_DOWN]:
        vely1 = vely1 + 0.025
    
    #   BIF
    if keys[pygame.K_a]:
        velx = velx - 0.025
    elif keys[pygame.K_d]:
        velx = velx + 0.025

    if keys[pygame.K_w]:
        vely = vely - 0.025
    elif keys[pygame.K_s]:
        vely = vely + 0.025
    
    if keys[pygame.K_SPACE] and math.floor(reload) == 3:
        reload = 0.5
    
    # damp

    velx = velx / 1.001
    velx1 = velx1 / 1.001
    vely = vely / 1.001
    vely1 = vely1 / 1.001
    if reload < 3:
        reload = reload * 1.000125
    if math.floor(reload) == 3 and wait == 0:
        string = "READY " + str(round(reload, 2))
    elif not math.floor(reload) == 3 and wait == 0:
        string = "CHARGING " + str(round(reload, 2))
    
    

