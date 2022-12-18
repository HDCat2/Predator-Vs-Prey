from CellInformation import *
from pygame import *

init()

WIDTH, HEIGHT = 1280, 720

screen = display.set_mode((WIDTH, HEIGHT))

map = Map(WIDTH, HEIGHT, 10, 10, 100, 100)

running = True

while running:
    for action in event.get():
        if action.type == QUIT:
            running = False
            map.kill()
            break

    map.update()
    map.draw(screen)

    display.flip()

quit()























