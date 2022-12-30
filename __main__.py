import CellInformation as ci
import pygame as pyg

pyg.init()

WIDTH, HEIGHT = 1280, 720

screen = pyg.display.set_mode((WIDTH, HEIGHT))

map = ci.Map(WIDTH, HEIGHT, 10, 10, 100, 100, True)

running = True

while running:
    for action in pyg.event.get():
        if action.type == pyg.QUIT:
            running = False
            map.kill()
            break

    map.update()
    map.draw(screen)

    pyg.display.flip()

quit()
