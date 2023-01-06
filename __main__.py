import CellInformation as ci
import pygame as pyg

pyg.init()

WIDTH, HEIGHT = 1280, 720

FPS = 60 # frames per second setting
fpsClock = pyg.time.Clock()


screen = pyg.display.set_mode((WIDTH, HEIGHT))

map = ci.Map(WIDTH, HEIGHT, 30, 30, 100, 100, True)

running = True

while running:
    for action in pyg.event.get():
        if action.type == pyg.QUIT:
            running = False
            map.kill()
            quit()

    map.update()
    map.draw(screen)

    pyg.display.flip()
    fpsClock.tick(FPS)

quit()
