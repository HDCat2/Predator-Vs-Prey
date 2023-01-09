import CellInformation as ci
import pygame as pyg
import math

pyg.init()

WIDTH, HEIGHT = 1280, 720

FPS = 60 # frames per second setting
fpsClock = pyg.time.Clock()


screen = pyg.display.set_mode((WIDTH, HEIGHT))


map = ci.Map(WIDTH, HEIGHT, 30, 30, 100, 100, True)

running = True

while running:
    mb = pyg.mouse.get_pressed()
    mousePos = pyg.mouse.get_pos()

    for action in pyg.event.get():
        if action.type == pyg.QUIT:
            running = False
            map.kill()
            quit()

    map.update()
    map.draw(screen)



    for cell in map.predList:
        if cell.selected:
            cell.draw(screen, True)
            print("000")
            print(cell)
            print(cell.tensorVision)


    for cell in map.preyList:
        if cell.selected:
            cell.draw(screen, True)
            print("111")
            print(cell)
            print(cell.tensorVision)


    if mb[0]:
        for cell in map.predList:
            cell.selected = math.dist(mousePos, cell.xyPos) < ci.Cell.CELL_RADIUS

        for cell in map.preyList:
            cell.selected = math.dist(mousePos, cell.xyPos) < ci.Cell.CELL_RADIUS

    pyg.display.flip()
    fpsClock.tick(FPS)

quit()
